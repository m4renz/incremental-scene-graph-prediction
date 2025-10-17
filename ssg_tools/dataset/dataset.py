from __future__ import annotations
import torch.utils.data as data
import random
import torch
import numpy as np
import logging
from pathlib import Path
import dataclasses
from torch_geometric.data import HeteroData
from lightning.pytorch.utilities import AttributeDict
from typing import Sequence, Optional, Literal
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG
from ssg_tools.dataset.util.point_augmentations import augment_points, normalize_tensor, point_descriptor, zero_mean
from ssg_tools.dataset.util.graph import (sample_neighbors, 
                                          sample_edges_from_nodes, 
                                          make_complete,
                                          visualize_graph)
from ssg_tools.dataset.util.weights import compute_weights
from ssg_tools.dataset.mesh import get_stacked_points
from ssg_tools.dataset.structured_array import join_record_arrays
from ssg_tools.dataset.util.random_drop import random_drop
import networkx as nx
from cachetools import Cache, cachedmethod
from copy import deepcopy

log = logging.getLogger(__name__)

__all__ = ["SceneGraphDataset", "SceneGraphSGPNDataset"]

class SceneGraphDataset(data.Dataset):
    def __init__(self, 
                 path: Path,
                 hparams: AttributeDict,
                 split: Optional[str | Sequence[str] | Sequence[bool]] = None, 
                 train_mode: bool = True,
                 cache: bool = True):
        super().__init__()
        self.hparams = hparams

        # open the dataset to access the data
        self.dataset = DatasetInterface3DSSG(path, scan_ids=split)

        self.train_mode = train_mode

        if cache:
            cachesize = self.dataset.nscans
        else:
            cachesize = 1
        self.cachesize = cachesize
        self.scene_graph_cache = Cache(cachesize)
        self.neighbors_graph_cache = Cache(cachesize)

        #if self.train_mode:
        #    log.info("Computing class weights from the scene graph...")
        #    self.node_weights, self.edge_weights = self._compute_weights()

    def _sample_nodes(self, scene_graph: nx.MultiDiGraph) -> list[int]:
        """Samples a set of nodes and their neighbors from the scene graph.
        Args:
            scene_graph: the scene graph to sample the nodes from
        """
        assert 0 not in scene_graph.nodes # 0 is the invalid instance

        nodes = list(scene_graph.nodes)
        if self.hparams.runtime_sampling.enabled and self.train_mode:
            
            sample_n_hops = self.hparams.runtime_sampling.num_hops
            sample_n_nodes = self.hparams.runtime_sampling.num_nodes
            if sample_n_hops == 0 or sample_n_nodes == 0:
                filtered_nodes = nodes # use all nodes
            else:
                filtered_nodes = sample_neighbors(scene_graph, nodes, sample_n_hops, sample_n_nodes)

            assert 0 not in filtered_nodes
            nodes = list(filtered_nodes)

            max_num_nodes = self.hparams.runtime_sampling.max_nodes

            if max_num_nodes > 0 and len(nodes) > max_num_nodes:
                nodes = random_drop(nodes, max_num_nodes)

        sampled_graph = scene_graph.subgraph(nodes)

        return sampled_graph #nodes, node_to_index, index_to_node
    
    def _sample_neighbor_edges(self, neighbor_graph: nx.Graph): #, selected_nodes: list[int], instance_id_to_index: dict) -> list[int]:
        if self.hparams.runtime_sampling.enabled:
            if self.hparams.runtime_sampling.edge_sampling_mode == "fully_connected":
                # use dense edges (fully connected)
                neighbor_graph = make_complete(neighbor_graph)
            elif self.train_mode:
                # sample edges from neighbors
                max_neighbors = self.hparams.runtime_sampling.edge_sampling_max_neighbors
                if max_neighbors > 0:
                    neighbor_graph = sample_edges_from_nodes(neighbor_graph, max_edges_per_node=max_neighbors)
        else:
            # use dense edges (fully connected)
            neighbor_graph = make_complete(neighbor_graph)
        return neighbor_graph

    def _generate_relationship_tensors(self, scene_graph: nx.MultiDiGraph, neighbor_graph_directed: nx.DiGraph, instance_id_to_index: dict):
        num_relationship_classes = len(self.dataset.edge_classes())
        
        neighbor_edges = neighbor_graph_directed.edges()
        if self.hparams.multi_relations:
            gt_relationships = torch.zeros(len(neighbor_edges), num_relationship_classes, dtype=torch.float)
        else:
            gt_relationships = torch.zeros(len(neighbor_edges), dtype=torch.long)

        edge_indices = torch.empty((len(neighbor_edges), 2), dtype=torch.long)
        edges_with_gt = []

        scene_graph_edges = scene_graph.edges()

        # the neighbor graph is not directed so check both directions in the undirected scene graph
        for i, (subject_instance, object_instance) in enumerate(neighbor_edges):
            subject_index = instance_id_to_index[subject_instance]
            object_index = instance_id_to_index[object_instance]
            key = (subject_instance, object_instance)

            edge_indices[i, 0] = subject_index
            edge_indices[i, 1] = object_index

            if key in scene_graph_edges:
                if self.hparams.multi_relations:
                    # iterate all relationships for this key which might be more than one
                    for x in scene_graph[key[0]][key[1]].keys():
                        gt_relationships[i, x] = 1 # one hot encoding per edge
                else:
                    key_edges = scene_graph[key[0]][key[1]]
                    if len(key_edges) != 1:
                        raise ValueError(f"Only singular relationships supported. Got {len(key_edges)}.")
                    gt_relationships[i] = next(iter(key_edges)) # get the only relationship from the edges
                edges_with_gt.append(i)
        
        return gt_relationships, edge_indices, edges_with_gt
    
    def _drop_edges(self, 
                    relationships_gt: torch.Tensor, 
                    edge_indices: torch.Tensor, 
                    edge_index_with_gt: list):
        if len(edge_indices) == 0:  # no edges -> just return
            return relationships_gt, edge_indices

        all_indices = set(range(edge_indices.shape[0]))
        edge_index_without_gt = list(all_indices.difference(edge_index_with_gt))
        if len(edge_index_without_gt) == 0:
            return relationships_gt, edge_indices  # all edges have ground truth so all edges are needed

        if self.train_mode:
            edge_index_without_gt = random_drop(edge_index_without_gt, self.hparams.runtime_sampling.drop_edges_factor)

        num_edges = len(edge_index_without_gt) + len(edge_index_with_gt)
        max_edges = self.hparams.runtime_sampling.max_edges
        if max_edges > 0 and num_edges > max_edges:
            # only process with max_num_edge is set, and the total number is larger
            # and the edges with gt is smaller
            if len(edge_index_with_gt) < max_edges:
                n_to_sample = max_edges - len(edge_index_with_gt)
                # sample the missing number of edges from the non gt edges
                edge_index_without_gt = np.random.choice(
                    edge_index_without_gt, n_to_sample, replace=False).tolist()
            else:
                edge_index_without_gt = []

        final_edge_indices = list(edge_index_with_gt) + list(edge_index_without_gt)
        edge_indices = edge_indices[final_edge_indices]

        relationships_gt = relationships_gt[final_edge_indices]

        return relationships_gt, edge_indices
    
    @cachedmethod(lambda self: self.scene_graph_cache)
    def _get_scene_graph(self, scan_id):
        scan = self.dataset.scan(scan_id)
        return scan.scene_graph(raw=False)
    
    @cachedmethod(lambda self: self.neighbors_graph_cache)
    def _get_neighbor_graph(self, scan_id):
        scan = self.dataset.scan(scan_id)
        return scan.neighbors_graph()

    def __getitem__(self, index):
        #scan_id = snp.unpack(self.scans, index)  # self.scans[idx]

        # get SG data
        scan = self.dataset.scan(index)
        
        # shortcuts
        scene_graph = self._get_scene_graph(index) #scan.scene_graph(raw=False)
        neighbor_graph = self._get_neighbor_graph(index) #scan.neighbors_graph()

        # sample node instances from the scene graph
        #filtered_instances, instance_id_to_index, index_to_instance_id = self._sample_nodes(scene_graph)
        scene_graph = self._sample_nodes(scene_graph)
        neighbor_graph = neighbor_graph.subgraph(scene_graph.nodes())

        # sample neighboring edges
        neighbor_graph = self._sample_neighbor_edges(neighbor_graph)
        if len(neighbor_graph.nodes()) != len(scene_graph.nodes()):
            #log.error("Nodes removed from neighbor graph in sampling step. This should not happen.")
            scene_graph = scene_graph.subgraph(neighbor_graph.nodes())

        # Generate mapping from selected entities to the ground truth entities (for evaluation)
        # Save the mapping in edge_index format to allow PYG to rearrange them.

        # Collect GT entity list
        instances_gt_tmp = list(scene_graph.nodes(data="rio27_enc"))

        # reorder the nodes to introduce variety
        if self.hparams.shuffle:
            random.shuffle(instances_gt_tmp)

        instances_gt, instances_labels_gt = zip(*instances_gt_tmp)

        # mapping to keep track of the indices
        instance_id_to_index = {id: i for i, id in enumerate(instances_gt)}
        #index_to_instance_id = {i: id for i, id in enumerate(instances_gt)}

        index_gt_to_index = [[index_gt, instance_id_to_index[instance_gt]] for index_gt, instance_gt in enumerate(instances_gt)]
        index_gt_to_index = torch.tensor(index_gt_to_index, dtype=torch.long).t().contiguous()

        # map all edges to their repective indices and create edge-index formatted tensors
        scene_graph_edges = np.array([(instance_id_to_index[s], 
                                       instance_id_to_index[o], 
                                        v) for s, o, v in scene_graph.edges(keys=True)])
        
        # build torch tensors from the edge and node indices to connect them later in the graph
        edge_indices_gt = torch.from_numpy(scene_graph_edges[:, :2]).t().contiguous()
        edge_labels_gt = torch.from_numpy(scene_graph_edges[:, 2].squeeze())

        # convert to neighbor graph into a directed representation to generated the relationship tensors from both graphs
        neighbor_graph_directed = nx.DiGraph(neighbor_graph)

        # sample edges
        relationships_gt, edge_indices, edges_with_gt = self._generate_relationship_tensors(scene_graph, neighbor_graph_directed, instance_id_to_index)

        # drop edges to fit in memory
        relationships_gt, edge_indices = self._drop_edges(relationships_gt, edge_indices, edges_with_gt)

        # transpose the edge indices for pyg
        edge_indices = edge_indices.t().contiguous()

        # collect attributes for nodes
        #TODO: for inseg or any other oversegmentations the segment instance should be converted back to the GT instances
        #TODO: right now this is not necessary since we only operate on the ground truth input data
        #inst_indices = list(instance_id_to_index.keys())

        # convert  everything to tensors
        tensor_classes_gt = torch.from_numpy(np.asarray(instances_labels_gt))
        tensor_instance_ids = torch.from_numpy(np.asarray(instances_gt))

        # Gather output in HeteroData
        output = HeteroData()
        # store the id as graph data
        output['scan_id'] = scan.scan_id  # str

        # node features
        output['node'].num_nodes = tensor_classes_gt.shape[0]# = torch.zeros([tensor_classes_gt.shape[0], 1])  # dummy # TODO ist this required?
        output['node'].y = tensor_classes_gt
        output['node'].instance_id = tensor_instance_ids

       # node ground truth features
        #output['node_gt'].x = torch.zeros(
        #    [len(tensor_classes_gt), 1])  # dummy # TODO ist this required?
        #output['node_gt'].class_id = tensor_classes_gt if tensor_classes_gt.shape[0] > 0 else torch.zeros([tensor_classes_gt.shape[0], 1])  # dummy # TODO ist this required?
        
        # edge features for ground truth nodes
        #output['node_gt', 'to', 'node'].edge_index = index_gt_to_index
        #output['node_gt', 'to', 'node_gt'].class_id = edge_labels_gt if edge_labels_gt.shape and edge_labels_gt.shape[0] > 0 else torch.zeros([tensor_classes_gt.shape[0], 1])  # dummy
        #output['node_gt', 'to', 'node_gt'].edge_index = edge_indices_gt

        # edge features for nodes
        output['node', 'to', 'node'].edge_index = edge_indices
        output['node', 'to', 'node'].y = relationships_gt

        return output

    def __len__(self):
        return self.dataset.nscans

    def class_names(self) -> list[str]:
        return self.dataset.node_classes()
    
    def relationship_names(self) -> list[str]:
        return self.dataset.edge_classes()
    
    def _compute_weights(self):
        fully_connected = self.hparams.runtime_sampling.edge_sampling_mode == "fully_connected"
        if fully_connected:
            edge_mode = 'fully_connected'
        else:
            edge_mode = 'nn'
        
        node_weights, edge_weights, node_occurrences, edge_occurrences = compute_weights(self.dataset, 
                                                                                         edge_mode=edge_mode,
                                                                                         normalize=self.hparams.normalize_weights ,
                                                                                         use_bce=self.hparams.multi_relations)
        node_weights = torch.from_numpy(node_weights)
        edge_weights = torch.from_numpy(edge_weights)
        return node_weights, edge_weights

@dataclasses.dataclass
class PointCloud:
    points: np.ndarray
    labels: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None

    def get_stacked(self) -> np.ndarray:
        points = self.points.astype(np.float32).view([("points", np.float32, 3)])
        to_join = [points]
        if self.colors is not None:
            colors = colors[:, :3]
            # scale to [-1.0, 1.0]
            colors = (colors.astype(np.float32) / 255 * 2.0 - 1).view([("colors", np.float32, 3)])
            to_join.append(colors)
        if self.normals is not None:
            normals = self.normals.astype(np.float32).view([("normals", np.float32, 3)])
            to_join.append(normals)
        return join_record_arrays(to_join)


class SceneGraphSGPNDataset(SceneGraphDataset):
    def __init__(self, 
                 path: Path,
                 hparams: AttributeDict, 
                 split: Optional[str | Sequence[str] | Sequence[bool]] = None, 
                 train_mode: bool = True,
                 cache: bool = False):
        super().__init__(path, hparams, split=split, train_mode=train_mode, cache=cache)
        assert self.hparams.point_source
        self.point_dim = 3 + 3 * sum((self.hparams.point_colors, self.hparams.point_normals))
        self.point_cache = Cache(self.cachesize)
    
    def _sample_node_points(self, points: np.ndarray, instance_labels: np.ndarray, filtered_instances: Sequence[int], padding=0.2, epsilon=1e-12):
        bboxes = []
        npoints = self.hparams.samples_node

        points = points.view(np.float32).reshape(-1, self.point_dim)
        obj_points = torch.zeros([len(filtered_instances), npoints, self.point_dim])
        descriptor = torch.zeros([len(filtered_instances), 11])
        instance_masks = np.empty((len(filtered_instances), points.shape[0]), dtype=np.uint8)
        dummy_points = False
        for i, instance_id in enumerate(filtered_instances):
            mask = instance_labels == int(instance_id)
            instance_masks[i] = mask
            obj_pointset = points[mask]
            if obj_pointset.shape[0] == 0:
            #    # use dummy points if there are no points for this specifiy instance
            #    #raise ValueError(f"Invalid pointset. No points found for instance {instance_id}. This indicates something went wrong #during pre-processing.")
                obj_pointset = np.zeros((self.hparams.samples_node, self.point_dim), dtype=np.float32)
                min_box = np.zeros(3, dtype=np.float32)
                max_box = min_box.copy()
                dummy_points = True
            else:
                min_box = np.min(obj_pointset[:, :3], axis=0) - padding
                max_box = np.max(obj_pointset[:, :3], axis=0) + padding
                choice = np.random.choice(len(obj_pointset), npoints, replace=len(obj_pointset) < npoints)
                obj_pointset = obj_pointset[choice]

            bboxes.append([min_box, max_box])
            descriptor[i] = point_descriptor(obj_pointset)
            descriptor[i] += epsilon
            obj_pointset = torch.from_numpy(obj_pointset)
            #TODO: cache this?
            if not dummy_points:
                obj_pointset[:, :3] = normalize_tensor(obj_pointset[:, :3])
            obj_points[i] = obj_pointset
        obj_points = obj_points.permute(0, 2, 1)
        return obj_points, descriptor, bboxes, instance_masks
    
    def _sample_relationship_points(self, 
                                    points: np.ndarray, 
                                    instances: np.ndarray, 
                                    graph_instances: list, 
                                    bboxes: list, 
                                    edge_indices: list,
                                    instance_masks: np.ndarray):
        relationships_points = list()
        points = points.view(np.float32).reshape(-1, self.point_dim)
        point_coords = points[:, :3]

        npoints_union = self.hparams.samples_relationship
        points4d = np.pad(points, ((0,0), (0, 1)))

        #points4d = np.concatenate([points, ], 1)
        for index1, index2 in edge_indices.t():
            #mask1 = (instances == int(graph_instances[index1])).astype(np.int32) * 1
            #mask2 = (instances == int(graph_instances[index2])).astype(np.int32) * 2
            #mask_ = np.expand_dims(mask1 + mask2, 1)

            # write the mask into the pointset
            points4d[:, -1] = instance_masks[index1] * 1 + instance_masks[index2] * 2
            
            bbox1 = bboxes[index1]
            bbox2 = bboxes[index2]

            min_box = np.minimum(bbox1[0], bbox2[0])
            max_box = np.maximum(bbox1[1], bbox2[1])

            filter_mask = np.logical_and(np.all(point_coords >= min_box, axis=1), np.all(point_coords <= max_box, axis=1))

            # filter_mask = (points[:, 0] > min_box[0]) * (points[:, 0] < max_box[0]) \
            #     * (points[:, 1] > min_box[1]) * (points[:, 1] < max_box[1]) \
            #     * (points[:, 2] > min_box[2]) * (points[:, 2] < max_box[2])
            #points4d = np.concatenate([points, mask_], 1)
            pointset = points4d[filter_mask, :]

            # sample from the common points of both bounding boxes
            choice = np.random.choice(
                len(pointset), npoints_union, replace=len(pointset) < npoints_union)
            pointset = pointset[choice, :]
            pointset = torch.from_numpy(pointset.astype(np.float32))

            # normalize
            pointset[:, :3] = zero_mean(pointset[:, :3], False)
            relationships_points.append(pointset)

        if self.train_mode:
            try:
                relationships_points = torch.stack(relationships_points, 0)
            except:
                relationships_points = torch.zeros([0, npoints_union, 4])
        else:
            if len(relationships_points) == 0:
                # sometimes there will be no edge because of only 1 node existing
                # this is due to the label mapping/filtering process in the data generation
                relationships_points = torch.zeros([0, npoints_union, 4])
            else:
                relationships_points = torch.stack(relationships_points, 0)
        relationships_points = relationships_points.permute(0, 2, 1)
        return relationships_points
    
    @cachedmethod(lambda self: self.point_cache)
    def _get_scan_points(self, scan_id, point_source, colors: bool, normals: bool):
        scan = self.dataset.scan(scan_id)
        if point_source == "raw":
            #data = scan.color_mesh()
            #points = data.vertices
            label_data = scan.label_mesh()
            points = label_data.vertices
            #from scipy.spatial import cKDTree
            #kdt = cKDTree(label_data.vertices)
            #_, ii = kdt.query(data.vertices, k=1)
            labels = label_data.vertex_attributes["labels"]#[ii.ravel()]

            colors = data.vertex_attributes["colors"] if colors else None
            normals = data.vertex_attributes["normals"] if normals else None

            point_cloud = PointCloud(points, labels, colors, normals)
        elif point_source == "sampled":
            data = scan.sampled_points()
            points = data.vertices
            labels = data.vertex_attributes["labels"]
            colors = data.vertex_attributes["colors"] if colors else None
            normals = data.vertex_attributes["normals"] if normals else None

            point_cloud = PointCloud(points, labels, colors, normals)
        else:
            raise ValueError(f"Invalid point source: {point_source}")
        
        return point_cloud.get_stacked(), point_cloud.labels.squeeze()
    
    def __getitem__(self, index):
        output = super().__getitem__(index)

        scan = self.dataset.scan(index)
        points, instance_labels = self._get_scan_points(index, self.hparams.point_source, self.hparams.point_colors, self.hparams.point_normals)
        assert points.shape[0] == instance_labels.shape[0]

        if self.hparams.point_augmentation and self.train_mode:
            points = augment_points(points)

        graph_instances = output['node'].instance_id

        try:
            # random sample points for the nodes
            obj_points, descriptor, bboxes, instance_masks = self._sample_node_points(points, instance_labels, graph_instances)
        except ValueError as e:
            raise ValueError(f"Error during processing of scan {scan.scan_id}.") from e
        
        assert not obj_points.isnan().any()
        # random sample point for the edges
        if self.hparams.edge_data_type == 'points':
            edge_indices = output['node', 'to', 'node'].edge_index
            relationship_points = self._sample_relationship_points(points, 
                                                                instance_labels, 
                                                                graph_instances, 
                                                                bboxes, 
                                                                edge_indices,
                                                                instance_masks)
            output['node', 'to', 'node'].points = relationship_points
        
        output['node'].points = obj_points
        output['node'].descriptor = descriptor

        return output




class SceneGraphSSGPointDataset(SceneGraphDataset):
    def __init__(self, 
                 path: Path,
                 hparams: AttributeDict, 
                 split: Optional[str | Sequence[str] | Sequence[bool]] = None, 
                 train_mode: bool = True,
                 cache: bool = False):
        super().__init__(path, hparams, split=split, train_mode=train_mode, cache=cache)
        assert self.hparams.point_source
        self.point_dim = 3 + 3 * sum((self.hparams.point_colors, self.hparams.point_normals))
        self.point_cache = Cache(self.cachesize)


    def _sample_points(self, points, instance_labels):
        if instance_labels.shape[0] > self.hparams.point_samples:          # self.max_npoint is set to 4096
            sampling_ratio = self.max_npoint / instance_labels.shape[0]
            all_idxs = []                                       # scene-level instance_idx of points being selected this time
            for iid in np.unique(instance_labels):              # sample points on object-level
                indices = (instance_labels == iid).nonzero()[0]     # locate these points of a specific instance_idx
                end = int(sampling_ratio * len(indices)) + 1        # num_of_points_to_be_sampled + 1
                np.random.shuffle(indices)                          # uniform sampling among each object instance
                selected_indices = indices[ :end]                   # get the LuckyPoints that get selected in fortune's favourite
                all_idxs.extend(selected_indices)                   # append them to the scene-level list
            valid_idxs = np.array(all_idxs)
        else:
            valid_idxs = np.ones(instance_labels.shape, dtype=bool) # no sampling is required
        return valid_idxs

    def __getitem__(self, index):
        output = super().__getitem__(index)

        scan = self.dataset.scan(index)
        points, instance_labels = self._get_scan_points(index, self.hparams.point_source, self.hparams.point_colors, self.hparams.point_normals)
        assert points.shape[0] == instance_labels.shape[0]

        # sample the point cloud first


        if self.hparams.point_augmentation and self.train_mode:
            points = augment_points(points)

        graph_instances = output['node'].instance_id

        try:
            # random sample points for the nodes
            obj_points, descriptor, bboxes, instance_masks = self._sample_node_points(points, instance_labels, graph_instances)
        except ValueError as e:
            raise ValueError(f"Error during processing of scan {scan.scan_id}.") from e
        
        assert not obj_points.isnan().any()
        # random sample point for the edges
        if self.hparams.edge_data_type == 'points':
            edge_indices = output['node', 'to', 'node'].edge_index
            relationship_points = self._sample_relationship_points(points, 
                                                                instance_labels, 
                                                                graph_instances, 
                                                                bboxes, 
                                                                edge_indices,
                                                                instance_masks)
            output['node', 'to', 'node'].points = relationship_points
        
        output['node'].points = obj_points

        #if self.hparams.edge_descriptor_type == "points":
        #    output['node'].descriptor = descriptor

        return output



