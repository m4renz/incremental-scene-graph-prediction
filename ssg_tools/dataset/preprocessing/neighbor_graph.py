from __future__ import annotations
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG
import logging
from typing import Literal, Optional
import numpy as np
from tqdm_loggable.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from ssg_tools.dataset.mesh import mesh_get_labels
from scipy.spatial import cKDTree
#from ssg_tools.config import Config
import json

log = logging.getLogger(__name__)


def find_neighbors(points: np.ndarray, 
                   labels: np.ndarray, 
                   receptive_field: float = 0.5,
                   search_method: Literal['bbox', 'knn'] = 'bbox',
                   labels_unique: Optional[np.ndarray] = None, 
                   filtered_labels: Optional[list[int]] = None) -> dict[int, list[int]]:
    """Find the neighbors between each segment within a point cloud.

    Args:
        points: Array of shape (N, 3) with the points to search the neighbors in.
        labels: Array of shape (N,) with the labels for each point in points. 
        selected_labels: If not None, only the intersection between this an the selected labels are considered. Defaults to None.
    Returns:
        Dict containing the neighbors for each segment in labels.
    """

    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    labels = np.asarray(labels).ravel()

    label_ids = labels_unique or np.unique(labels)
    
    # use only the common subset of selected and actual labels
    if filtered_labels is not None:
        label_ids = set(label_ids.tolist())
        label_ids = np.array(list(label_ids.intersection(filtered_labels)))

    # Get all segments points and bounding boxes
    segment_points = {idx: points[np.where(labels==idx)] for idx in label_ids}
    segment_bboxes = {idx: (segment_points[idx].min(axis=0) - receptive_field, segment_points[idx].max(axis=0) + receptive_field) for idx in label_ids}

    segs_neighbors = dict()
    if search_method == 'bbox':
        for seg_idx in label_ids:
            bbox = segment_bboxes[seg_idx]
            segment_nb = segs_neighbors[int(seg_idx)] = []
            for seg_target_idx in label_ids:            
                if seg_idx == seg_target_idx: continue # skip same objects
                bbox_target = segment_bboxes[seg_target_idx]
                if np.any(bbox[0] > bbox_target[1]) or np.any(bbox_target[0] > bbox[1]): 
                    # no intersection
                    continue                
                segment_nb.append(int(seg_target_idx))

    elif search_method == 'knn':
        # search neighbor for each segments using a kdtree
        segment_kdtrees = {idx: cKDTree(segment_points[idx]) for idx in label_ids}
        # helper function
        def find_knn(seg_idx: int, trees: dict, bboxes: dict, search_radius: float):
                tree = trees[seg_idx]
                bbox = bboxes[seg_idx]

                neighbors = set()
                for target_idx, target_tree in trees.items():
                    if target_idx == seg_idx or target_idx in neighbors: continue

                    bbox_target = bboxes[target_idx]
                    if np.any(bbox[0] > bbox_target[1]) or np.any(bbox_target[0] > bbox[1]): 
                        continue # skip if there is not overlap in the bounding boxes
                    
                    npairs = tree.count_neighbors(target_tree, search_radius)
                    if npairs > 0:
                        neighbors.add(int(target_idx))
                return neighbors
        
        for seg_idx in label_ids:    
            neighbors = find_knn(seg_idx, segment_kdtrees, segment_bboxes, self.receptive_field)
            neighbors = [n for n in neighbors]
            segs_neighbors[int(seg_idx)] = neighbors

    return segs_neighbors


def neighborhood_graph(dataset: DatasetInterface3DSSG,
                       receptive_field: float = 0.5,
                       search_method: str = "bbox",  
                       overwrite: bool = False):
    
    #config = dataset.config

    #random_seed = config.random_seed
    #set_random_seed(random_seed)    

    ## for est. seg., add "same part" in the case of oversegmentation.
    #if segment_type != "ground_truth": 
    #    target_relationships.append(str(config.defines.name_same_part))
    scene_graphs = {}
    with logging_redirect_tqdm():
        for scan in tqdm(dataset.scans(), total=dataset.nscans, desc="Neighborhood graph"):
            scan_id = scan.scan_id
            scene_graph = scan.scene_graph(raw=True)
            log.debug("Processing scan %s", scan_id)

            # load gt
            labels_mesh = scan.label_mesh()
            points = labels_mesh.vertices
            labels = mesh_get_labels(labels_mesh, label_type='segment').ravel()
            assert points.shape[0] == labels.shape[0] # ensure the mesh is not processed as the labels are read from the raw data and not mapped to the mesh itself

            scene_nodes = scene_graph["nodes"]

            first_node = next(iter(scene_nodes.values()))
            if "neighbors" in first_node and not overwrite:
                log.info("neighbors already generated for scan %s. Skipping...", scan_id)
                continue
            
            filtered_labels = list(scene_nodes.keys())
            segment_neighbors = find_neighbors(points, labels, search_method=search_method, receptive_field=receptive_field, filtered_labels=filtered_labels)

            # process nodes
            for object_id, object_info in scene_nodes.items():
                neighbors = segment_neighbors.get(object_id, [])
                # write the neighbors into the nodes section of the scene graph
                object_info["neighbors"] = neighbors

            scene_graphs[scan.scan_id] = scene_graph

    output_path = dataset.path_scenegraph_data


    scene_graph_data = {"scenes": scene_graphs, 
                        "node_classes": dataset.node_classes(),
                        "edge_classes": dataset.edge_classes()}
    # export the scene graph again
    log.info("Saving modified scene graph...")
    with open(output_path, 'w') as f:
        json.dump(scene_graph_data, f, indent=4)
    log.info("done.")

# def main():
#     from ssg_tools import ConfigArgumentParser, init_logging

#     parser = ConfigArgumentParser(
#         description='Generate the neighborhood graph from the raw 3DSSG data.')
#     parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing files.')

#     args = parser.parse_args()

#     config = args.config
#     init_logging(config)

#     dataset = DatasetInterface3DSSG(config)
#     neighborhood_graph(dataset, overwrite=args.overwrite)

# if __name__ == "__main__":
#     main()


