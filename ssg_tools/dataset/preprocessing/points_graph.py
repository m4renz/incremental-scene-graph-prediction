from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG, ScanInterface
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.logging import logging_redirect_tqdm
from functools import partial
import logging
from ssg_tools.dataset.mesh import mesh_get_labels, mesh_set_labels, get_stacked_points
from ssg_tools.dataset.points import sample_by_instance_id, normalize_points, filter_invalid_instances, farthest_point_sample
import numpy as np
import h5py
from pathlib import Path

log = logging.getLogger(__name__)


def _process(scan: ScanInterface, nsample: int = 1024, normalize: bool = True):
    data = scan.sampled_points()
    points = get_stacked_points(data)
    instance_labels = mesh_get_labels(data)
    
    scene_graph = scan.scene_graph()
    scene_graph_nodes = scene_graph["nodes"]

    valid_mask = filter_invalid_instances(instance_labels, set(list(scene_graph_nodes.keys())))

    points = points[valid_mask]
    instance_labels = instance_labels[valid_mask]
    
    node_points = np.full((len(scene_graph_nodes), nsample), fill_value=np.nan, dtype=points.dtype)
    instance_id_to_index = {}
    node_num_points = []
    node_pairs = set()
    for i, (instance_id, node) in enumerate(scene_graph_nodes.items()):
        points_instance_mask = instance_labels == instance_id
        # TODO: do farthest point sampling here?
        points_tmp = points[points_instance_mask]
        farthest_samples = farthest_point_sample(points_tmp, nsample=nsample)
        points_tmp = points_tmp[farthest_samples]
        if normalize:
            points_tmp = normalize_points(points_tmp)
        node_points[i, :points_tmp.shape[0]] = points_tmp
        node_num_points.append(points_tmp.shape[0])
        instance_id_to_index[instance_id] = i
        
        # add all the pairs
        for neighbor in node["neighbors"]:
             node_pairs.add((instance_id, neighbor))

    edge_points = np.empty((len(node_pairs), nsample), dtype=points.dtype)
    index_to_pair = {}
    for i, (subject, object) in enumerate(node_pairs):
        index_subject = instance_id_to_index[subject]
        index_object = instance_id_to_index[object]
        points_fused = np.concatenate((node_points[index_subject, ], node_points[index_object]), axis=0)
        # sample again from the fused group
        farthest_samples = farthest_point_sample(points_fused, nsample=nsample)
        points_fused = points_fused[farthest_samples]
        edge_points[i] = points_fused
        index_to_pair[i] = (subject, object)
    
    return node_points, instance_id_to_index, edge_points, index_to_pair

    

def points_graph(dataset: DatasetInterface3DSSG,
                 overwrite: bool = False):
    config = dataset.config

    output_path = Path(config.points_graph.path_points_graph)

    nsample = config.get("points_graph.npoints_sample", 4096)
    normalize = config.get("points_graph.normalize", True)

    if output_path.exists():
         if overwrite:
              output_path.unlink()
         else:
              log.warning("File %s exists. Skipping...")
              return

    with h5py.File(output_path, 'w') as f:
        with logging_redirect_tqdm():
            pbar = tqdm(dataset.scans(), total=dataset.nscans)
            for scan in pbar:
                scan_grp = f.create_group(scan.scan_id)
                node_points, instance_id_to_index, edge_points, index_to_pair = _process(scan, nsample=nsample, normalize=normalize)
                # store all the node points in a single dataset
                nodes_dset = scan_grp.create_dataset("node_points", data=node_points, chunks=(1, nsample))
                for instance_id, index in instance_id_to_index.items():
                    nodes_dset.attrs[str(instance_id)] = index

                # store all the edge points in a single dataset
                edge_dset = scan_grp.create_dataset("edge_points", data=edge_points, chunks=(1, nsample))
                for index, pair in index_to_pair.items():
                     edge_dset.attrs[str(index)] = pair