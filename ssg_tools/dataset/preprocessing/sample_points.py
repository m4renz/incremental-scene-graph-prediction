from __future__ import annotations
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG, ScanInterface
#from ssg_tools import Config
import numpy as np
from tqdm_loggable.auto import tqdm
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.logging import logging_redirect_tqdm
from ssg_tools.dataset.preprocessing.subprocess import run
from functools import partial
import logging
from scipy.spatial import cKDTree
from .sampling import sample_mesh_points
from ssg_tools.dataset.mesh import mesh_get_labels, mesh_set_labels

log = logging.getLogger(__name__)


def validate_sampled_points(scan: ScanInterface, points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scene_graph = scan.scene_graph(raw=False)
    graph_labels = list(scene_graph.nodes())
    
    unique_labels = set(np.unique(labels))
    diff = list(unique_labels.difference(graph_labels))
    ninvalid = np.sum(labels <= 0)
    if len(diff) > 0 or ninvalid > 0:
        valid_indices = labels > 0
        for diff_value in diff:
            log.info("Filtering invalid instance %d from scan %s", diff_value, scan.scan_id)
            valid_index_current = labels != diff_value
            valid_indices *= valid_index_current
        # keep only the valid points
        points = points[valid_indices]
        labels = labels[valid_indices]
    return points, labels


def _process(scan: ScanInterface, density: int = 10000, overwrite: bool = False) -> None:
    color_mesh = scan.color_mesh()
    points_filename = scan.sampled_points_filename()
    #if scan.scan_id == '0cac75b7-8d6f-2d13-8cb2-0b4e06913140':
    #    print("!!!!!!")
    #    id_of_interest = 130167
    
    if not points_filename.exists() or overwrite:
        labels_mesh = scan.label_mesh()
        point_labels = mesh_get_labels(labels_mesh).squeeze()#[indices]
        points_sampled, new_point_labels = sample_mesh_points(color_mesh, labels=point_labels, density=density, sample_color=True)

        # load the label mesh and derive the labels
        #knn = cKDTree(labels_mesh.vertices)
        #_, indices = knn.query(points_sampled.vertices, k=2)
        #indices = indices[:, 0].squeeze()
        #point_labels = mesh_get_labels(labels_mesh)#[indices]
        #point_labels_new = point_labels[indices]
        # assert all labels from original mesh are preserved in the sampled points
        #assert np.all(np.unique(point_labels) == np.unique(new_point_labels))
        points_sampled, new_point_labels = validate_sampled_points(scan, points_sampled, new_point_labels)
        mesh_set_labels(points_sampled, new_point_labels)

        points_sampled.export(points_filename)

def sample_points(dataset: DatasetInterface3DSSG,
                  nworkers: int = 0,
                  density: int = 10000,
                  overwrite: bool = False) -> None:
    
    #density = dataset.config.get("dataset.point_sampling.sampled_point_density", 10000)
    process_func = partial(_process, density=density, overwrite=overwrite)
    
    with logging_redirect_tqdm():
        if nworkers < 0 or nworkers > 0:
            process_map(process_func, dataset.scans(), max_workers=nworkers, chunksize=1, total=dataset.nscans, desc="Sampling point clouds from meshes", tqdm_class=tqdm)
        else:
            pbar = tqdm(dataset.scans(), total=dataset.nscans, desc="Sampling point clouds from meshes")
            for scan in pbar:
                process_func(scan)


# def main():
#     from ssg_tools import ConfigArgumentParser, init_logging

#     parser = ConfigArgumentParser(
#         description='Generate the sampled points from the original mesh data')
#     parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing files.')
#     parser.add_argument('-n', '--nworkers', type=int, required=False, default=0, help='use multiple workers for processing of some tasks.')

#     args = parser.parse_args()

#     config = args.config
#     init_logging(config)

#     dataset = DatasetInterface3DSSG(config)
#     log.info("Sampling points from scene meshes...")
#     sample_points(dataset, nworkers=args.nworkers, overwrite=args.overwrite)

# if __name__ == "__main__":
#     main()