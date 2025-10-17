from __future__ import annotations
import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map 
from functools import partial
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG, ScanInterface
from ssg_tools.dataset.bounding_box_2d import BoundingBox2D
from ssg_tools.config import Config
import logging
import json


log = logging.getLogger(__name__)

def mask_occupancy(mask: np.ndarray) -> float:
    count = np.count_nonzero(mask)
    return count / mask.size # normalize by the number of pixel in the mask


def detect(instance_image: np.ndarray, labels: dict, scene_graph_nodes: dict) -> dict:
    detections = {}
    for instance_id, label in labels.items():
        indices = (instance_image == instance_id).nonzero()

        bbox = BoundingBox2D(indices[1].min(), indices[0].min(), indices[1].max(), indices[0].max())
        if not bbox.valid(): continue

        occupancy = mask_occupancy(instance_image[bbox.y_min:bbox.y_max,bbox.x_min:bbox.x_max]==instance_id)

        det = scene_graph_nodes[instance_id].copy()
        det["bbox"] = bbox.tolist()
        det["max_iou"] = float(occupancy)
        detections[int(instance_id)] = det
    return detections


def process(scan: ScanInterface, overwrite: bool = False):
    # check if file exist
    config = scan.config
    scan_path = scan.scan_path
    out_file = scan_path / config.dataset.ground_truth_2d.filename_ground_truth_2d
    if out_file.is_file():
        if out_file.stat().st_size == 0:
            out_file.unlink()
        else:
            if overwrite:
                out_file.unlink()
            else:
                log.debug("Skipping existing file %s", out_file)
                return    
    
    # load semseg
    log.debug("Loading scene graph data...")
    scene_graph = scan.scene_graph(raw=True)
    scene_graph_nodes = scene_graph['nodes']
    #scene_graph_instances = set(scene_graph_nodes.keys())

    # get number of scans
    nimages = scan.nimages

    log.debug("Processing %d sequence images...", nimages)
    # check all images exist
    for frame_id in range(nimages):
        image_filename = scan.image_filename(frame_id, "instance")        
        if not image_filename.is_file(): 
            log.warning("File %s does not exist", image_filename)
            return # skip this scan    
    
    ground_truth_2d_data = {}

    for frame_id in range(nimages):
        instance_image = scan.image(frame_id, "instance")
        
        instance_ids = set(np.unique(instance_image))
        
        # ignore all instances not in the scene graph
        labels = {int(i): scene_graph_nodes[i]['rio27_name'] for i in instance_ids if i in scene_graph_nodes}

        ground_truth_2d_data[frame_id] = detect(instance_image, labels, scene_graph_nodes)      

    with open(out_file, 'w') as f:
        json.dump(ground_truth_2d_data, f, indent=4)


def ground_truth_2d(dataset: DatasetInterface3DSSG, nworkers: int = 0, overwrite: bool = False):
    process_func = partial(process, overwrite=overwrite)
    dataset._read_scene_graph_data()

    if nworkers < 0 or nworkers > 0:
        process_map(process_func, dataset.scans(), max_workers=nworkers, total=dataset.nscans)
    else:
        for scan in tqdm(dataset.scans(), total=dataset.nscans):
            process_func(scan)

def main():
    from ssg_tools import ConfigArgumentParser, init_logging

    parser = ConfigArgumentParser(
        description='Generate the 2 ground truth masks from the raw 3DSSG images.')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing files.')
    parser.add_argument('-n', '--nworkers', type=int, required=False, default=0, help='use multiple workers for processing of some tasks.')

    args = parser.parse_args()

    config = args.config
    init_logging(config)

    dataset = DatasetInterface3DSSG(config)
    ground_truth_2d(dataset, nworkers=args.nworkers, overwrite=args.overwrite)

if __name__ == "__main__":
    main()