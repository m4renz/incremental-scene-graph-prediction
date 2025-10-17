from __future__ import annotations
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG, ScanInterface
import logging
import numpy as np
from pathlib import Path
import h5py
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from collections import defaultdict, namedtuple
from ssg_tools.dataset.bounding_box_2d import BoundingBox2D

log = logging.getLogger(__name__)

class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError( key )
        else:
            ret = self[key] = self.default_factory(key)
            return ret
        
ObjectInstance = namedtuple("ObjectInstance", "object_id, object_label, occulsion, x_min, y_min, x_max, y_max")

def filter_objects(scan: ScanInterface):
    ground_truth = scan.ground_truth_2d()

    filtered_objects = defaultdict(dict)
    width, height = scan.image_size("color")
    filter_labels = scan.config.get("dataset.visibility_graph.filter_labels", [])
    filter_edge = scan.config.get("dataset.visibilty_graph.filter_edge", False)
    min_occlusion = scan.config.get("dataset.visibility_graph.min_occlusion", 1.0)
    min_shape = scan.config.get("dataset.visibility_graph.min_shape", None)
    min_objects = scan.config.get("dataset.visibility_graph.min_objects", 1)
    for frame_id, instances in ground_truth.items():
        frame_id = int(frame_id)

        for instance_id, instance in instances.items():
            label = instance["rio27_name"]
            occulsion = round(instance["max_iou"], 3)
            bbox = BoundingBox2D(*instance["bbox"])

            # filter labels if available
            if label in filter_labels:
                 continue
            
            # filter by bounding box size
            if filter_edge:
                if bbox.x_min < 1 or bbox.y_min < 1 or width < bbox.x_max or height < bbox.y_max:
                    continue

            if occulsion < min_occlusion: # if the object is occluded
                continue

            if min_shape:
                shape = bbox.shape()
                if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
                    continue
            
            # normalize bounding box
            bbox.normalize((width, height))
            instance["bbox"] = bbox.tolist()
            filtered_objects[frame_id][instance_id] = instance

        if len(filtered_objects[frame_id]) < min_objects:
            del filtered_objects[frame_id] # discard the entire frame if there are not enough objects

    return filtered_objects
    

def visibility_graph(dataset: DatasetInterface3DSSG,
                     overwrite: bool = False):
    config = dataset.config

    output_file = Path(config.dataset.visibility_graph.filename_visibility_graph)
    
    log.info("Building visibility graph for %d scans", dataset.nscans)

    if output_file.exists():
        if overwrite:
            output_file.unlink()
        else:
            log.info("File exists. Skipping ...")
            return
    with h5py.File(output_file, 'w') as f:
        with logging_redirect_tqdm():
            for scan in tqdm(dataset.scans(), total=dataset.nscans):
                assert scan.scan_id in dataset.scan_ids
                scene_graph = scan.scene_graph()["nodes"]

                keyframes = filter_objects(scan)

                # create the inverse mapping from instance_id to keyframe
                node_to_keyframes = defaultdict(list)
                for frame_id, objects in keyframes.items():
                    for instance_id in objects.keys():
                        node_to_keyframes[instance_id].append(frame_id)

                # check that each node has at least one kf
                for instance_id, frames in node_to_keyframes.items():
                    if len(frames) == 0:
                        del node_to_keyframes[instance_id]

                # check if empty and skip
                if len(node_to_keyframes) == 0:
                    log.warning("Skipping invalid scan: %s", scan.scan_id)
                    continue
                
                scan_group = f.create_group(scan.scan_id)
                objects_group = scan_group.create_group("objects")
                keyframes_group = scan_group.create_group("keyframes")

                for frame_id, instances in keyframes.items():
                    frame_group = keyframes_group.create_group(str(frame_id))

                    bboxes = np.asarray([instance["bbox"] for instance in instances.values()], dtype=np.float32)
                    occlusions = np.asarray([instance["max_iou"] for instance in instances.values()], dtype=np.float32)
                    index_to_instance_ids = np.asarray(list(instances.keys()), dtype=np.int32)

                    frame_group.create_dataset("bboxes", data=bboxes)
                    frame_group.create_dataset("occlusion", data=occlusions)
                    frame_group.create_dataset("index_to_instance_id", data=index_to_instance_ids)

                for object_id, kfs in node_to_keyframes.items():
                    # generate references to the keyframes group
                    kfs = np.asarray(kfs, dtype=np.int32)
                    #keyframe_refs = [keyframes_group[str(k)].ref for k in kfs]
                    ref_dset = objects_group.create_dataset(str(object_id), data=kfs)

                    # store the object's label data as well
                    for k, v in scene_graph[int(object_id)].items():
                        ref_dset.attrs[k] = v

    log.info("done.")

def main():
    from ssg_tools import ConfigArgumentParser, init_logging

    parser = ConfigArgumentParser(
        description='Generate the sampled points from the original mesh data')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing files.')
    args = parser.parse_args()

    config = args.config
    init_logging(config)

    dataset = DatasetInterface3DSSG(config)
    visibility_graph(dataset, overwrite=args.overwrite)

if __name__ == "__main__":
    main()

