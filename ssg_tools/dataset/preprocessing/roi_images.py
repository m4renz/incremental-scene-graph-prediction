import logging
from pathlib import Path
import h5py
import numpy as np
from torchvision.ops import roi_align
from torchvision import transforms
import torch
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm.contrib.concurrent import process_map
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG, ScanInterface
from ssg_tools.config import Config
from functools import partial
from ssg_tools.dataset.bounding_box_2d import BoundingBox2D

log = logging.getLogger(__name__)

def safe_acos(x, epsilon=1e-7): 
    # 1e-7 for float is a good value
    return torch.acos(torch.clamp(x, -1 + epsilon, 1 - epsilon))

def getAngle(P,Q):
    # R = P @ Q.T
    theta = (torch.trace(P) -1)/2
    return safe_acos(theta) * (180/np.pi)

def pose_is_close(pose, poses: list, t_a: float = 5, t_t: float = 0.3):
    # if len(poses)==0: return False
    for p_t in poses:
        diff_t = np.linalg.norm(pose[:3, 3]-p_t[:3, 3])

        if diff_t < t_t:
            # print('  t',diff_t)
            return True
        diff_angle = getAngle(pose[:3, :3], p_t[:3, :3])

        if diff_angle < t_a:
            # print('a  ',diff_angle)
            return True
    return False


def process(scan: ScanInterface, *, output_dir: Path, overwrite: bool = False):
    # specify the image transforms
    to_tensor = transforms.ToTensor()
    size = scan.dataset.config.get("dataset.roi_images.image_size", [256, 256])
    resize = transforms.Resize(size)

    output_file = (Path(output_dir) / scan.scan_id).with_suffix(".h5")
    if output_file.is_file():
        if not overwrite:
            log.info("Skipping existing file %s", output_file)
            return
        else:
            output_file.unlink()

    visibility_graph = scan.visibility_graph()

    objects = visibility_graph["objects"]
    keyframes_group = visibility_graph["keyframes"]

    if len(objects) == 0:
        return # nothing to do
        
    with h5py.File(output_file, 'w') as f:
        for object_id, keyframes in objects.items():
            img_bboxes = []
            frame_id_to_index = {}
            poses = []
            for i, frame_id in enumerate(keyframes):
                # get the keyframe from the hdf5 file
                keyframe = keyframes_group[str(frame_id)]

                image_rgb = scan.image(frame_id, type='color')
                image_rgb = np.rot90(image_rgb, 3) # rotate image
                pose = torch.from_numpy(scan.pose(frame_id))

                height, width = image_rgb.shape[:2]

                if pose.isnan().any() or pose.isinf().any():
                    continue # skip invalid poses

                if pose_is_close(pose, poses):
                    continue # skip similar poses

                poses.append(pose)

                object_index = list(keyframe["index_to_instance_id"]).index(int(object_id))

                bbox = BoundingBox2D(*keyframe["bboxes"][object_index])
                #occlusion = keyframe["occlusion"][object_index]
                assert bbox.valid()

                # Denormalize bounds to the image shape
                bbox.denormalize((width, height))

                bbox = torch.from_numpy(np.array(bbox.tolist())).float().view(1, -1)
                rgb_tensor = to_tensor(image_rgb.copy()).unsqueeze(0)

                w = bbox[:, 2] - bbox[:, 0]
                h = bbox[:, 3] - bbox[:, 1]

                region = roi_align(rgb_tensor, [bbox], [h, w])
                region = resize(region).squeeze(0)
                img_bboxes.append(region)
                frame_id_to_index[frame_id] = i

            if len(img_bboxes) == 0:
                raise RuntimeError(f"Scan {scan.scan_id} node id: {object_id} has no image bboxes.")
            
            img_bboxes = torch.stack(img_bboxes)
            chunk_size = list(img_bboxes.shape)
            chunk_size[0] = 1
            dataset = f.create_dataset(str(object_id), data=img_bboxes.numpy(), compression="gzip", compression_opts=9, chunks=tuple(chunk_size))
            # store the mapping from frame_id to image index in the attributes
            for frame_id, index in frame_id_to_index.items():
                dataset.attrs[str(frame_id)] = index
    

def image_rois(dataset: DatasetInterface3DSSG, nworkers: int = 0, overwrite: bool = False):
    config = dataset.config

    roi_output_dir = Path(config.dataset.roi_images.dir_roi_images)
    rois_output_file = Path(config.dataset.roi_images.filename_roi_images)

    roi_output_dir.mkdir(parents=True, exist_ok=True)
        
    process_func = partial(process, output_dir=roi_output_dir, overwrite=overwrite)
    with logging_redirect_tqdm():
        if nworkers < 0 or nworkers > 0:
            process_map(process_func, dataset.scans(), total=dataset.nscans, max_workers=nworkers, chunksize=1)
        else:
            pbar = tqdm(dataset.scans(), total=dataset.nscans)
            for scan in pbar:
                pbar.set_description(scan.scan_id)
                process_func(scan)

    # create the linking h5 file
    with h5py.File(rois_output_file, 'w') as f:
        for path in roi_output_dir.glob("*.h5"):
            name = path.stem
            f[name] = h5py.ExternalLink(name, str(path.relative_to(rois_output_file.parent)))


def main():
    from ssg_tools import ConfigArgumentParser, init_logging

    parser = ConfigArgumentParser(
        description='Generate the roi images from the raw 3DSSG images.')
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing files.')
    parser.add_argument('-n', '--nworkers', type=int, required=False, default=0, help='use multiple workers for processing of some tasks.')

    args = parser.parse_args()

    config = args.config
    init_logging(config)

    dataset = DatasetInterface3DSSG(config)
    image_rois(dataset, nworkers=args.nworkers, overwrite=args.overwrite)

if __name__ == "__main__":
    main()