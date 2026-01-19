import concurrent.futures
import logging
import queue
import threading
from copy import deepcopy
from pathlib import Path

import cv2 as cv
import networkx as nx
import numpy as np
import open3d as o3d
import torch
from ssg_tools.dataset.util.sg_handler import SGHandler


def rotate_image(image, angle=90, scale=1.0):

    # Get image dimensions
    (h, w) = image.shape[:2]

    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Compute the rotation matrix
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)

    # Calculate the sine and cosine (rotation components)
    # cos = np.abs(rotation_matrix[0, 0])
    # sin = np.abs(rotation_matrix[0, 1])

    # Compute the new bounding dimensions of the image
    # new_w = int((h * sin) + (w * cos))
    # new_h = int((h * cos) + (w * sin))

    new_w = h
    new_h = w

    # Adjust the rotation matrix to take into account translation
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]

    # Perform the rotation
    rotated = cv.warpAffine(image, rotation_matrix, (new_w, new_h))

    return rotated


def merge_edge_labels(edges: dict, num_classes: int):
    """
    Take a dict of edges from nx.MultiGraph and merge the edge labels into a multilabel one-hot encoded vector.

    Args:
        edges (dict): A dict of edges from nx.MultiGraph with label_idx.
        num_classes (int): The number of classes in the dataset.
    """
    edge_labels = []
    for edge in edges.values():
        edge_labels.append(edge["label_idx"])

    # torch.nn.functional.BCEWithLogitsLoss expects a float tensor
    return torch.nn.functional.one_hot(torch.tensor(edge_labels), num_classes=num_classes).sum(dim=0).type(torch.FloatTensor)


def sample_points(points, num_points, dtype=np.float32, min_frac=0.5):

    if isinstance(points, o3d.t.geometry.PointCloud):
        points: np.ndarray = points.point.positions.numpy()

    if len(points) < num_points:
        # if no or too few points
        # duplicating singular points will result in nan values
        # since length, dim and volume in descriptor are 0 and log is taken in ksgn
        if points.shape[0] == 0 or points.shape[0] * min_frac < num_points:
            return None
        else:
            # upsample points if too few
            div = num_points // points.shape[0]
            remainder = num_points % points.shape[0]
            return np.concatenate([points] * div + [points[:remainder]], dtype=dtype)
    else:
        rng = np.random.default_rng()
        sample_idx = rng.choice(len(points), num_points, replace=False)
        return points[sample_idx].astype(dtype)


class PointCloudLoader:

    def __init__(self, root: str | Path, scan: str, num_feature_points=32, num_workers=1, num_frames: int | None = None) -> None:

        # if device:
        #     self.device = device.lower()
        # else:
        #     self.device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
        # self.o3d_device = o3d.core.Device(self.device)

        self.root = Path(root)
        self.scan = scan
        self.sequence_path = self.root.joinpath(self.scan).joinpath("sequence")
        info_file = self.root.joinpath(self.scan).joinpath("sequence/_info.txt")
        info = self.load_info(info_file)

        self.frame_names = self.get_unique_frame_names(self.sequence_path)
        if num_frames:
            self.frame_names = self.frame_names[:num_frames]
        self.depth_max = 3.0
        self.num_workers = num_workers

        self.width, self.height, self.intrinsic = self.get_camera_info(info, depth=False)
        self.intrinsic = self.intrinsic[:3, :3]
        self.pinhole_intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.intrinsic[:3, :3])

        self.num_feature_points = num_feature_points

        self.global_pcd = None

        self.processed_frames = {}

        self.i = 0

        self.global_sg = nx.DiGraph()
        self.sg_handler = SGHandler(self.root.parent.joinpath("scenegraph.json"), self.scan)

        self.lock = threading.Lock()
        self.queue = queue.Queue()
        self.frame_counter = 0
        self.frame_buffer = {}

        self.thread = threading.Thread(target=self.thread_safe_merge, daemon=True)
        self.thread.start()

    def load_info(self, file):
        info = {}
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                key, value = line.split("=")
                info[key.strip()] = value.strip()
        return info

    def get_camera_info(self, info, depth=False):

        cam_type = "color" if not depth else "depth"

        width = int(info[f"m_{cam_type}Width"])
        height = int(info[f"m_{cam_type}Height"])
        intrinsic = [float(x) for x in info[f"m_calibration{cam_type.capitalize()}Intrinsic"].split(" ")]
        intrinsic = np.array(intrinsic).reshape(4, 4)

        return width, height, intrinsic

    @staticmethod
    def get_unique_frame_names(directory, sort=True):
        # Create a Path object for the directory
        path = Path(directory)

        # Use a set to store unique frame names
        unique_frame_names = {
            str(file.stem).split(".")[0]
            for file in path.glob("frame-*")  # Adjust the pattern to match your file extensions
            if file.is_file()
        }

        # Convert the set to a list and return
        return sorted(unique_frame_names) if sort else list(unique_frame_names)

    def load_rgbd_image(self, frame):

        # load the rendered color here since it dosn't matter for graph learning
        color_raw = o3d.t.io.read_image(str(self.sequence_path.joinpath(frame).with_suffix(".color.jpg")))
        # depth_raw = o3d.t.io.read_image(str(self.sequence_path.joinpath(frame).with_suffix('.rendered.depth.png')))
        depth_raw = cv.imread(self.sequence_path.joinpath(frame).with_suffix(".rendered.depth.png"), cv.IMREAD_UNCHANGED)
        if depth_raw.shape[0] > depth_raw.shape[1]:
            depth_raw = rotate_image(depth_raw)
        depth_raw = o3d.t.geometry.Image(o3d.core.Tensor(depth_raw))
        # return RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
        return o3d.t.geometry.RGBDImage(color_raw, depth_raw)

    def load_rgbd_image_legacy(self, frame):
        color_raw = o3d.io.read_image(str(self.sequence_path.joinpath(frame).with_suffix(".color.jpg")))
        depth_raw = cv.imread(self.sequence_path.joinpath(frame).with_suffix(".rendered.depth.png"), cv.IMREAD_UNCHANGED)
        if depth_raw.shape[0] > depth_raw.shape[1]:
            depth_raw = rotate_image(depth_raw)
        depth_raw = o3d.geometry.Image(depth_raw)

        return o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

    def load_instance_mask(self, frame):

        mask = cv.imread(self.sequence_path.joinpath(frame).with_suffix(".rendered.instances.png"), cv.IMREAD_UNCHANGED)
        if mask.shape[0] > mask.shape[1]:
            mask = rotate_image(mask)
        return mask

    def load_pose(self, frame):

        pose = np.loadtxt(self.sequence_path.joinpath(frame).with_suffix(".pose.txt"))
        return pose

    @staticmethod
    def get_descriptor(points: np.ndarray, dtype=np.float32):
        """
        Computes a descriptor for a given set of 3D points.
        Parameters:
        points (np.ndarray): A numpy array of shape (N, 3) representing the 3D points.
        Returns:
        np.ndarray: A 1D numpy array containing the concatenated descriptor values, which include:
            - center: The center of the point cloud.
            - std: The standard deviation of the points along each axis.
            - bb_extent: The extent of the axis-aligned bounding box.
            - max_l: The maximum length of the bounding box extents.
            - volume: The volume of the axis-aligned bounding box.
        """

        pcl = o3d.t.geometry.PointCloud()
        pcl.point.positions = o3d.core.Tensor(points, o3d.core.float32)

        center = pcl.get_center().numpy()
        bb = pcl.get_axis_aligned_bounding_box()

        bb_extent = bb.get_extent().numpy()
        max_l = np.max(bb_extent)
        volume = bb_extent[0] * bb_extent[1] * bb_extent[2]

        if max_l == 0 or volume == 0:
            logging.warning("Singular point")

        std = np.std(points, axis=0)

        return np.concatenate([center, std, bb_extent, [max_l], [volume]], dtype=dtype)

    def process_instance(
        self,
        instance: int,
        instance_pcd: o3d.t.geometry.PointCloud,
        nodes_dict: dict,
        sg_handler: SGHandler,
        num_feature_points: int,
    ):

        if instance == 0:
            return

        if instance not in sg_handler.sg.nodes:
            return

        if instance_pcd.is_empty():
            return

        points = sample_points(instance_pcd, num_feature_points)
        if points is None:
            return

        descriptor = self.get_descriptor(points)

        y = sg_handler.sg.nodes[instance]["rio27_enc"]

        center = instance_pcd.get_center()
        nodes_dict[instance] = {
            "center": center,
            "descriptor": descriptor,
            "points": points,
            "y": y,
            "instance_id": instance,
        }

    def process_frame(self, frame: str):

        rgbd = self.load_rgbd_image_legacy(frame)
        # rgbd = self.load_rgbd_image(frame)
        instance_mask = self.load_instance_mask(frame)
        depth = np.array(rgbd.depth)
        valid_depth = np.logical_and(depth > 0, depth < self.depth_max * 1000)
        # check if values are non finite
        valid_depth = np.logical_and(valid_depth, np.isfinite(depth)).flatten()

        instance_mask = instance_mask.flatten()
        instance_mask = instance_mask[valid_depth]

        # rgbd_legacy = self.load_rgbd_image_legacy(frame)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.pinhole_intrinsic, project_valid_depth_only=False)
        pcd = pcd.select_by_index(np.where(valid_depth)[0])

        instances = np.unique(instance_mask).astype(int)
        masks = instance_mask[:, None] == instances
        i_pcds = [pcd.select_by_index(np.where(mask)[0]) for mask in masks.T]

        sg = nx.DiGraph()

        nodes_dict = {}

        for instance, pcd in zip(instances, i_pcds):
            self.process_instance(
                instance,
                o3d.t.geometry.PointCloud.from_legacy(pcd),
                nodes_dict,
                self.sg_handler,
                self.num_feature_points,
            )

        sg.add_nodes_from(nodes_dict.items())

        none_gt = torch.zeros((self.sg_handler.num_edge_classes,), dtype=torch.float32)

        for u, v, _ in self.sg_handler.sg.to_directed().edges:
            if sg.has_node(u) and sg.has_node(v):
                sg.add_edge(
                    u,
                    v,
                    edge_label=merge_edge_labels(self.sg_handler.sg[u][v], num_classes=self.sg_handler.num_edge_classes),
                )

        self.queue.put((frame, sg))

    def thread_safe_merge(self):
        while True:
            if self.frame_counter >= len(self.frame_names):
                break
            next_frame = self.frame_names[self.frame_counter]
            if next_frame in self.frame_buffer:
                sg = self.frame_buffer.pop(next_frame)
                self.processed_frames[next_frame] = {"scene_graph": deepcopy(self.global_sg), "local_sg": deepcopy(sg)}
                self.merge_into_global_sg(sg, self.load_pose(next_frame))
                self.frame_counter += 1
                continue

            frame, sg = self.queue.get()
            if frame == next_frame:
                self.processed_frames[frame] = {"scene_graph": deepcopy(self.global_sg), "local_sg": deepcopy(sg)}
                self.merge_into_global_sg(sg, self.load_pose(frame))
                self.frame_counter += 1
            else:
                self.frame_buffer[frame] = sg

    def process_frames(self):

        if self.num_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                executor.map(self.process_frame, self.frame_names)
        else:
            for frame in self.frame_names:
                self.process_frame(frame)

        self.thread.join()

    def merge_into_global_sg(self, sg: nx.Graph, pose: np.ndarray):

        for node in sg.nodes:
            # transform into global frame before merging
            sg.nodes[node]["points"] = sg.nodes[node]["points"] @ pose[:3, :3].T + pose[:3, 3]
            if node in self.global_sg:
                points = np.concatenate([self.global_sg.nodes[node]["points"], sg.nodes[node]["points"]], axis=0)
                self.global_sg.nodes[node]["points"] = sample_points(points, self.num_feature_points)
                self.global_sg.nodes[node]["descriptor"] = self.get_descriptor(self.global_sg.nodes[node]["points"])
                self.global_sg.nodes[node]["cnt"] += 1
            else:
                self.global_sg.add_node(node, **sg.nodes[node])
                self.global_sg.nodes[node]["cnt"] = 1
        for edge in sg.edges:
            if self.sg_handler.sg.has_edge(*edge):
                self.global_sg.add_edge(*edge, **sg.edges[edge])
