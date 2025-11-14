from __future__ import annotations
from pathlib import Path
from typing import Literal, Generator, Sequence, Any, Optional
from functools import cached_property
from ssg_tools.dataset.camera import Camera
from ssg_tools.dataset.mesh import preprocess_mesh
import dataclasses
import numpy as np
import json
import logging
import trimesh
import h5py
from typing import Mapping
import networkx as nx

log = logging.getLogger(__name__)

__all__ = ["DatasetInterface3DSSG", "ScanInterface"]


_name_patterns = {
    "color": "frame-{0:06d}.color.jpg",
    "instance": "frame-{0:06d}.rendered.instances.png",
    "label": "frame-{0:06d}.rendered.labels.png",
    "depth": "frame-{0:06d}.depth.pgm",
    "pose": "frame-{0:06d}.pose.txt",
}


class DatasetInterface3DSSG:
    def __init__(self, path: Path, scan_ids: Optional[str | Sequence[str] | Sequence[bool]] = None, use_scene_json: bool = False) -> None:
        self.path = Path(path)
        self.path_3rscan = self.path / "3RScan"
        self.path_3dssg = self.path / "3DSSG"
        self.path_scenegraph_data = self.path / "scenegraph.json"
        self.path_visibility_graph = self.path / "visibility_graph.h5"
        self.path_roi_images = self.path / "roi_images.h5"
        self.use_scene_json = use_scene_json
        if scan_ids is not None:
            self.scan_ids = scan_ids

    def filename_split(self, split: Literal["train", "validate", "test"]):
        return self.path / f"split_{split}.json"

    @cached_property
    def _read_scan_ids(self) -> Sequence[str]:
        if self.use_scene_json and self.path_scenegraph_data.exists():
            # use the scans mapped into the scene graph
            scan_ids = self._read_scene_graph_data["scenes"].keys()
        else:
            log.warning("Determining scan id from directory structure.")
            # use all the scans in the directory
            scan_ids = [str(file.name) for file in self.path_3rscan.glob("*") if file.is_dir()]
            log.info("Automatically determined %d scans.", len(scan_ids))
        return sorted(scan_ids)

    @property
    def scan_ids(self):
        custom_ids = getattr(self, "_scan_ids", None)
        if custom_ids is None:
            # use the default ids
            return self._read_scan_ids
        else:
            return custom_ids

    @scan_ids.setter
    def scan_ids(self, scan_ids: str | Sequence[str] | Sequence[bool]):
        if isinstance(scan_ids, (list, tuple)):
            if isinstance(scan_ids[0], str):
                # assume string array
                self._scan_ids = sorted(scan_ids)
            elif isinstance(scan_ids[0], bool):
                # mask the default scan ids
                self._scan_ids = np.array(self._read_scan_ids, dtype=object)[scan_ids].tolist()
            else:
                raise ValueError(f"Invalid type for scan ids {scan_ids}")
        else:
            # assume string containing a filename
            scan_ids_path = Path(scan_ids)
            if not scan_ids_path.exists():
                raise FileExistsError(f"Scan ids file {scan_ids_path} does not exist.")
            with open(scan_ids_path, "r") as f:
                data = json.load(f)
            self.scan_ids = data

    @property
    def nscans(self) -> int:
        return len(self.scan_ids)

    def scans(self) -> Generator[ScanInterface, None, None]:
        for scan_id in self.scan_ids:
            yield ScanInterface(self, scan_id)

    def scan(self, id: str | int) -> ScanInterface:
        if isinstance(id, int):
            id = list(self.scan_ids)[id]
        else:
            if id not in self.scan_ids:
                raise ValueError(f"Scan id {id} does not exist.")
        return ScanInterface(self, id)

    def __getitem__(self, item):
        return self.scan(item)

    def __iter__(self):
        return self.scans()

    def __len__(self):
        return self.nscans

    def __repr__(self):
        return f"{self.__class__.__name__}(path={self.path}, nscans={self.nscans})"

    @cached_property
    def _read_scene_graph_data(self) -> dict[str, Any]:
        with open(self.path_scenegraph_data, "r") as f:
            data = json.load(f)
            # convert the string keys back into ints for consistency
            for scene_id, scene in data["scenes"].items():
                scene["nodes"] = {int(obj_id): obj for obj_id, obj in scene["nodes"].items()}
        return data

    def has_scene_graph(self) -> bool:
        return self.path_scenegraph_data.exists()

    @cached_property
    def _visibility_graph_file(self) -> Mapping[str, Any]:
        f = h5py.File(self.path_visibility_graph, "r")
        return f

    @cached_property
    def _roi_images_file(self) -> Mapping[str, Any]:
        f = h5py.File(self.path_roi_images, "r")
        return f

    def node_classes(self):
        if not self.has_scene_graph():
            return []
        return self._read_scene_graph_data["node_classes"]

    def edge_classes(self):
        if not self.has_scene_graph():
            return []
        return self._read_scene_graph_data["edge_classes"]


ImageTypeStr = Literal["color", "instance", "label", "depth"]
_filename_labels_mesh = "labels.instances.annotated.v2.ply"
_filename_color_mesh = "mesh.refined.v2.obj"
_filename_sampled_points = "points.sampled.ply"
_directory_images = "sequence"
_filename_images_info = "_info.txt"
_filename_ground_truth_2d = "ground_truth_2d.json"
_filename_semantic_segmentation = "semseg.v2.json"


@dataclasses.dataclass
class ScanInterface:
    dataset: DatasetInterface3DSSG
    scan_id: str

    def label_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(self.label_mesh_filename(), process=False)
        return preprocess_mesh(mesh, label_type="segment")

    def label_mesh_filename(self) -> Path:
        return self.scan_path / _filename_labels_mesh

    def has_label_mesh(self) -> bool:
        return self.label_mesh_filename().exists()

    def color_mesh_filename(self) -> Path:
        return self.scan_path / _filename_color_mesh

    def color_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(self.color_mesh_filename(), process=False)
        return mesh

    def has_color_mesh(self) -> bool:
        return self.color_mesh_filename().exists()

    def sampled_points_filename(self) -> Path:
        return self.scan_path / _filename_sampled_points

    def sampled_points(self) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(self.sampled_points_filename(), process=False)
        return preprocess_mesh(mesh, label_type="segment")

    def has_sampled_points(self) -> bool:
        return self.sampled_points_filename().exists()

    @cached_property
    def scan_path(self) -> Path:
        return self.dataset.path_3rscan / self.scan_id

    @property
    def nimages(self) -> int:
        return int(self._read_images_info["m_frames.size"])

    def image_filename(self, index: int, type: ImageTypeStr) -> Path:
        return self.scan_path / _directory_images / _name_patterns[type].format(index)

    def image(self, index: int, type: ImageTypeStr) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.image_filename(index, type)))

    def image_size(self, type: ImageTypeStr) -> tuple[int, int]:
        info = self._read_images_info
        # we already rotated the input view when generating the rendered views. so swap h and w
        if type == "depth":
            return int(info["m_depthHeight"]), int(info["m_depthWidth"])
        else:
            return int(info["m_colorHeight"]), int(info["m_colorWidth"])

    def pose_filename(self, index: int) -> Path:
        return self.scan_path / _directory_images / _name_patterns["pose"].format(index)

    def pose(self, index: int) -> np.ndarray:
        return np.loadtxt(self.pose_filename(index))

    @cached_property
    def _read_images_info(self) -> dict[str, str]:
        info_file = self.scan_path / _directory_images / _filename_images_info

        if not info_file.exists():
            log.debug("Try to unpack sequence data for scan {self.scan_id}.")
            # try to unpack the sequence.zip
            sequence_zip_file = self.scan_path / f"{_directory_images}.zip"
            if not sequence_zip_file.exists():
                raise FileNotFoundError("Sequence data is not downloaded for scan {self.scan_id}.")
            from .preprocessing.download_util import unzip_file

            unzip_file(sequence_zip_file, destination=self.scan_path / _directory_images, overwrite=True)

            if not info_file.exists():
                raise FileNotFoundError("failed to extract sequence data for scan {self.scan_id}.")

        with open(info_file, "r") as f:
            lines = f.readlines()
            output = dict()
            for line in lines:
                split = line.rstrip().split(" = ")
                output[split[0]] = split[1]
        return output

    def camera(self, type: ImageTypeStr = "color"):
        info = self._read_images_info
        if type == "depth":
            intrinsic = np.asarray([float(sp) for sp in info["m_calibrationDepthIntrinsic"].split(" ")], dtype=np.float32).reshape(4, 4)
        else:
            intrinsic = np.asarray([float(sp) for sp in info["m_calibrationColorIntrinsic"].split(" ")], dtype=np.float32).reshape(4, 4)
        return Camera.from_camera_matrix(intrinsic[:3, :3], self.image_size(type))

    def scene_graph(self, raw: bool = True) -> dict[str, Any] | nx.MultiDiGraph:
        scene_graph_data = self.dataset._read_scene_graph_data["scenes"]
        if raw:
            return scene_graph_data[self.scan_id]
        g = nx.MultiDiGraph(scan_id=self.scan_id)
        scene_graph_nodes = scene_graph_data[self.scan_id]["nodes"]
        g.add_nodes_from(scene_graph_nodes.items())

        scene_graph_edges = scene_graph_data[self.scan_id]["edges"]
        g.add_edges_from(((subject, object, label, dict(name=name)) for subject, object, label, name in scene_graph_edges))
        return g

    def neighbors_graph(self) -> nx.Graph:
        scene_graph_data = self.dataset._read_scene_graph_data["scenes"]
        g = nx.Graph(scan_id=self.scan_id)
        scene_graph_nodes = scene_graph_data[self.scan_id]["nodes"]
        g.add_nodes_from(scene_graph_nodes.items())

        neighbors_edges = ((s, o) for s, d in g.nodes(data="neighbors") for o in d)
        g.add_edges_from(neighbors_edges)
        return g

    @cached_property
    def _read_ground_truth_2d(self) -> dict[str, Any]:
        ground_truth_file = self.scan_path / _filename_ground_truth_2d
        if not ground_truth_file.is_file():
            raise RuntimeError(f"Ground truth data has not been generated for scan {self.scan_id}.")
        with open(ground_truth_file, "r") as f:
            data = json.load(f)
        return data

    def ground_truth_2d(self) -> dict[str, Any]:
        return self._read_ground_truth_2d

    def has_ground_truth_2d(self) -> bool:
        return (self.scan_path / _filename_ground_truth_2d).exists()

    def visibility_graph(self) -> Mapping[str, Any]:
        return self.dataset._visibility_graph_file[self.scan_id]

    def roi_images(self) -> Mapping[str, Any]:
        return self.dataset._roi_images_file[self.scan_id]

    @cached_property
    def _read_instance_obbs(self):
        info_file = self.scan_path / _filename_semantic_segmentation
        with open(info_file, "r") as f:
            data = json.load(f)
        objs_obbinfo = dict()
        for seg_group in data["segGroups"]:
            obb = seg_group["obb"]
            obj_obbinfo = objs_obbinfo[int(seg_group["id"])] = {}
            obj_obbinfo["center"] = np.asarray(obb["centroid"], dtype=np.float32)
            obj_obbinfo["dimension"] = np.asarray(obb["axesLengths"], dtype=np.float32)
            obj_obbinfo["normAxes"] = np.asarray(obb["normalizedAxes"], dtype=np.float32).reshape(3, 3).transpose()
        return objs_obbinfo

    def instance_obbs(self):
        return self._read_instance_obbs
