import concurrent.futures
import json
import logging
import logging.handlers
import multiprocessing
import shutil
import warnings
from pathlib import Path
from typing import Callable, Iterable, Literal

import lightning as L
import numpy as np
import torch
import tqdm
from ssg_tools.utils.training import corrupt_labels
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater  # noqa: F401
from torch_geometric.utils import from_networkx
import torch_geometric.utils as utils
from torch import Tensor
from ssg_tools.dataset.preprocessing.hetero_frames import PointCloudLoader
import networkx as nx
from networkx.readwrite import json_graph


LOG_PATH = Path().cwd().joinpath("logs")
LOG_PATH.mkdir(exist_ok=True)
LOG_LVL = logging.INFO

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


def to_undirected(edge_index, edge_attr, reduce="mean", keep_multi=False):

    if utils.is_undirected(edge_index=edge_index, edge_attr=edge_attr):
        return edge_index, edge_attr

    if keep_multi:
        row, col = edge_index[0], edge_index[1]
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        edge_index = torch.stack([row, col], dim=0)

        if isinstance(edge_attr, Tensor):
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        elif isinstance(edge_attr, (list, tuple)):
            edge_attr = [torch.cat([e, e], dim=0) for e in edge_attr]

        if utils.contains_self_loops(edge_index):
            e_i, a_i, e_j, a_j = utils.segregate_self_loops(edge_index, edge_attr)
            e_j, a_j = utils.coalesce(edge_index=e_j, edge_attr=a_j, reduce="mean")

            edge_index = torch.cat([e_i, e_j], dim=1)
            edge_attr = torch.cat([a_i, a_j])
            edge_index, edge_attr = utils.sort_edge_index(edge_index, edge_attr)

            return edge_index, edge_attr
        else:
            return edge_index, edge_attr
    else:
        return utils.to_undirected(edge_index=edge_index, edge_attr=edge_attr, reduce=reduce)


def load_split_list(root: str | Path, split: str):
    """
    Helper function to load scene graph splits from a JSON file.
    """
    root = Path(root)

    if split not in ["train", "validate", "test"]:
        raise ValueError(f"Invalid split: {split}")

    with open(root.joinpath(f"split_{split}.json"), "r") as f:
        return json.load(f)


def listerner_configurer():
    root = logging.getLogger()
    file_handler = logging.FileHandler(LOG_PATH.joinpath("hetero_dataset.log"), mode="w")
    message_format = logging.Formatter("%(asctime)s - %(process)d - %(threadName)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(message_format)
    root.addHandler(file_handler)

    s_handler = logging.StreamHandler()
    s_handler.setFormatter(logging.Formatter("%(message)s"))
    s_handler.setLevel(LOG_LVL)
    root.addHandler(s_handler)


def listener_process(queue: multiprocessing.Queue, configurer: Callable):
    """
    Lister process for multiprocessing logging.
    """
    configurer()
    try:
        while True:
            record = queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
    except Exception:
        import sys
        import traceback

        print("Whoops! Problem:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)


def worker_configurer(queue: multiprocessing.Queue):
    """
    Configurerer for multiprocessing logging.
    """
    root = logging.getLogger()

    # check if root logger already has handler
    # otherwise a handler is added each time configurer is called
    # resulting in multiple logs from subprocesses/threads
    if not any(isinstance(h, logging.handlers.QueueHandler) for h in root.handlers):
        root.addHandler(logging.handlers.QueueHandler(queue))

    root.setLevel(logging.WARNING)


def load_and_convert_hierarchical_nxg(path: str | Path):

    # Load hierarchical graph from JSON (node-link format)
    with open(path, "r") as f:
        data = json.load(f)
    g: nx.Graph | nx.DiGraph = json_graph.node_link_graph(data, edges="links")

    edges_u = []
    edges_v = []
    features = []
    edge_type = []

    for u, v, d in g.edges(data=True):
        u_id = int(u.split(":")[1])
        v_id = int(v.split(":")[1])

        edges_u.append(u_id)
        edges_v.append(v_id)
        edge_type.append(d["type"])
        features.append(
            torch.tensor(
                np.array(
                    [
                        d["weight"],
                        d["overlap_obb"],
                        d["overlap_geometry"],
                        *d["center_difference"],
                        d["log_obb_volume_ratio"],
                        d["log_geom_volume_ratio"],
                        g.nodes[u]["centrality"],
                        g.nodes[v]["centrality"],
                    ]
                )
            )
        )

    edges = torch.stack([torch.tensor(edges_u), torch.tensor(edges_v)])
    features = torch.stack(features)

    return edges, features


class HeteroSceneGraphDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        split: str = "all",
        num_feature_points: int = 32,
        reprocess: bool = False,
        num_workers: int = 1,
        num_processes: int = 1,
        scans: Iterable[str] | None = None,
        corruption_rate: float = 0.0,
        homogenious: bool = False,
        single_type: str | None = "new",
        embedding_type: Literal["numberbatch"] | None = None,
        edges_to_remove: Iterable[tuple[str, str, str]] | None = None,
        hierarchical: bool = False,
    ) -> None:
        self.root = Path(root)
        self.rscan = self.root.joinpath("3RScan")
        self.data_path = self.root.joinpath("hetero_scene_graph")
        self.corruption_rate = corruption_rate
        self.homogenious = homogenious
        self.single_type = single_type
        self.embedding_type = embedding_type
        self.edges_to_remove = edges_to_remove
        self.hierarchical = hierarchical
        self.hierarchical_path = self.data_path / "hierarchical"

        if embedding_type:
            if embedding_type == "numberbatch":
                file_name = "rio27_numberbatch_embeddings.npy"
            elif embedding_type == "clip":
                file_name = "rio27_clip_embeddings.npy"
            else:
                raise ValueError(f"Invalid embedding type: {embedding_type}")

            embedding_file_path = self.data_path.joinpath(file_name)
            if not embedding_file_path.exists():
                raise FileNotFoundError(f"Embeddings for {embedding_type} not found at {embedding_file_path}")

            self.embeddings = np.load(embedding_file_path).astype("float32")
        else:
            self.embeddings = None

        if scans:
            self.scans = scans
        else:
            if split == "all":
                self.scans = self.load_split_list("train") + self.load_split_list("validate") + self.load_split_list("test")
            else:
                self.scans = self.load_split_list(split)

        self.num_feature_points = num_feature_points

        if reprocess:
            if self.data_path.joinpath("processed").exists():
                for scan_directory in self.data_path.joinpath("processed").iterdir():
                    if scan_directory.is_dir() and scan_directory.name in self.scans:
                        shutil.rmtree(scan_directory)
                        print(f"Deleted {scan_directory}")

        if not self.data_path.exists():
            self.data_path.mkdir()

        self.node_classes, self.edge_classes = self.load_classes(none_edge=False)

        self.num_node_classes = len(self.node_classes)
        self.num_edge_classes = len(self.edge_classes)

        self.num_processes = num_processes
        self.num_workers = num_workers

        super().__init__(root=str(self.data_path))

        self.graphs = []

        for scan_dir in Path(self.processed_dir).iterdir():
            if scan_dir.name in self.scans:
                self.graphs.extend(list(scan_dir.glob("*.pt")))

        try:
            self._metadata = self.get(0).metadata()
        except AttributeError:
            self._metadata = None

        self.weights = {}
        try:
            self.weights = torch.load(self.data_path.joinpath("stats/weights.pkl"))
        except FileNotFoundError:
            pass

        self.hierarchical_graphs = {}

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return self.scans

    @property
    def metadata(self):
        return self._metadata

    def download(self):
        pass

    def process(self):
        # Multiprocess logging
        queue = multiprocessing.Manager().Queue(-1)
        listener = multiprocessing.Process(target=listener_process, args=(queue, listerner_configurer))
        listener.start()

        if self.num_processes > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # Submit all the tasks to the executor
                futures = [executor.submit(self.process_scan, scan, queue, worker_configurer) for scan in self.scans]

                # Use tqdm to track the progress of futures as they complete
                for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Scans"):
                    try:
                        future.result()  # this will raise exceptions if any
                    except Exception as e:
                        queue.put_nowait(
                            logging.LogRecord(
                                name="root",
                                level=logging.ERROR,
                                msg=f"Error occurred during processing: {e}",
                                args=(),
                                exc_info=None,
                            )
                        )
                        print(f"Error occurred during processing: {e}")
        else:
            for scan in tqdm.tqdm(self.scans, desc="Scans"):
                self.process_scan(scan, queue, worker_configurer)
        # finish logging
        queue.put_nowait(None)
        listener.join()

    def process_scan(self, scan, queue: multiprocessing.Queue, configurer: Callable):
        """
        Processes a scan and saves the processed data to a file.
        Args:
            scan (str): The identifier for the scan to be processed.
            queue (multiprocessing.Queue): A multiprocessing queue for multiprocess file logging.
            configurer (Callable): A callable function to configure the queue logger.
        Returns:
            None
        Raises:
            ValueError: If the number of nodes and descriptor length do not match.
            Exception: If there is an error processing frames or creating data.
        The function performs the following steps:
        1. Configures the queue logger using the provided configurer function.
        2. Sets up a logger for the scan.
        3. Checks if the processed file already exists and skips processing if it does.
        4. Loads the point cloud data using PointCloudLoader.
        5. Processes the frames and handles any exceptions that occur.
        6. Ignores specific warnings related to tensor creation.
        7. Iterates over the processed frames and creates HeteroData objects.
        8. Handles cases where the scene graph (global and local) is empty.
        9. Validates the number of nodes and descriptor lengths.
        10. Saves the processed data to a file.
        """

        configurer(queue)
        logger = logging.getLogger(multiprocessing.current_process().name)
        logger.setLevel(logging.WARNING)

        # only reprocess if the file does not exist
        scan_dir = Path(self.processed_dir).joinpath(scan)

        if scan_dir.exists():
            logger.debug(f"Skipping {scan} as it already exists")
            return
        else:
            scan_dir.mkdir()

        try:
            loader = PointCloudLoader(self.rscan, scan, num_feature_points=self.num_feature_points, num_workers=self.num_workers)

            loader.process_frames()
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError processing frames in scan {scan}: {e}")
            return
        except Exception as e:
            logger.error(f"Error processing frames in scan {scan}: {e}")
            return

        warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy")

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self.convert_and_save_frame, frame, scan_dir, frame_name, logger)
                for frame_name, frame in loader.processed_frames.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error creating data {scan_dir}: {e}")
                    return

    def convert_and_save_frame(self, frame: dict, scan_dir: Path, frame_name: str, logger: logging.Logger):

        try:
            data = HeteroData()
            global_sg = from_networkx(frame["scene_graph"])
            if global_sg.num_nodes == 0:
                # handle empty global sg
                data["old"].descriptor = torch.empty((0, 11), dtype=torch.float)
                data["old"].points = torch.empty((0, self.num_feature_points, 3), dtype=torch.float)
                data["old"].y = torch.tensor([], dtype=torch.long)
                data["old"].instance_id = torch.tensor([], dtype=torch.long)
                data["old", "to", "old"].edge_index = torch.empty((2, 0), dtype=torch.long)
                data["old", "to", "old"].edge_label = torch.empty((0, self.num_edge_classes), dtype=torch.float)
            else:
                data["old"].descriptor = global_sg.descriptor.type(torch.float32)
                data["old"].points = global_sg.points.type(torch.float32)
                data["old"].y = global_sg.y
                data["old"].instance_id = global_sg.instance_id
                data["old", "to", "old"].edge_index = global_sg.edge_index
                if global_sg.num_edges == 0:  # case 1 node 0 edges
                    data["old", "to", "old"].edge_label = torch.empty((0, self.num_edge_classes), dtype=torch.float)
                else:
                    data["old", "to", "old"].edge_label = global_sg.edge_label

            data["old"].num_nodes = global_sg.num_nodes
            if data["old"].num_nodes != data["old"].descriptor.shape[0]:
                raise ValueError(
                    f"Number of nodes and descriptor length do not match: {data['old'].num_nodes} != {data['old'].descriptor.shape[0]}"  # noqa E501
                )

            local_sg = from_networkx(frame["local_sg"])
            if local_sg.num_nodes == 0:
                data["new"].descriptor = torch.empty((0, 11), dtype=torch.float)
                data["new"].points = torch.empty((0, self.num_feature_points, 3), dtype=torch.float)
                data["new"].y = torch.tensor([], dtype=torch.long)
                data["new"].instance_id = torch.tensor([], dtype=torch.long)
                data["new", "to", "new"].edge_index = torch.empty((2, 0), dtype=torch.long)
                data["new", "to", "new"].edge_label = torch.empty((0, self.num_edge_classes), dtype=torch.float)
            else:
                data["new"].descriptor = local_sg.descriptor.type(torch.float32)
                data["new"].points = local_sg.points.type(torch.float32)
                data["new"].y = local_sg.y
                data["new"].instance_id = local_sg.instance_id
                data["new", "to", "new"].edge_index = local_sg.edge_index
                if local_sg.num_edges == 0:  # case 1 node 0 edges
                    data["new", "to", "new"].edge_label = torch.empty((0, self.num_edge_classes), dtype=torch.float)
                else:
                    data["new", "to", "new"].edge_label = local_sg.edge_label

            data["new"].num_nodes = local_sg.num_nodes
            if data["new"].num_nodes != data["new"].descriptor.shape[0]:
                raise ValueError(
                    f"Number of nodes and descriptor length do not match: {data['new'].num_nodes} != {data['new'].descriptor.shape[0]}"  # noqa E501
                )

            data["new", "to", "old"].edge_index = self.map_edge_index(data["new"].instance_id, data["old"].instance_id)
            data["old", "to", "new"].edge_index = self.map_edge_index(data["old"].instance_id, data["new"].instance_id)

            if data["new"].num_nodes == 0:  # empty camera frames should not be added to dataset
                logger.warning(f"Empty new data at - {scan_dir.name} - {frame_name}")
                return
            torch.save(data, scan_dir.joinpath(f"{frame_name}.pt"))
        except ValueError as e:
            logger.error(f"VauleError creating data {frame_name} at {scan_dir} : {e}")
            return
        except Exception as e:
            logger.error(f"Error creating data {scan_dir}: {e}")
            return

    @staticmethod
    def map_edge_index(id_old, id_new, direction="old_to_new"):
        """
        Maps the edge index from the old to the new graph.
        Args:
            id_old (torch.Tensor): The instance IDs of the old graph.
            id_new (torch.Tensor): The instance IDs of the new graph.
            direction (str): The direction of the edge mapping.
        Returns:
            torch.Tensor: The mapped edge index.
        """

        edges_old = []
        edges_new = []

        id_old_to_index = {id.item(): idx for idx, id in enumerate(id_old)}

        for i, id in enumerate(id_new):
            index_old = id_old_to_index.get(id.item())
            if index_old is not None:
                edges_old.append(index_old)
                edges_new.append(i)

        if direction == "old_to_new":
            return torch.tensor([edges_old, edges_new], dtype=torch.long)
        elif direction == "new_to_old":
            return torch.tensor([edges_new, edges_old], dtype=torch.long)
        else:
            raise ValueError(f"Invalid direction: {direction}")

    @staticmethod
    def get_unseen_mask(g: HeteroData):
        return ~torch.isin(torch.arange(end=g["new"].num_nodes), g["old", "to", "new"].edge_index[1].unique())

    @staticmethod
    def get_hierarchical_subgraph_mask(h_edges, instance_id):
        return torch.all(torch.isin(h_edges, instance_id), dim=0)

    @staticmethod
    def to_homogeneous(g: HeteroData):
        g_h = g.to_homogeneous()
        for store in g_h.stores:
            for key, value in store.items():
                if torch.any(torch.isnan(value)):
                    g_h.update_tensor(torch.nan_to_num(value), attr_name=key)
        # raise NotImplementedError("Homogenious mode requires a single type to be specified.")
        g_h.node_types = g.node_types
        g_h.edge_types = g.edge_types

        return g_h

    def load_classes(self, none_edge=True):
        with open(self.root.joinpath("scenegraph.json"), "r") as f:
            _sg = json.load(f)

        if none_edge:
            _sg["edge_classes"].append("none")
        _sg["node_classes"].remove("-")
        return _sg["node_classes"], _sg["edge_classes"]

    def load_split_list(self, split):
        return load_split_list(self.root, split)

    def len(self):
        return len(self.graphs)

    def get(self, idx):

        # remove '-' label
        g: HeteroData = torch.load(self.graphs[idx])
        for k in g.y_dict:
            g[k].y = g[k].y - 1

        # # remove none label
        # for k in g.edge_label_dict:
        #     g[k].edge_label = g[k].edge_label[:, :-1]

        g["old"].corrupted_labels = corrupt_labels(g["old"].y, corruption_rate=self.corruption_rate, num_classes=self.num_node_classes)
        if self.embeddings is not None:
            g["old"].embeddings = torch.tensor(self.embeddings[g["old"].corrupted_labels.numpy()])

        # TODO: fix this when creating graphs
        i, j = g["new", "to", "old"].edge_index
        g["old", "to", "new"].edge_index = torch.stack([j, i])

        # some layers like HGT throw an error, when edges are in the batch that aren't configured in the model
        if self.edges_to_remove:
            for edge_type in self.edges_to_remove:
                del g[edge_type]

        g["new"].unseen = self.get_unseen_mask(g)

        if self.homogenious:
            if self.single_type:
                return self.to_homogeneous(g.node_type_subgraph(self.single_type))
            else:
                return self.to_homogeneous(g)

        if self.hierarchical:
            scan = self.graphs[idx].parents[0].name
            h_g = self.hierarchical_graphs.get(scan)
            if h_g is None:
                h_edges, h_features = load_and_convert_hierarchical_nxg(self.hierarchical_path / f"{scan}.graph.json")
                self.hierarchical_graphs[scan] = (h_edges, h_features)
            else:
                h_edges, h_features = h_g

            mask = self.get_hierarchical_subgraph_mask(h_edges, g["old"].instance_id)
            h_sub_edge_index = h_edges[:, mask]

            h_edge_index = torch.stack(
                [
                    torch.where((h_sub_edge_index[:, :, None] == g["old"].instance_id)[0, :, :])[1],
                    torch.where((h_sub_edge_index[:, :, None] == g["old"].instance_id)[1, :, :])[1],
                ]
            )
            h_edge_attr = h_features[mask].to(torch.float32)

            g["old", "h", "old"].edge_index, g["old", "h", "old"].edge_feature = to_undirected(h_edge_index, h_edge_attr, reduce="mean")

        return g


class HeteroSceneGraphModule(L.LightningDataModule):

    def __init__(
        self,
        root: str | Path,
        scan_dir: str | None = None,
        batch_size: int = 8,
        shuffle=True,
        num_workers=1,
        persistent_workers=False,
        pin_memory=False,
        num_processes=1,
        corruption_rate=0.2,
        embedding_type: Literal["numberbatch", "clip"] | None = None,
        edges_to_remove: Iterable[tuple[str, str, str]] | None = None,
        hierarchical: bool = False,
    ):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.shuffle = shuffle
        if scan_dir:
            self.scan_dir = Path(scan_dir)
        else:
            self.scan_dir = Path(self.root).joinpath("hetero_scene_graph/processed")
        self.metadata = None
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.num_processes = num_processes
        self.num_node_classes = None
        self.num_edge_classes = None
        self.corruption_rate = corruption_rate
        self.embedding_type = embedding_type
        self.edges_to_remove = edges_to_remove
        self.hierarchical = hierarchical

        self.loader = DataLoader
        self.loader_args = {}

    def prepare_data(self):
        d = HeteroSceneGraphDataset(
            self.root,
            split="all",
            num_workers=self.num_workers,
            num_processes=self.num_processes,
            reprocess=False,
            corruption_rate=self.corruption_rate,
            edges_to_remove=self.edges_to_remove,
            hierarchical=self.hierarchical,
        )
        self.metadata = d.metadata
        self.num_node_classes = d.num_node_classes
        self.num_edge_classes = d.num_edge_classes

    def setup(self, stage):

        if stage == "fit":
            self.data_train = HeteroSceneGraphDataset(
                self.root,
                split="train",
                num_workers=self.num_workers,
                num_processes=self.num_processes,
                reprocess=False,
                corruption_rate=self.corruption_rate,
                embedding_type=self.embedding_type,
                edges_to_remove=self.edges_to_remove,
                hierarchical=self.hierarchical,
            )
            self.data_val = HeteroSceneGraphDataset(
                self.root,
                split="validate",
                num_processes=self.num_processes,
                num_workers=self.num_workers,
                reprocess=False,
                corruption_rate=self.corruption_rate,
                embedding_type=self.embedding_type,
                edges_to_remove=self.edges_to_remove,
                hierarchical=self.hierarchical,
            )

            self.metadata = self.data_train.metadata
            self.num_node_classes = self.data_train.num_node_classes
            self.num_edge_classes = self.data_train.num_edge_classes

        if stage == "validate":
            self.data_val = HeteroSceneGraphDataset(
                self.root,
                split="validate",
                num_processes=self.num_processes,
                num_workers=self.num_workers,
                reprocess=False,
                corruption_rate=self.corruption_rate,
                embedding_type=self.embedding_type,
                edges_to_remove=self.edges_to_remove,
                hierarchical=self.hierarchical,
            )

            self.metadata = self.data_val.metadata
            self.num_node_classes = self.data_val.num_node_classes
            self.num_edge_classes = self.data_val.num_edge_classes

        if stage == "test":
            self.data_test = HeteroSceneGraphDataset(
                self.root,
                split="test",
                num_processes=self.num_processes,
                num_workers=self.num_workers,
                reprocess=False,
                corruption_rate=self.corruption_rate,
                embedding_type=self.embedding_type,
                edges_to_remove=self.edges_to_remove,
                hierarchical=self.hierarchical,
            )

            self.metadata = self.data_test.metadata
            self.num_node_classes = self.data_test.num_node_classes
            self.num_edge_classes = self.data_test.num_edge_classes

    def train_dataloader(self):
        return self.loader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            **self.loader_args,
        )

    def val_dataloader(self):
        return self.loader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            **self.loader_args,
        )

    def test_dataloader(self):
        return self.loader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            **self.loader_args,
        )


class HomoSceneGraphModule(L.LightningDataModule):

    def __init__(
        self,
        root: str | Path,
        scan_dir: str | None = None,
        batch_size: int = 8,
        shuffle=True,
        num_workers=1,
        persistent_workers=False,
        pin_memory=False,
        num_processes=1,
        single_type: str | None = None,
        embedding_type: Literal["clip", "numberbatch"] | None = None,
        corruption_rate: float = 0.0,
        edges_to_remove: Iterable[tuple[str, str, str]] | None = None,
    ):
        super().__init__()
        self.root = root
        if scan_dir:
            self.scan_dir = Path(scan_dir)
        else:
            self.scan_dir = Path(self.root).joinpath("hetero_scene_graph/processed")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scan_dir = Path(self.root).joinpath("hetero_scene_graph/processed")

        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.num_processes = num_processes
        self.num_node_classes = None
        self.num_edge_classes = None
        self.single_type = single_type
        self.embedding_type = embedding_type
        self.corruption_rate = corruption_rate
        self.edges_to_remove = edges_to_remove

    def prepare_data(self):
        d = HeteroSceneGraphDataset(
            self.root,
            split="all",
            num_workers=self.num_workers,
            num_processes=self.num_processes,
            reprocess=False,
            homogenious=True,
            single_type=self.single_type,
            embedding_type=self.embedding_type,
            corruption_rate=self.corruption_rate,
            edges_to_remove=self.edges_to_remove,
        )

        self.num_node_classes = d.num_node_classes
        self.num_edge_classes = d.num_edge_classes

    def setup(self, stage):

        if stage == "fit":
            self.data_train = HeteroSceneGraphDataset(
                self.root,
                split="train",
                num_workers=self.num_workers,
                num_processes=self.num_processes,
                reprocess=False,
                homogenious=True,
                single_type=self.single_type,
                embedding_type=self.embedding_type,
                corruption_rate=self.corruption_rate,
                edges_to_remove=self.edges_to_remove,
            )
            self.data_val = HeteroSceneGraphDataset(
                self.root,
                split="validate",
                num_processes=self.num_processes,
                num_workers=self.num_workers,
                reprocess=False,
                homogenious=True,
                single_type=self.single_type,
                embedding_type=self.embedding_type,
                corruption_rate=self.corruption_rate,
                edges_to_remove=self.edges_to_remove,
            )

            self.num_node_classes = self.data_train.num_node_classes
            self.num_edge_classes = self.data_train.num_edge_classes

        if stage == "validate":
            self.data_val = HeteroSceneGraphDataset(
                self.root,
                split="validate",
                num_processes=self.num_processes,
                num_workers=self.num_workers,
                reprocess=False,
                homogenious=True,
                single_type=self.single_type,
            )

            self.num_node_classes = self.data_val.num_node_classes
            self.num_edge_classes = self.data_val.num_edge_classes

        if stage == "test":
            self.data_test = HeteroSceneGraphDataset(
                self.root,
                split="test",
                num_processes=self.num_processes,
                num_workers=self.num_workers,
                reprocess=False,
                homogenious=True,
                single_type=self.single_type,
                embedding_type=self.embedding_type,
                corruption_rate=self.corruption_rate,
                edges_to_remove=self.edges_to_remove,
            )

            self.num_node_classes = self.data_test.num_node_classes
            self.num_edge_classes = self.data_test.num_edge_classes

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )


class FullSceneGraphModule(L.LightningDataModule):
    """
    FullSceneGraphModule is a PyTorch Lightning DataModule for handling the loading and processing of
    3dssg scene graph data for training, validation, and testing. Returns Rio27 labels without `-`.
    Args:
        root (str | Path): The root directory where the dataset splits are defined.
        data_path (str | Path): The directory where the .pt files are stored.
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 8.
        num_workers (int, optional): The number of worker processes for data loading. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the Train DataLoader. Defaults to True.
    Attributes:
        root (Path): The root directory where the dataset splits are defined.
        data_path (Path): The directory where the .pt files are stored.
        batch_size (int): The batch size for the DataLoader.
        num_workers (int): The number of worker processes for data loading.
        scans (list): List of all .pt files in the data_path directory.
        train_split (list): List of training data identifiers.
        val_split (list): List of validation data identifiers.
        test_split (list): List of test data identifiers.
        train_data (list): List of loaded training data.
        val_data (list): List of loaded validation data.
        test_data (list): List of loaded test data.
    Methods:
        prepare_data(): Loads the data for training, validation, and testing splits.
        setup(stage): Sets up the data for the specified stage ('fit', 'validate', or 'test').
        train_dataloader(): Returns a DataLoader for the training data.
        val_dataloader(): Returns a DataLoader for the validation data.
        test_dataloader(): Returns a DataLoader for the test data.
    """

    def __init__(self, root: str | Path, data_path: str | Path | None, batch_size: int = 8, num_workers: int = 1, shuffle: bool = True):
        super().__init__()
        self.root = Path(root)
        self.data_path = Path(data_path) if data_path is not None else root / "full_scene_graph"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.scans = [x for x in self.data_path.glob("*.pt")]
        self.train_split = load_split_list(self.root, "train")
        self.val_split = load_split_list(self.root, "validate")
        self.test_split = load_split_list(self.root, "test")

        self.train_data = []
        self.val_data = []
        self.test_data = []

    def prepare_data(self):
        self.train_data = [torch.load(x) for x in self.scans if x.stem in self.train_split]
        self.val_data = [torch.load(x) for x in self.scans if x.stem in self.val_split]
        self.test_data = [torch.load(x) for x in self.scans if x.stem in self.test_split]

    def setup(self, stage):
        if stage == "fit":
            if not self.train_data:
                self.train_data = [torch.load(x) for x in self.scans if x.stem in self.train_split]
            if not self.val_data:
                self.val_data = [torch.load(x) for x in self.scans if x.stem in self.val_split]
        if stage == "validate":
            if not self.val_data:
                self.val_data = [torch.load(x) for x in self.scans if x.stem in self.val_split]
        if stage == "test":
            if not self.test_data:
                self.test_data = [torch.load(x) for x in self.scans if x.stem in self.test_split]

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


# if __name__ == "__main__":
#     multiprocessing.set_start_method("spawn")

#     warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy")
#     warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

# dataset = HeteroSceneGraphDataset(
#     "/path/to/dataset",
#     #   split='train',
#     reprocess=True,
#     num_feature_points=256,
#     num_processes=2,
#     num_workers=16,
# )
