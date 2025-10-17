from __future__ import annotations
from typing import Any, Literal
from pathlib import Path
import lightning as L
import dataclasses

from ssg_tools.dataset.dataset import SceneGraphDataset, SceneGraphSGPNDataset, SceneGraphSSGPointDataset
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG

from torch_geometric.loader import DataLoader



@dataclasses.dataclass
class PreprocessingArgs:
    overwrite: bool = False
    nworkers: int = 0
    download_raw: bool = False
    download_script: str = None
    download_sequences: bool = False
    download_rendered_views: bool = False
    scene_graph: bool = True
    neighbor_graph: bool = True
    max_neighbor_distance: float = 0.5
    neighbor_search_method: str = "bbox"
    point_data: bool = True
    sampling_density: int = 10000
    splits: bool = True
    train_validate_percentage: float = 0.8


@dataclasses.dataclass
class SamplingArgs:
    enabled: bool = False
    num_hops: int = 0
    num_nodes: int = 0
    max_nodes: int = -1
    edge_sampling_mode: Literal["fully_connected", "neighbors"] = "neighbors"
    edge_sampling_max_neighbors: int = -1
    drop_edges_factor: float = 0.0
    max_edges: int = -1

class SceneGraphDataModule(L.LightningDataModule):
    def __init__(self,
                 path: Path = "./dataset",
                 num_workers: int = 0,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 runtime_sampling: SamplingArgs = SamplingArgs(),
                 normalize_weights: bool = True,
                 multi_relations: bool = True,
                 cache: bool = False,
                 preprocess_params: PreprocessingArgs = PreprocessingArgs()) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.path = Path(path)

        dataset = DatasetInterface3DSSG(self.path)
        node_classes = dataset.node_classes()
        edge_classes = dataset.edge_classes()

        # update the hyperparameters with the number of classes
        self.hparams.num_node_classes = len(node_classes)
        self.hparams.num_edge_classes = len(edge_classes)

        self.ds_class = SceneGraphDataset

    def setup(self, stage: str) -> None:

        interface = DatasetInterface3DSSG(self.path)
        if stage == "fit":
            split_file = interface.filename_split("train")
            self.train_dataset = self.ds_class(self.path, self.hparams, split=split_file, train_mode=True, cache=self.hparams.cache)
            split_file = interface.filename_split("validate")
            self.val_dataset = self.ds_class(self.path, self.hparams, split=split_file, train_mode=False, cache=self.hparams.cache)
        elif stage == "test":
            split_file = interface.filename_split("test")
            self.test_dataset = self.ds_class(self.path, self.hparams, split=split_file, train_mode=False, cache=self.hparams.cache)
        elif stage == "predict":
            split_file = interface.filename_split("validate")
            self.predict_dataset = self.ds_class(self.path, self.hparams, split=split_file, train_mode=False, cache=self.hparams.cache)
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, persistent_workers=self.hparams.num_workers > 0, shuffle=self.hparams.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, persistent_workers=self.hparams.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1)
    
    def predict_dataloader(self) -> Any:
        return DataLoader(self.predict_dataset, batch_size=1)
    
    def weights(self):
        return self.train_dataset.node_weights, self.train_dataset.edge_weights

    def preprocess(self):
        params = self.hparams.preprocess_params
        if params.download_raw:
            # download the raw 3dssg dataset
            from .preprocessing.download_3dssg import download_3dssg
            dataset = DatasetInterface3DSSG(self.path)
            download_3dssg(dataset, sequences=params.download_sequences, rendered_views=params.download_rendered_views, download_script=params.download_script, overwrite=params.overwrite)

        if params.scene_graph:
            # generate the scene graph json data
            from .preprocessing.scene_graph_generation import scene_graph_remapping
            # recreate the interface to parse the downloaded scans
            dataset = DatasetInterface3DSSG(self.path)
            scene_graph_remapping(dataset, overwrite=params.overwrite)

        if params.neighbor_graph:
            from .preprocessing.neighbor_graph import neighborhood_graph
            dataset = DatasetInterface3DSSG(self.path)
            neighborhood_graph(dataset, receptive_field=params.max_neighbor_distance, 
                               search_method=params.neighbor_search_method, overwrite=params.overwrite)

        if params.point_data:
            from .preprocessing.sample_points import sample_points
            dataset = DatasetInterface3DSSG(self.path)#, scan_ids=['0cf75f50-564d-23f5-8a6b-cf1f98afcbce'])
            sample_points(dataset, 
                          nworkers=params.nworkers, 
                          density=params.sampling_density,
                          overwrite=params.overwrite)
            
        if params.splits:
            from .preprocessing.splits import splits
            dataset = DatasetInterface3DSSG(self.path)
            splits(dataset, train_validate_percentage=params.train_validate_percentage, overwrite=params.overwrite)


class SGPNDataModule(SceneGraphDataModule):
    def __init__(self, 
                 path: Path = "./dataset", 
                 num_workers: int = 0, 
                 batch_size: int = 1, 
                 shuffle: bool = False, 
                 runtime_sampling: SamplingArgs = SamplingArgs(), 
                 normalize_weights: bool = True, 
                 multi_relations: bool = True, 
                 cache: bool = False, 
                 preprocess_params: PreprocessingArgs = PreprocessingArgs(),
                 point_source: Literal["raw", "sampled"] = "raw",
                 point_colors: bool = False,
                 point_normals: bool = False,
                 samples_node: int = 128, # for sampling method "separate"
                 samples_relationship: int = 256, # for sampling method "separate"
                 point_augmentation: bool = False,
                 edge_data_type: str = "points",
                 edge_descriptor_type: str = "points",
                 ) -> None:
        super().__init__(path, num_workers, batch_size, shuffle, runtime_sampling, normalize_weights, multi_relations, cache, preprocess_params)
        self.save_hyperparameters()
        self.ds_class = SceneGraphSGPNDataset


class SSGPointDataModule(SceneGraphDataModule):
    def __init__(self, 
                 path: Path = "./dataset", 
                 num_workers: int = 0, 
                 batch_size: int = 1, 
                 shuffle: bool = False, 
                 runtime_sampling: SamplingArgs = SamplingArgs(), 
                 normalize_weights: bool = True, 
                 multi_relations: bool = True, 
                 cache: bool = False, 
                 preprocess_params: PreprocessingArgs = PreprocessingArgs(),
                 point_source: Literal["raw", "sampled"] = "sampled",
                 point_colors: bool = True,
                 point_normals: bool = True,
                 point_samples: int = 4096, # total number of points per scene
                 point_augmentation: bool = False
                 ) -> None:
        super().__init__(path, num_workers, batch_size, shuffle, runtime_sampling, normalize_weights, multi_relations, cache, preprocess_params)
        self.save_hyperparameters()
        self.ds_class = SceneGraphSSGPointDataset