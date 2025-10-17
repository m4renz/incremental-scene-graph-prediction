import dataclasses
import json
from pathlib import Path
import warnings
from typing import Dict, Literal, Sequence

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics import Accuracy, F1Score, Recall
from ssg_tools.models.pointnet import PointNetEncoder
from ssg_tools.utils.metrics import MeanNodeEdgeAccuracy, TripleRecallWithScores
from ssg_tools.utils.metrics_util import triple_recall_scores
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import to_hetero

from ssg_tools.models.gnn_layers import (
    EdgeFeatureEmbedder,
    SceneGAT,
    SceneSAGE,
    SceneHGT,
    SceneHAN,
    HeteroClassifier,
    EdgeMLP,
    LinearClassifier,
    EdgeFeatureEmbedderHomo,
    HierarchicalHeteroConv,
)

warnings.filterwarnings("ignore", category=FutureWarning, message="You are using")


def concat_hetero_edges(gt, pred):

    assert set(gt.keys()) == set(pred.keys()), "Keys of ground truth and prediction do not match"

    _gt = []
    _pred = []

    for k, v in pred.items():
        _gt.append(gt[k])
        _pred.append(v)

    return torch.cat(_gt, dim=0), torch.cat(_pred, dim=0)


@dataclasses.dataclass
class HeteroSGLayout:
    node_types: Sequence[str] = ("old", "new")
    edge_types: Sequence[Sequence[str]] = (
        ("old", "to", "old"),
        ("new", "to", "new"),
        ("new", "to", "old"),
        ("old", "to", "new"),
    )

    def to_metadata(self) -> tuple:
        return (list(self.node_types), list(tuple(et) for et in self.edge_types))


class KSGNBase(L.LightningModule):

    def __init__(
        self,
        lr: float = 0.0001,
        weights_node: torch.Tensor | str | None = None,
        weights_edge: torch.Tensor | str | None = None,
        weights_path: str | None = None,
        sync_dist: bool = False,
        gamma_edge: float = 1.0,
        nodes_to_predict: Sequence[str] = ("new",),
        edges_to_predict: Sequence[tuple[str, str, str]] = (("new", "to", "new"),),
        embedding_type: Literal["label", "numberbatch", "clip"] | None = None,
        pointnet_output_dim: int = 256,
        num_node_classes: int = 27,  # RIO27 without "-"
        num_edge_classes: int = 16,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=kwargs.get("log_hyperparameters", True))

        self.nodes_to_predict = nodes_to_predict
        self.edges_to_predict = edges_to_predict
        self.lr = lr
        self.sync_dist = sync_dist
        self.embedding_type = embedding_type
        self.pointnet_output_dim = pointnet_output_dim
        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.test_recall = True

        if weights_path is not None:
            if weights_node is not None and isinstance(weights_node, str):
                if Path(weights_path).suffix == ".pkl":
                    weights_node = torch.load(weights_path)[weights_node]
                elif Path(weights_path).suffix == ".json":
                    weights_node = json.load(open(weights_path))[weights_node]
                    weights_node = torch.tensor(weights_node, dtype=torch.float32)
            if weights_edge is not None and isinstance(weights_edge, str):
                if Path(weights_path).suffix == ".pkl":
                    weights_edge = torch.load(weights_path)[weights_edge]
                elif Path(weights_path).suffix == ".json":
                    weights_edge = json.load(open(weights_path))[weights_edge]
                    weights_edge = torch.tensor(weights_edge, dtype=torch.float32)

        self.register_buffer("weights_node", weights_node)
        if weights_edge is not None:
            weights_edge = weights_edge * gamma_edge

        self.register_buffer("pos_weights_edge", weights_edge)

        self.point_net = PointNetEncoder(
            global_feat=True,
            norm="batch",
            input_transform=False,
            feature_transform=False,
            output_size=self.pointnet_output_dim,
            init_weights=True,
        )

        self.train_node_metrics = torchmetrics.MetricCollection(
            {
                "mean_acc_node_1": Accuracy(task="multiclass", average="macro", num_classes=num_node_classes, top_k=1),
                "f1_node_1": F1Score(task="multiclass", average="macro", num_classes=num_node_classes, top_k=1),
                "mean_acc_node_5": Accuracy(task="multiclass", average="macro", num_classes=num_node_classes, top_k=5),
                "f1_node_5": F1Score(task="multiclass", average="macro", num_classes=num_node_classes, top_k=5),
            },
            prefix="train_",
        )

        self.train_edge_metrics = torchmetrics.MetricCollection(
            {
                "mean_acc_edge": Accuracy(task="multilabel", average="macro", num_labels=num_edge_classes),
                "rec_edge": Recall(task="multilabel", average="macro", num_labels=num_edge_classes),
            },
            prefix="train_",
        )

        self.val_node_metrics = self.train_node_metrics.clone(prefix="val_")
        self.val_edge_metrics = self.train_edge_metrics.clone(prefix="val_")
        self.test_node_metrics = self.train_node_metrics.clone(prefix="test_")
        self.test_edge_metrics = self.train_edge_metrics.clone(prefix="test_")

        self.hyperparam_metric = MeanNodeEdgeAccuracy(num_node_classes, num_edge_classes, average="macro", k=5)
        self.recall_at_k_50 = TripleRecallWithScores(k=50)
        self.recall_at_k_100 = TripleRecallWithScores(k=100)
        self.unseen_mean_acc_1 = Accuracy(task="multiclass", average="macro", num_classes=num_node_classes, top_k=1)
        self.unseen_mean_acc_5 = Accuracy(task="multiclass", average="macro", num_classes=num_node_classes, top_k=5)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.05)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def log_loss(
        self, loss: torch.Tensor, node_loss: torch.Tensor, edge_loss: torch.Tensor, split: Literal["train", "val"], batch_size: int
    ):

        self.log(
            f"{split}_loss_node",
            node_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=self.sync_dist,
        )
        self.log(
            f"{split}_loss_edge",
            edge_loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=self.sync_dist,
        )
        self.log(f"{split}_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=self.sync_dist)

    def log_node_metrics(
        self, pred: torch.Tensor, gt: torch.Tensor, split: Literal["train", "val", "test"], batch_size: int, batch: Batch = None
    ):

        if split == "train":
            self.train_node_metrics.update(pred, gt)
            self.log_dict(self.train_node_metrics, on_step=False, on_epoch=True, batch_size=batch_size)
        elif split == "val":
            self.val_node_metrics.update(pred, gt)
            self.log_dict(self.val_node_metrics, on_step=False, on_epoch=True, batch_size=batch_size)
        elif split == "test":
            self.test_node_metrics.update(pred, gt)
            self.log_dict(self.test_node_metrics, on_step=False, on_epoch=True, batch_size=batch_size)

            # if isinstance(batch, Data) and hasattr(batch, "unseen"):
            if hasattr(batch, "unseen"):
                pred_unseen = pred[batch.unseen] if torch.any(batch.unseen) else None
                gt_unseen = gt[batch.unseen] if torch.any(batch.unseen) else None
            elif isinstance(batch, HeteroData) and "new" in getattr(batch, "node_types", []) and hasattr(batch["new"], "unseen"):
                pred_unseen = pred[batch["new"].unseen] if torch.any(batch["new"].unseen) else None
                gt_unseen = gt[batch["new"].unseen] if torch.any(batch["new"].unseen) else None
            else:
                pred_unseen = gt_unseen = None

            if pred_unseen is not None and gt_unseen is not None:  # Not all batches have unseen nodes
                logs = {
                    "test_unseen_mean_acc_1": self.unseen_mean_acc_1(pred_unseen, gt_unseen),
                    "test_unseen_mean_acc_5": self.unseen_mean_acc_5(pred_unseen, gt_unseen),
                }
            else:
                logs = {}

            self.log_dict(logs, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        else:
            raise AttributeError(f"Split has to be in [`train`, `val`, `test`], got {split}")

    def log_edge_metrics(self, pred: torch.Tensor, gt: torch.Tensor, split: Literal["train", "val", "test"], batch_size: int):

        if gt.numel() > 0:
            if split == "train":
                self.train_edge_metrics.update(pred, gt)
                self.log_dict(self.train_edge_metrics, on_step=False, on_epoch=True, batch_size=batch_size)
            elif split == "val":
                self.val_edge_metrics.update(pred, gt)
                self.log_dict(self.val_edge_metrics, on_step=False, on_epoch=True, batch_size=batch_size)
            elif split == "test":
                self.test_edge_metrics.update(pred, gt)
                self.log_dict(self.test_edge_metrics, on_step=False, on_epoch=True, batch_size=batch_size)
            else:
                raise AttributeError(f"Split has to be in [`train`, `val`, `test`], got {split}")

    def log_recall_at_k(
        self,
        node_logits: torch.Tensor,
        edge_logits: torch.Tensor,
        gt_nodes: torch.Tensor,
        gt_edges: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int,
        batch_ptr: torch.tensor,
    ):

        for _i in range(len(batch_ptr) - 1):
            i = batch_ptr[_i]
            j = batch_ptr[_i + 1]

            edge_mask = torch.all(torch.logical_and(edge_index.t() >= i, edge_index.t() < j), dim=1)

            if not torch.any(edge_mask):
                continue  # some frames don't have edges due to single nodes or distance between nodes

            _edge_index = edge_index.t()[edge_mask].t() - i  # extracted graphs need edge index from 0 to n_nodes
            _, edges, triples, ground_truth = triple_recall_scores(
                node_logits[i:j], edge_logits[edge_mask], gt_nodes[i:j], gt_edges[edge_mask], _edge_index
            )

            if ground_truth is not None:
                self.recall_at_k_50.update(edges, triples, ground_truth),
                self.recall_at_k_100.update(edges, triples, ground_truth),

        self.log(
            "test_triple_recall_at_50",
            self.recall_at_k_50,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "test_triple_recall_at_100",
            self.recall_at_k_100,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        torch.cuda.empty_cache()

    @staticmethod
    def get_unseen(pred: dict, d: HeteroData, device=None):
        new_nodes_idx = torch.arange(0, d["new"].num_nodes)
        if device is not None:
            new_nodes_idx = new_nodes_idx.to(device)
        unseen_nodes = new_nodes_idx[~torch.isin(new_nodes_idx, d["old", "to", "new"].edge_index[1].unique())]

        if len(unseen_nodes) == 0:
            return None, None

        unseen_pred = pred["new"][unseen_nodes]
        unseen_y = d["new"].y[unseen_nodes]

        return unseen_pred, unseen_y


class IncrementalKSGN(KSGNBase):

    def __init__(
        self,
        metadata: Sequence[Sequence[str | Sequence[str]]] | Dict,  # metadata of HeteroData
        edge_emb_dim: int = 128,
        gnn_hidden_dim: int = 256,
        gnn_out_dim: int = 256,
        gnn_num_layers: int = 2,
        dropout: float = 0.5,
        norm=None,
        gnn_type: Literal["sage", "gat", "gcn", "hgt", "han", "hierarchical", "kg", "gatv2", "kg_att", "sage_edge", "kg_v2"] = "sage",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if isinstance(metadata, HeteroSGLayout):
            print("WARNING: Using HeteroSGLayout is deprecated. Use a list of node and edge types instead.")
            self.metadata = metadata.to_metadata()
        elif isinstance(metadata, dict):
            try:
                self.metadata = [metadata["node_types"], [tuple(edge_type) for edge_type in metadata["edge_types"]]]
            except KeyError as e:
                raise KeyError(e).add_note("Got metadata: " + str(metadata.keys()))
        elif isinstance(metadata, (list, tuple)):
            assert len(metadata) == 2, "Got wrong metadata format"
            self.metadata = metadata

        assert all([n in self.metadata[0] for n in self.nodes_to_predict]), "Nodes to predict not in metadata"
        assert all([e in self.metadata[1] for e in self.edges_to_predict]), "Edges to predict not in metadata"

        feature_edges = [e for e in self.metadata[1] if "kg" not in e]  # kg nodes have no edge descriptor

        self.edge_feature_embedder = EdgeFeatureEmbedder(
            output_dim=edge_emb_dim, norm=norm, dropout=dropout, edge_types=feature_edges, device=self.device
        )

        if gnn_type == "gat":
            self.gnn = SceneGAT(
                node_channels=(-1, -1),
                edge_channels=edge_emb_dim,
                hidden_channels=gnn_hidden_dim,
                out_channels=gnn_out_dim,
                heads=kwargs.get("heads", 4),
                num_layers=gnn_num_layers,
                dropout=dropout,
                norm=norm,
            )
        elif gnn_type == "sage" or gnn_type == "gcn":
            if gnn_type == "gcn":
                print(
                    "[WARNING]: in the past, sage was always been used even though gcn was selected ."
                    + "GCN is not implemented, using SAGE instead."
                )
            self.gnn = SceneSAGE(
                node_channels=-1,
                hidden_channels=gnn_hidden_dim,
                out_channels=gnn_out_dim,
                num_layers=gnn_num_layers,
                dropout=dropout,
                norm=norm,
            )
        elif gnn_type == "hgt":
            self.gnn = SceneHGT(
                node_channels=-1,
                hidden_channels=gnn_hidden_dim,
                out_channels=gnn_out_dim,
                heads=kwargs.get("heads", 4),
                num_layers=gnn_num_layers,
                dropout=dropout,
                metadata=self.metadata,
                norm=norm,
            )
        elif gnn_type == "han":
            self.gnn = SceneHAN(
                node_channels=-1,
                hidden_channels=gnn_hidden_dim,
                out_channels=gnn_out_dim,
                heads=kwargs.get("heads", 4),
                num_layers=gnn_num_layers,
                dropout=dropout,
                attn_dropout=kwargs.get("attn_dropout", 0.5),
                metadata=self.metadata,
                norm=norm,
            )

        elif gnn_type == "hierarchical":
            self.gnn = HierarchicalHeteroConv(
                edge_types=self.metadata[1],
                node_channels=(-1, -1),
                hidden_channels=gnn_hidden_dim,
                out_channels=gnn_out_dim,
                edge_channels=kwargs.get("edge_channels", 10),
                num_layers=gnn_num_layers,
                dropout=dropout,
                norm=norm,
            )

        self.node_classifier = HeteroClassifier(
            in_channels=gnn_out_dim, out_channels=self.num_node_classes, types=self.nodes_to_predict, dropout=dropout, norm=norm
        )

        self.edge_classifier = EdgeMLP(
            in_channels=gnn_out_dim * 2 + edge_emb_dim,
            out_channels=self.num_edge_classes,
            types=self.edges_to_predict,
            dropout=dropout,
            norm=norm,
        )

        if isinstance(self.gnn, (SceneSAGE, SceneGAT)):  # SceneGAT and SceneHAN hetero by default
            self.gnn = to_hetero(self.gnn, self.metadata)

        if self.embedding_type == "label":
            emb_size = 1
        elif self.embedding_type == "numberbatch":
            emb_size = 300
        elif self.embedding_type == "clip":
            emb_size = 768
        else:
            emb_size = 0

        # dummy forward pass to initialize the model (necessary for distributed training)
        with torch.no_grad():

            dummy_data = {
                "old": torch.zeros((2, self.pointnet_output_dim + 11 + emb_size)).to(self.device),
                "new": torch.zeros((2, self.pointnet_output_dim + 11)).to(self.device),
            }

            dummy_edge_index = {k: torch.tensor([[0], [0]]).long().to(self.device) for k in self.metadata[1]}
            if isinstance(self.gnn, HierarchicalHeteroConv):
                dummy_edge_attr = {("old", "h", "old"): torch.zeros(1, kwargs.get("edge_channels", 10)).to(self.device)}
            else:
                dummy_edge_attr = None
            self.gnn(dummy_data, dummy_edge_index, edge_features=dummy_edge_attr)

    def forward(self, batch: Batch | HeteroData):  # Add HeteroData for typing

        # encode the point cloud
        for node in self.metadata[0]:
            try:
                point_emb = self.point_net(
                    batch[node].points.reshape(batch[node].points.shape[0], 3, 256)
                )  # TODO: make this parametrizable?
                batch[node].x = torch.concat([point_emb, batch[node].descriptor], dim=1)
            except AttributeError:
                # if a node type doesn't have points
                continue

        if self.embedding_type is not None:
            if self.embedding_type == "label":
                batch["old"].x = torch.cat([batch["old"].x, batch["old"].corrupted_labels.unsqueeze(-1)], dim=1)
            else:
                batch["old"].x = torch.cat([batch["old"].x, batch["old"].embeddings], dim=1)

        # build edge features
        edge_features = self.edge_feature_embedder(batch.descriptor_dict, batch.edge_index_dict)

        # for k, v in edge_features.items():
        #     batch[k].edge_features = v

        try:
            edge_feature_dict = batch.edge_feature_dict
        except KeyError:
            edge_feature_dict = None

        # message passing
        gnn_node_feature = self.gnn(batch.x_dict, batch.edge_index_dict, edge_features=edge_feature_dict)

        node_logits = self.node_classifier(gnn_node_feature)
        edge_logits = self.edge_classifier(edge_features, gnn_node_feature, batch.edge_index_dict)

        return node_logits, edge_logits

    def training_step(self, batch: Batch, batch_no: int):

        node_logits, edge_logits = self.forward(batch)
        gt_nodes = batch.y_dict
        gt_edges = batch.edge_label_dict

        node_loss = 0
        edge_loss = 0

        for node in self.nodes_to_predict:
            node_loss += F.cross_entropy(node_logits[node], gt_nodes[node], weight=self.weights_node)

        for edge in self.edges_to_predict:
            if gt_edges[edge].numel() > 0:  # otherwise loss is nan
                edge_loss += F.binary_cross_entropy_with_logits(edge_logits[edge], gt_edges[edge], pos_weight=self.pos_weights_edge)

        loss = node_loss + edge_loss

        self.log_loss(loss, node_loss, edge_loss, split="train", batch_size=batch.batch_size)

        logits, gt = self.flatten_pred_and_gt(node_logits, gt_nodes)
        self.log_node_metrics(logits, gt, split="train", batch_size=batch.batch_size)

        logits, gt = self.flatten_pred_and_gt(edge_logits, gt_edges)
        self.log_edge_metrics(logits, gt, split="train", batch_size=batch.batch_size)

        return loss

    def validation_step(self, batch: Batch, batch_no: int):

        node_logits_dict, edge_logits_dict = self.forward(batch)
        gt_nodes = batch.y_dict
        gt_edges = batch.edge_label_dict

        node_loss = 0
        edge_loss = 0

        for node in self.nodes_to_predict:
            node_loss += F.cross_entropy(node_logits_dict[node], gt_nodes[node], weight=self.weights_node)

        for edge in self.edges_to_predict:
            if gt_edges[edge].numel() > 0:  # otherwise loss is nan
                edge_loss += F.binary_cross_entropy_with_logits(edge_logits_dict[edge], gt_edges[edge], pos_weight=self.pos_weights_edge)

        loss = node_loss + edge_loss
        if torch.isnan(loss):
            raise ValueError("NaN encountered in loss")

        self.log_loss(loss, node_loss, edge_loss, split="val", batch_size=batch.batch_size)

        node_logits, node_gt = self.flatten_pred_and_gt(node_logits_dict, gt_nodes)
        self.log_node_metrics(node_logits, node_gt, split="val", batch_size=batch.batch_size)

        edge_logits, edge_gt = self.flatten_pred_and_gt(edge_logits_dict, gt_edges)
        self.log_edge_metrics(edge_logits, edge_gt, split="val", batch_size=batch.batch_size)

        if edge_gt.numel() > 0:
            self.hyperparam_metric(node_logits, edge_logits, node_gt, edge_gt)
            self.log("hyperparam_metric", self.hyperparam_metric, on_step=False, on_epoch=True, batch_size=batch.batch_size)

    def test_step(self, batch: Batch, batch_no: int):

        node_logits, edge_logits = self.forward(batch)
        gt_nodes = batch.y_dict
        gt_edges = batch.edge_label_dict
        ptr = batch["new"].ptr  # TODO: adjustthis for multiple nodes to predict
        if len(self.nodes_to_predict) > 1 and self.nodes_to_predict != "new":
            raise ValueError("testing for other node types than `new` is not implemented yet.")

        logits, gt = self.flatten_pred_and_gt(node_logits, gt_nodes)
        self.log_node_metrics(logits, gt, split="test", batch_size=batch.batch_size, batch=batch)

        logits, gt = self.flatten_pred_and_gt(edge_logits, gt_edges)
        self.log_edge_metrics(logits, gt, split="test", batch_size=batch.batch_size)

        if self.test_recall:
            d = self.flatten_topk_recall(node_logits, edge_logits, gt_nodes, gt_edges, batch.edge_index_dict)
            self.log_recall_at_k(d.node_logits, d.edge_logits, d.gt_nodes, d.gt_edges, d.edge_index, batch.batch_size, ptr)

    def flatten_pred_and_gt(self, pred, gt):

        pred_flat = []
        gt_flat = []

        for k, v in pred.items():
            pred_flat.append(v)
            gt_flat.append(gt[k])

        return torch.concat(pred_flat, dim=0), torch.concat(gt_flat, dim=0)

    def flatten_topk_recall(self, node_logits, edge_logits, gt_nodes, gt_edges, edge_index_dict):

        data = HeteroData()

        for n in self.nodes_to_predict:
            data[n].node_logits = node_logits[n]
            data[n].gt_nodes = gt_nodes[n]

        for e in self.edges_to_predict:
            data[e].edge_index = edge_index_dict[e]
            data[e].edge_logits = edge_logits[e]
            data[e].gt_edges = gt_edges[e]

        return data.to_homogeneous()


class IncrementalKSGNHomo(KSGNBase):

    def __init__(
        self,
        edge_emb_dim: int = 128,
        gnn_hidden_dim: int = 256,
        gnn_out_dim: int = 256,
        heads: int = 8,
        dropout: float = 0.5,
        norm=None,
        gnn_num_layers: int = 2,
        gnn_type: Literal["gat", "sage"] = "gat",
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.edge_feature_embedder = EdgeFeatureEmbedderHomo(output_dim=edge_emb_dim, norm=norm, dropout=dropout)

        if gnn_type == "gat":
            self.gnn = SceneGAT(
                node_channels=self.pointnet_output_dim + 11,
                edge_channels=edge_emb_dim,
                hidden_channels=gnn_hidden_dim,
                out_channels=gnn_out_dim,
                heads=heads,
                num_layers=gnn_num_layers,
                dropout=dropout,
                norm=norm,
            )
        elif gnn_type == "sage":
            self.gnn = SceneSAGE(
                node_channels=self.pointnet_output_dim + 11,
                hidden_channels=gnn_hidden_dim,
                out_channels=gnn_out_dim,
                num_layers=gnn_num_layers,
                dropout=dropout,
                norm=norm,
            )

        self.node_classifier = LinearClassifier(in_channels=gnn_out_dim, out_channels=self.num_node_classes, dropout=dropout, norm=norm)

        self.edge_classifier = LinearClassifier(
            in_channels=gnn_out_dim * 2 + edge_emb_dim, out_channels=self.num_edge_classes, dropout=dropout, norm=norm
        )

    def forward(self, batch: Batch):

        point_emb = self.point_net(batch.points.reshape(batch.points.shape[0], 3, -1))
        batch.x = torch.concat([point_emb, batch.descriptor], dim=1)

        # build edge features
        edge_features = self.edge_feature_embedder(batch.descriptor, batch.edge_index)

        # message passing
        gnn_node_feature = self.gnn(batch.x, batch.edge_index, edge_features=edge_features)

        node_logits = self.node_classifier(gnn_node_feature)

        # one-liner of death
        # build [source_node_feature, edge_feature, target_node_feature] for each edge as input of edge classifier
        edge_logits = self.edge_classifier(
            torch.cat([gnn_node_feature[batch.edge_index[0]], edge_features, gnn_node_feature[batch.edge_index[1]]], dim=1)
        )

        return node_logits, edge_logits

    def training_step(self, batch: Batch, batch_no: int):

        node_logits, edge_logits = self.forward(batch)
        gt_nodes = batch.y
        gt_edges = batch.edge_label

        node_loss = 0
        edge_loss = 0

        node_loss = F.cross_entropy(node_logits, gt_nodes, weight=self.weights_node)

        if gt_edges.numel() > 0:  # otherwise loss is nan
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits, gt_edges, pos_weight=self.pos_weights_edge)

        loss = node_loss + edge_loss

        self.log_loss(loss, node_loss, edge_loss, split="train", batch_size=batch.batch_size)
        self.log_node_metrics(node_logits, gt_nodes, split="train", batch_size=batch.batch_size)
        self.log_edge_metrics(edge_logits, gt_edges, split="train", batch_size=batch.batch_size)

        return loss

    def validation_step(self, batch: Batch, batch_no: int):

        node_logits, edge_logits = self.forward(batch)
        gt_nodes = batch.y
        gt_edges = batch.edge_label

        node_loss = 0
        edge_loss = 0

        node_loss = F.cross_entropy(node_logits, gt_nodes, weight=self.weights_node)

        if gt_edges.numel() > 0:  # otherwise loss is nan
            edge_loss = F.binary_cross_entropy_with_logits(edge_logits, gt_edges, pos_weight=self.pos_weights_edge)

        loss = node_loss + edge_loss

        if torch.isnan(loss):
            raise ValueError("NaN encountered in loss")

        self.log_loss(loss, node_loss, edge_loss, split="val", batch_size=batch.batch_size)
        self.log_node_metrics(node_logits, gt_nodes, split="val", batch_size=batch.batch_size)
        self.log_edge_metrics(edge_logits, gt_edges, split="val", batch_size=batch.batch_size)

        if gt_edges.numel() > 0:
            self.hyperparam_metric(node_logits, edge_logits, gt_nodes, gt_edges)
            self.log("hyperparam_metric", self.hyperparam_metric, on_step=False, on_epoch=True, batch_size=batch.batch_size)

    def test_step(self, batch: Batch, batch_no: int):

        node_logits, edge_logits = self.forward(batch)
        gt_nodes = batch.y
        gt_edges = batch.edge_label

        self.log_node_metrics(node_logits, gt_nodes, split="test", batch_size=batch.batch_size, batch=batch)
        self.log_edge_metrics(edge_logits, gt_edges, split="test", batch_size=batch.batch_size)
        if self.test_recall:
            self.log_recall_at_k(node_logits, edge_logits, gt_nodes, gt_edges, batch.edge_index, batch.batch_size, batch.ptr),


class KSGNRandomClassifier(KSGNBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        delattr(self, "point_net")
        delattr(self, "weights_node")
        delattr(self, "pos_weights_edge")

    def forward(self, batch: Batch):

        random_nodes = torch.randint(high=self.num_node_classes, size=(batch["new"].num_nodes,))
        out_nodes = torch.nn.functional.one_hot(random_nodes, self.num_node_classes).to(dtype=torch.float32, device=self.device)

        out_edges = torch.rand((batch["new", "to", "new"].num_edges, self.num_edge_classes)).to(dtype=torch.float32, device=self.device)
        # out_edges = torch.nn.functional.one_hot(random_edges, self.num_edge_classes).to(dtype=torch.float32, device=self.device)

        return out_nodes, out_edges

    def test_step(self, batch: Batch, batch_no: int):

        node_pred, edge_pred = self.forward(batch)
        gt_nodes = batch["new"].y
        gt_edges = batch["new", "to", "new"].edge_label
        ptr = batch["new"].ptr

        self.log_node_metrics(node_pred, gt_nodes, split="test", batch_size=batch.batch_size, batch=batch)
        self.log_edge_metrics(edge_pred, gt_edges, split="test", batch_size=batch.batch_size)

        if self.test_recall:
            self.log_recall_at_k(
                node_pred,
                edge_pred,
                gt_nodes,
                gt_edges,
                batch["new", "to", "new"].edge_index,
                batch_size=batch.batch_size,
                batch_ptr=ptr,
            )


class IncrementalKSGNLinear(IncrementalKSGNHomo):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        delattr(self, "gnn")
        # self.linear = Linear(kwargs.get("pointnet_output_dim", 256), kwargs.get("gnn_out_dim", 256))
        assert kwargs.get("pointnet_output_dim", 256) == kwargs.get(
            "gnn_out_dim", 256
        ), "pointnet_output_dim and gnn_out_dim must be equal for IncrementalKSGNLinear"

    def forward(self, batch):

        point_emb = self.point_net(batch.points.reshape(batch.points.shape[0], 3, -1))
        batch.x = torch.concat([point_emb, batch.descriptor], dim=1)

        # build edge features
        edge_features = self.edge_feature_embedder(batch.descriptor, batch.edge_index)

        node_logits = self.node_classifier(point_emb)

        # one-liner of death
        # build [source_node_feature, edge_feature, target_node_feature] for each edge as input of edge classifier
        edge_logits = self.edge_classifier(
            torch.cat([point_emb[batch.edge_index[0]], edge_features, point_emb[batch.edge_index[1]]], dim=1)
        )

        return node_logits, edge_logits


class IncrementalKSGNFull(KSGNBase):

    def __init__(
        self,
        edge_emb_dim=128,
        gnn_hidden_dim=256,
        gnn_out_dim=256,
        heads=8,
        dropout=0.5,
        norm=None,
        gnn_num_layers=2,
        gnn_type="sage",
        embedding_type: Literal["label", "numberbatch", "clip"] | None = None,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.embedding_type = embedding_type

        self.edge_feature_embedder = EdgeFeatureEmbedderHomo(output_dim=edge_emb_dim, norm=norm, dropout=dropout)

        feature_dim = 0
        if embedding_type == "label":
            feature_dim = 1
        elif embedding_type == "numberbatch":
            feature_dim = 300
        elif embedding_type == "clip":
            feature_dim = 768

        print("feature dim is", feature_dim)

        if gnn_type == "gat":
            self.gnn = SceneGAT(
                node_channels=self.pointnet_output_dim + feature_dim + 11,
                edge_channels=edge_emb_dim,
                hidden_channels=gnn_hidden_dim,
                out_channels=gnn_out_dim,
                heads=heads,
                num_layers=gnn_num_layers,
                dropout=dropout,
                norm=norm,
            )
        elif gnn_type == "sage":
            self.gnn = SceneSAGE(
                node_channels=self.pointnet_output_dim + feature_dim + 11,
                hidden_channels=gnn_hidden_dim,
                out_channels=gnn_out_dim,
                num_layers=gnn_num_layers,
                dropout=dropout,
                norm=norm,
            )

        self.node_classifier = LinearClassifier(in_channels=gnn_out_dim, out_channels=self.num_node_classes, dropout=dropout, norm=norm)

        self.edge_classifier = LinearClassifier(
            in_channels=gnn_out_dim * 2 + edge_emb_dim, out_channels=self.num_edge_classes, dropout=dropout, norm=norm
        )

    def forward(self, batch: Batch):

        point_emb = self.point_net(batch.points.reshape(batch.points.shape[0], 3, -1))
        batch.x = torch.concat([point_emb, batch.descriptor], dim=1)

        if self.embedding_type is not None:
            if self.embedding_type == "label":
                batch.x = torch.cat([batch.x, batch.corrupted_labels.unsqueeze(-1)], dim=1)
            else:
                batch.x = torch.cat([batch.x, batch.embeddings], dim=1)

        # build edge features
        edge_features = self.edge_feature_embedder(batch.descriptor, batch.edge_index)

        # message passing
        gnn_node_feature = self.gnn(batch.x, batch.edge_index, edge_features=edge_features)

        node_logits = self.node_classifier(gnn_node_feature)

        # one-liner of death
        # build [source_node_feature, edge_feature, target_node_feature] for each edge as input of edge classifier
        edge_logits = self.edge_classifier(
            torch.cat([gnn_node_feature[batch.edge_index[0]], edge_features, gnn_node_feature[batch.edge_index[1]]], dim=1)
        )

        return node_logits, edge_logits

    def training_step(self, batch: Batch, batch_no: int):

        node_logits, edge_logits = self.forward(batch)
        gt_nodes = batch.y
        gt_edges = batch.edge_label
        new_node_mask, new_edge_mask = self.get_new_mask(batch)

        node_loss = 0
        edge_loss = 0

        node_loss = F.cross_entropy(node_logits[new_node_mask], gt_nodes[new_node_mask], weight=self.weights_node)

        if gt_edges.numel() > 0:  # otherwise loss is nan
            edge_loss = F.binary_cross_entropy_with_logits(
                edge_logits[new_edge_mask], gt_edges[new_edge_mask], pos_weight=self.pos_weights_edge
            )

        loss = node_loss + edge_loss

        self.log_loss(loss, node_loss, edge_loss, split="train", batch_size=batch.batch_size)
        self.log_node_metrics(node_logits[new_node_mask], gt_nodes[new_node_mask], split="train", batch_size=batch.batch_size)
        self.log_edge_metrics(edge_logits[new_edge_mask], gt_edges[new_edge_mask], split="train", batch_size=batch.batch_size)

        return loss

    def validation_step(self, batch: Batch, batch_no: int):

        node_logits, edge_logits = self.forward(batch)
        gt_nodes = batch.y
        gt_edges = batch.edge_label
        new_node_mask, new_edge_mask = self.get_new_mask(batch)

        node_loss = 0
        edge_loss = 0

        node_loss = F.cross_entropy(node_logits[new_node_mask], gt_nodes[new_node_mask], weight=self.weights_node)

        if gt_edges.numel() > 0:  # otherwise loss is nan
            edge_loss = F.binary_cross_entropy_with_logits(
                edge_logits[new_edge_mask], gt_edges[new_edge_mask], pos_weight=self.pos_weights_edge
            )

        loss = node_loss + edge_loss

        if torch.isnan(loss):
            raise ValueError("NaN encountered in loss")

        self.log_loss(loss, node_loss, edge_loss, split="val", batch_size=batch.batch_size)
        self.log_node_metrics(node_logits[new_node_mask], gt_nodes[new_node_mask], split="val", batch_size=batch.batch_size)
        self.log_edge_metrics(edge_logits[new_edge_mask], gt_edges[new_edge_mask], split="val", batch_size=batch.batch_size)

        if gt_edges.numel() > 0:
            self.hyperparam_metric(node_logits[new_node_mask], edge_logits[new_edge_mask], gt_nodes[new_node_mask], gt_edges[new_edge_mask])
            self.log("hyperparam_metric", self.hyperparam_metric, on_step=False, on_epoch=True, batch_size=batch.batch_size)

    def test_step(self, batch: Batch, batch_no: int):

        node_logits, edge_logits = self.forward(batch)
        new_node_mask, new_edge_mask = self.get_new_mask(batch)

        new_batch = self.extract_new_batch(batch)

        self.log_node_metrics(node_logits[new_node_mask], new_batch.y, split="test", batch_size=new_batch.batch_size, batch=new_batch)
        self.log_edge_metrics(edge_logits[new_edge_mask], new_batch.edge_label, split="test", batch_size=new_batch.batch_size)
        if self.test_recall:
            self.log_recall_at_k(
                node_logits[new_node_mask],
                edge_logits[new_edge_mask],
                new_batch.y,
                new_batch.edge_label,
                new_batch.edge_index,
                new_batch.batch_size,
                new_batch.ptr,
            ),

    @staticmethod
    def extract_new_batch(batch: Batch):

        new_data_list = []

        for i in range(batch.num_graphs):
            data = batch.get_example(i)
            node_mask, edge_mask = IncrementalKSGNFull.get_new_mask(data)

            new_graph = data.subgraph(node_mask)
            new_data_list.append(new_graph)

        return Batch.from_data_list(new_data_list)

    @staticmethod
    def get_new_mask(batch):

        if isinstance(batch.node_types[0], (list, tuple)):
            new_node_mask = batch.node_type == batch.node_types[0].index("new")
        else:
            new_node_mask = batch.node_type == batch.node_types.index("new")

        if isinstance(batch.edge_types[0][0], (list, tuple)):
            if isinstance(batch.edge_types[0][0], list):
                new_edge_mask = batch.edge_type == batch.edge_types[0].index(["new", "to", "new"])
            else:
                new_edge_mask = batch.edge_type == batch.edge_types[0].index(("new", "to", "new"))
        else:
            if isinstance(batch.edge_types[0], list):
                new_edge_mask = batch.edge_type == batch.edge_types.index(["new", "to", "new"])
            else:
                new_edge_mask = batch.edge_type == batch.edge_types.index(("new", "to", "new"))

        return new_node_mask, new_edge_mask
