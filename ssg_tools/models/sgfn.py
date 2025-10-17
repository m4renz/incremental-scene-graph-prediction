# Code based on code from https://github.com/ShunChengWu/3DSSG/tree/cvpr21
# Copyright (c) 2021, ShunChengWu
# All rights reserved.


import lightning as L
import torch
from typing import Optional, Literal
import torch.optim as optim
from lightning.pytorch.cli import OptimizerCallable, LRSchedulerCallable
import torchmetrics
from torchmetrics import Accuracy, F1Score, Recall

from ssg_tools.models.pointnet import PointNetEncoder, PointNetDecoder
from ssg_tools.models.edge_attention_gnn import GraphEdgeAttenNetworkLayers
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Batch
import torch.nn.functional as F

from ssg_tools.utils.metrics import MeanNodeEdgeAccuracy, TripleRecallWithScores
from ssg_tools.utils.metrics_util import triple_recall_scores


class Gen_edge_descriptor(MessagePassing):
    """A sequence of scene graph convolution layers"""

    def __init__(self, flow="source_to_target"):
        super().__init__(flow=flow)

    def forward(self, descriptor, edges_indices):
        size = self._check_input(edges_indices, None)
        coll_dict = self._collect(self._user_args, edges_indices, size, {"x": descriptor})
        msg_kwargs = self.inspector.collect_param_data("message", coll_dict)
        edge_feature = self.message(**msg_kwargs)
        return edge_feature

    def message(self, x_i, x_j):
        # source_to_target
        # (j, i)
        # 0-2: centroid, 3-5: std, 6-8:dims, 9:volume, 10:length
        # to
        # 0-2: offset centroid, 3-5: offset std, 6-8: dim log ratio, 9: volume log ratio, 10: length log ratio
        edge_feature = torch.zeros_like(x_i)
        # centroid offset
        edge_feature[:, 0:3] = x_i[:, 0:3] - x_j[:, 0:3]
        # std  offset
        edge_feature[:, 3:6] = x_i[:, 3:6] - x_j[:, 3:6]
        # dim log ratio
        edge_feature[:, 6:9] = torch.log(x_i[:, 6:9] / x_j[:, 6:9])
        # volume log ratio
        edge_feature[:, 9] = torch.log(x_i[:, 9] / x_j[:, 9])
        # length log ratio
        edge_feature[:, 10] = torch.log(x_i[:, 10] / x_j[:, 10])
        # edge_feature, *_ = self.ef(edge_feature.unsqueeze(-1))
        return edge_feature.unsqueeze(-1)


class SGFN(L.LightningModule):
    def __init__(
        self,
        node_point_size: int = 3,
        edge_descriptor_size: int = 11,
        use_spatial: bool = True,
        batch_norm: bool = False,
        feature_transform: bool = True,
        node_feature_size: int = 256,
        edge_feature_size: int = 256,
        gnn_dim_attention: int = 256,
        gnn_n_layers: int = 2,
        gnn_n_attention_heads: int = 8,
        gnn_aggr: str = "max",
        gnn_attention: str = "fat",
        gnn_use_edge: bool = True,
        gnn_drop_out_attention: Optional[float] = 0.5,
        n_node_classes: int = 28,
        n_edge_classes: int = 16,
        weight_edge_mode: Literal["bg", "dynamic"] = "dynamic",
        lambda_node: float = 0.1,
        weighting: bool = True,
        multilabel_edges: bool = True,
        optimizer: Optional[OptimizerCallable] = None,
        lr_scheduler: Optional[LRSchedulerCallable] = None,
        scheduler_metric: str = "val_loss",  # only for scheduler
        scheduler_frequency: int = 1,
        sync_dist: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        node_feature_size_adjusted = node_feature_size
        if use_spatial:
            node_feature_size_adjusted -= edge_descriptor_size - 3  # ignore centroid #TODO: was does this do?
        self.flow = "target_to_source"  # we want the mess

        if batch_norm:
            norm = "batch"
        else:
            norm = None

        self.node_feature_extractor = PointNetEncoder(
            global_feat=True,
            norm=norm,
            feature_transform=feature_transform,
            input_transform=False,
            point_size=node_point_size,
            output_size=node_feature_size_adjusted,
        )  # (x,y,z)

        self.edge_feature_extractor = PointNetEncoder(
            global_feat=True,
            norm=norm,
            feature_transform=feature_transform,
            input_transform=False,
            point_size=edge_descriptor_size,
            output_size=edge_feature_size,
        )

        self.gcn = GraphEdgeAttenNetworkLayers(
            node_feature_size,
            edge_feature_size,
            gnn_dim_attention,
            gnn_n_layers,
            gnn_n_attention_heads,
            gnn_aggr,
            flow=self.flow,
            attention=gnn_attention,
            use_edge=gnn_use_edge,
            drop_out_attention=gnn_drop_out_attention,
        )

        self.node_classifier = PointNetDecoder(n_node_classes, node_feature_size, batch_norm=batch_norm, drop_out=True)
        self.edge_classifier = PointNetDecoder(
            n_edge_classes, edge_feature_size, batch_norm=batch_norm, drop_out=True, multilabel=multilabel_edges
        )

        self.weighting = weighting
        self.multilabel_edges = multilabel_edges

        self.weight_edge_mode = weight_edge_mode
        self.lambda_node = lambda_node

        self.train_node_metrics = torchmetrics.MetricCollection(
            {
                "mean_acc_node_1": Accuracy(task="multiclass", average="macro", num_classes=n_node_classes, top_k=1),
                "f1_node_1": F1Score(task="multiclass", average="macro", num_classes=n_node_classes, top_k=1),
                "mean_acc_node_5": Accuracy(task="multiclass", average="macro", num_classes=n_node_classes, top_k=5),
                "f1_node_5": F1Score(task="multiclass", average="macro", num_classes=n_node_classes, top_k=5),
            },
            prefix="train_",
        )

        self.train_edge_metrics = torchmetrics.MetricCollection(
            {
                "mean_acc_edge": Accuracy(task="multilabel", average="macro", num_labels=n_edge_classes),
                "rec_edge": Recall(task="multilabel", average="macro", num_labels=n_edge_classes),
            },
            prefix="train_",
        )

        self.val_node_metrics = self.train_node_metrics.clone(prefix="val_")
        self.val_edge_metrics = self.train_edge_metrics.clone(prefix="val_")
        self.test_node_metrics = self.train_node_metrics.clone(prefix="test_")
        self.test_edge_metrics = self.train_edge_metrics.clone(prefix="test_")

        self.hyperparam_metric = MeanNodeEdgeAccuracy(n_node_classes, n_edge_classes, average="macro", k=5)
        self.recall_at_k_50 = TripleRecallWithScores(k=50)
        self.recall_at_k_100 = TripleRecallWithScores(k=100)
        self.unseen_mean_acc_1 = Accuracy(task="multiclass", average="macro", num_classes=n_node_classes, top_k=1)
        self.unseen_mean_acc_5 = Accuracy(task="multiclass", average="macro", num_classes=n_node_classes, top_k=5)

        self.optimizer_cls = optimizer
        self.lr_scheduler_cls = lr_scheduler
        self.scheduler_metric = scheduler_metric
        self.scheduler_frequency = scheduler_frequency
        self.sync_dist = sync_dist

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

    def forward(self, batch: Batch):

        try:
            batch_ids = batch["node"].batch
            obj_points = batch["node"].points
            descriptor = batch["node"].descriptor
            edge_indices = batch["node", "to", "node"].edge_index
        except KeyError:
            batch_ids = batch.batch
            obj_points = batch.points.reshape(batch.points.shape[0], 3, -1)
            descriptor = batch.descriptor
            edge_indices = batch.edge_index

        obj_feature = self.node_feature_extractor(obj_points)

        if self.hparams.use_spatial:
            tmp = descriptor[:, 3:].clone()
            tmp[:, 6:] = tmp[:, 6:].log()  # only log on volume and length
            obj_feature = torch.cat([obj_feature, tmp], dim=1)

        # Create edge feature vector
        with torch.no_grad():
            edge_feature = Gen_edge_descriptor(flow=self.flow)(descriptor, edge_indices)

        edge_feature = self.edge_feature_extractor(edge_feature)

        obj_center = descriptor[:, :3].clone()
        gcn_obj_feature, gcn_rel_feature, probs = self.gcn(obj_feature, edge_feature, edge_indices, obj_center, batch_ids)

        rel_cls = self.edge_classifier(gcn_rel_feature)

        obj_logits = self.node_classifier(gcn_obj_feature)

        return obj_logits, rel_cls

    def training_step(self, batch: Batch, batch_no: int):
        batch_size = batch.batch_size

        try:
            obj_gt = batch["node"].y  # ground truth of nodes
            edge_gt = batch["node", "to", "node"].y  # ground_truth of edges
        except KeyError:
            obj_gt = batch.y
            edge_gt = batch.edge_label

        obj_pred, edge_pred = self.forward(batch)

        loss, loss_obj, loss_edges = self.calculate_loss(obj_pred, obj_gt, edge_pred, edge_gt)

        self.log("train_loss_node", loss_obj, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=self.sync_dist)
        self.log("train_loss_edge", loss_edges, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=self.sync_dist)

        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=self.sync_dist)

        self.train_node_metrics(obj_pred, obj_gt)
        self.log_dict(self.train_node_metrics, on_step=False, on_epoch=True, batch_size=batch.batch_size)

        if edge_gt.numel() > 0:
            self.train_edge_metrics(edge_pred, edge_gt)
            self.log_dict(self.train_edge_metrics, on_step=False, on_epoch=True, batch_size=batch.batch_size)
        # self.compute_training_metrics(obj_pred, obj_gt, edge_pred, edge_gt, edge_indices, batch_size)

        return loss

    def validation_step(self, batch: Batch, batch_no: int):

        try:
            obj_gt = batch["node"].y  # ground truth of nodes
            edge_gt = batch["node", "to", "node"].y  # ground_truth of edges
        except KeyError:
            obj_gt = batch.y
            edge_gt = batch.edge_label

        obj_pred, edge_pred = self.forward(batch)

        loss, loss_obj, loss_edges = self.calculate_loss(obj_pred, obj_gt, edge_pred, edge_gt)

        self.log(
            "val_loss_node",
            loss_obj,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
            sync_dist=self.sync_dist,
        )
        self.log(
            "val_loss_edge",
            loss_edges,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
            sync_dist=self.sync_dist,
        )
        self.log("val_loss", loss, on_step=False, on_epoch=True, batch_size=batch.batch_size, sync_dist=self.sync_dist)

        self.val_node_metrics(obj_pred, obj_gt)
        self.log_dict(self.val_node_metrics, on_step=False, on_epoch=True, batch_size=batch.batch_size)

        if edge_gt.numel() > 0:
            self.val_edge_metrics(edge_pred, edge_gt)
            self.log_dict(self.val_edge_metrics, on_step=False, on_epoch=True, batch_size=batch.batch_size)
            self.hyperparam_metric(obj_pred, edge_pred, obj_gt, edge_gt)
            self.log("hyperparam_metric", self.hyperparam_metric, on_step=False, on_epoch=True, batch_size=batch.batch_size)

    def test_step(self, batch: Batch, batch_no: int):

        node_logits, edge_logits = self.forward(batch)
        gt_nodes = batch.y
        gt_edges = batch.edge_label

        self.test_node_metrics(node_logits, gt_nodes)
        self.log_dict(self.test_node_metrics, batch_size=batch.batch_size, on_step=False, on_epoch=True)

        if gt_edges.numel() > 0:
            self.test_edge_metrics(edge_logits, gt_edges)
            self.log_dict(self.test_edge_metrics, batch_size=batch.batch_size, on_step=False, on_epoch=True)

        self.log_recall_at_k(node_logits, edge_logits, gt_nodes, gt_edges, batch.edge_index, batch.batch_size, batch.ptr)

        if hasattr(batch, "unseen"):
            pred_unseen = node_logits[batch.unseen] if torch.any(batch.unseen) else None
            gt_unseen = gt_nodes[batch.unseen] if torch.any(batch.unseen) else None
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

    def configure_optimizers(self):

        optimizer_cls = self.optimizer_cls
        lr = 0.0001
        trainable_params = [
            {"params": self.node_feature_extractor.parameters()},
            {"params": self.edge_feature_extractor.parameters()},
            {"params": self.gcn.parameters()},
            {"params": self.node_classifier.parameters()},
            {"params": self.edge_classifier.parameters()},
        ]
        if optimizer_cls is None:

            # default optimizer from SGPN implementation
            optimizer = optim.AdamW(trainable_params, lr=lr, amsgrad=False, weight_decay=False)
        else:
            optimizer = optimizer_cls(trainable_params)

        scheduler_cls = self.lr_scheduler_cls
        if scheduler_cls is not None:
            scheduler = scheduler_cls(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": self.scheduler_metric,
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> torch.Any:

        obj_pred, edge_pred = self.forward(batch)
        return obj_pred.detach().cpu(), edge_pred.detach().cpu(), batch

    def calculate_loss(self, obj_pred, obj_gt, edge_pred, edge_gt):

        loss_obj = F.cross_entropy(obj_pred, obj_gt)

        if torch.isnan(loss_obj):
            raise ValueError("NaN encountered in object loss")

        if self.multilabel_edges:
            if self.weight_edge_mode == "dynamic":
                batch_mean = torch.sum(edge_gt, dim=(0))
                zeros = (edge_gt.sum(-1) == 0).sum().unsqueeze(0)
                batch_mean = torch.cat([zeros, batch_mean], dim=0)
                weight = torch.abs(1.0 / (torch.log(batch_mean + 1) + 1))  # +1 to prevent 1 /log(1) = inf
                # if ignore_none_rel:
                #    weight[0] = 0
                #    weight *= 1e-2 # reduce the weight from ScanNet
                # if 'NONE_RATIO' in self.mconfig:
                #    weight[0] *= self.mconfig.NONE_RATIO

                weight[torch.where(weight == 0)] = weight[0].clone()  # if not ignore_none_rel else 0# * 1e-3
                weight = weight[1:]
            elif self.weight_edge_mode == "none":
                weight = None
            else:
                raise NotImplementedError(f"unknown weight_edge_mode {self.weight_edge_mode}")

            loss_edges = F.binary_cross_entropy(edge_pred, edge_gt, weight=weight)
        else:
            if self.weight_edge_mode == "dynamic":
                one_hot_gt_rel = torch.nn.functional.one_hot(edge_gt, num_classes=self.hparams.n_edge_classes)
                batch_mean = torch.sum(one_hot_gt_rel, dim=(0), dtype=torch.float)
                weight = torch.abs(1.0 / (torch.log(batch_mean + 1) + 1))  # +1 to prevent 1 /log(1) = inf
                # if ignore_none_rel:
                #    weight[0] = 0 # assume none is the first relationship
                #    weight *= 1e-2 # reduce the weight from ScanNet
            elif self.weight_edge_mode == "none":
                weight = None
            loss_edges = F.nll_loss(edge_pred, edge_gt, weight=weight)

        if torch.isnan(loss_edges):
            raise ValueError("NaN encountered in edge loss")

        lambda_edge = 1.0
        lambda_node = self.lambda_node
        lambda_max = max(lambda_edge, lambda_node)
        lambda_edge /= lambda_max
        lambda_node /= lambda_max

        loss = lambda_node * loss_obj + lambda_edge * loss_edges

        if torch.isnan(loss):
            raise ValueError("NaN encountered in loss")

        return loss, loss_obj, loss_edges

        # self.compute_validation_metrics(obj_pred, obj_gt, edge_pred, edge_gt, edge_indices, batch_size)

    # def compute_training_metrics(self, obj_pred, obj_gt, rel_pred, rel_gt, edge_indices, batch_size):
    #         """
    #         Helper function to test the metrics from the original implementation before porting them
    #         """
    #         from ssg_tools.utils.metrics_util import evaluate_topk_object, get_gt, evaluate_topk_predicate

    #         # compute metric
    #         top_k_obj = evaluate_topk_object(obj_pred.detach().cpu(), obj_gt.detach().cpu(), topk=11)
    #         gt_edges = get_gt(
    #             obj_gt.detach().cpu(), rel_gt.detach().cpu(), edge_indices.detach().cpu().T, self.multilabel_edges
    #         )
    #         top_k_rel = evaluate_topk_predicate(rel_pred.detach().cpu(), gt_edges, self.multilabel_edges, topk=6)

    #         obj_topk_list = [(top_k_obj <= i).sum() / len(top_k_obj) for i in [1, 5, 10]]
    #         rel_topk_list = [(top_k_rel <= i).sum() / len(top_k_rel) for i in [1, 3, 5]]
    #         self.log_dict(
    #             {
    #                 "train_obj_acc_at_1": obj_topk_list[0],
    #                 "train_obj_acc_at_5": obj_topk_list[1],
    #                 "train_obj_acc_at_10": obj_topk_list[2],
    #                 "train_predicate_acc_at_1": rel_topk_list[0],
    #                 "train_predicate_acc_at_3": rel_topk_list[1],
    #                 "train_predicate_acc_at_5": rel_topk_list[2],
    #             },
    #             batch_size=batch_size,
    #         )
    # def on_validation_epoch_start(self) -> None:
    #     # init the lists to track the metrics during validation
    #     self.topk_obj_list = np.array([])
    #     self.topk_rel_list = np.array([])
    #     self.topk_triplet_list = np.array([])
    #     self.cls_matrix_list = []
    #     self.edge_feature_list = []
    #     self.sub_scores_list, self.obj_scores_list, self.rel_scores_list = [], [], []

    # def compute_validation_metrics(self, obj_pred, obj_gt, rel_pred, rel_gt, edge_indices, batch_size):
    #     from ssg_tools.utils.metrics_util import (
    #         evaluate_topk_object,
    #         get_gt,
    #         evaluate_topk_predicate,
    #         evaluate_triplet_topk,
    #     )

    #     with torch.no_grad():
    #         obj_pred = obj_pred.detach().cpu()
    #         rel_pred = rel_pred.detach().cpu()
    #         edge_indices = edge_indices.detach().cpu().T
    #         top_k_obj = evaluate_topk_object(obj_pred, obj_gt.detach().cpu(), topk=11)
    #         gt_edges = get_gt(obj_gt.detach().cpu(), rel_gt.detach().cpu(), edge_indices, self.multilabel_edges)
    #         top_k_rel = evaluate_topk_predicate(rel_pred, gt_edges, self.multilabel_edges, topk=6)
    #         top_k_triplet, cls_matrix, sub_scores, obj_scores, rel_scores = evaluate_triplet_topk(
    #             obj_pred,
    #             rel_pred,
    #             gt_edges,
    #             edge_indices,
    #             self.multilabel_edges,
    #             topk=101,
    #             use_clip=False,
    #             obj_topk=top_k_obj,
    #         )
    #     self.topk_obj_list = np.concatenate((self.topk_obj_list, top_k_obj))
    #     self.topk_rel_list = np.concatenate((self.topk_rel_list, top_k_rel))
    #     self.topk_triplet_list = np.concatenate((self.topk_triplet_list, top_k_triplet))
    #     if cls_matrix is not None:
    #         self.cls_matrix_list.extend(cls_matrix)
    #         self.sub_scores_list.extend(sub_scores)
    #         self.obj_scores_list.extend(obj_scores)
    #         self.rel_scores_list.extend(rel_scores)

    # def on_validation_epoch_end(self) -> None:
    #     from ssg_tools.utils.metrics_util import get_mean_recall

    #     self.cls_matrix_list = np.stack(self.cls_matrix_list)
    #     self.sub_scores_list = np.stack(self.sub_scores_list)
    #     self.obj_scores_list = np.stack(self.obj_scores_list)
    #     self.rel_scores_list = np.stack(self.rel_scores_list)
    #     mean_recall = get_mean_recall(self.topk_triplet_list, self.cls_matrix_list)

    #     obj_acc_1 = (self.topk_obj_list <= 1).sum() / len(self.topk_obj_list)
    #     obj_acc_5 = (self.topk_obj_list <= 5).sum() / len(self.topk_obj_list)
    #     obj_acc_10 = (self.topk_obj_list <= 10).sum() / len(self.topk_obj_list)
    #     rel_acc_1 = (self.topk_rel_list <= 1).sum() / len(self.topk_rel_list)
    #     rel_acc_3 = (self.topk_rel_list <= 3).sum() / len(self.topk_rel_list)
    #     rel_acc_5 = (self.topk_rel_list <= 5).sum() / len(self.topk_rel_list)
    #     triplet_acc_50 = (self.topk_triplet_list <= 50).sum() / len(self.topk_triplet_list)
    #     triplet_acc_100 = (self.topk_triplet_list <= 100).sum() / len(self.topk_triplet_list)

    #     def compute_mean_predicate(cls_matrix_list, topk_pred_list):
    #         cls_dict = defaultdict(list)
    #         # for i in range(26):
    #         #    cls_dict[i] = []

    #         # gather measurements for each class separately
    #         for idx, j in enumerate(cls_matrix_list):
    #             if j[-1] != -1:
    #                 cls_dict[j[-1]].append(topk_pred_list[idx])

    #         predicate_mean_1, predicate_mean_3, predicate_mean_5 = [], [], []
    #         for v in cls_dict.values():
    #             # for i in range(26):
    #             # l = len(cls_dict[i])
    #             # if l > 0:
    #             v = np.array(v)
    #             m_1 = (v <= 1).sum() / len(v)
    #             m_3 = (v <= 3).sum() / len(v)
    #             m_5 = (v <= 5).sum() / len(v)
    #             predicate_mean_1.append(m_1)
    #             predicate_mean_3.append(m_3)
    #             predicate_mean_5.append(m_5)

    #         predicate_mean_1 = np.mean(predicate_mean_1)
    #         predicate_mean_3 = np.mean(predicate_mean_3)
    #         predicate_mean_5 = np.mean(predicate_mean_5)

    #         return predicate_mean_1, predicate_mean_3, predicate_mean_5

    #     rel_acc_mean_1, rel_acc_mean_3, rel_acc_mean_5 = compute_mean_predicate(
    #         self.cls_matrix_list, self.topk_rel_list
    #     )

    #     logs = {
    #         "val_obj_acc_at_1": obj_acc_1,
    #         "val_obj_acc_at_5": obj_acc_5,
    #         "val_obj_acc_at_10": obj_acc_10,
    #         "val_predicate_acc_at_1": rel_acc_1,
    #         "val_predicate_mean_acc_at_1": rel_acc_mean_1,
    #         "val_predicate_acc_at_3": rel_acc_3,
    #         "val_predicate_mean_acc_at_3": rel_acc_mean_3,
    #         "val_predicate_acc_at_5": rel_acc_5,
    #         "val_predicate_mean_acc_at_5": rel_acc_mean_5,
    #         "val_triplet_acc_at_50": triplet_acc_50,
    #         "val_triplet_acc_at_100": triplet_acc_100,
    #         "val_mean_recall_at_50": mean_recall[0],
    #         "val_mean_recall_at_100": mean_recall[1],
    #     }
    #     self.log_dict(logs)
