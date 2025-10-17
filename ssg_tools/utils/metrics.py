from __future__ import annotations
from typing import Literal
from torchmetrics import Metric
from torchmetrics.classification.stat_scores import _multilabel_stat_scores_format
import torch
from torchmetrics.utilities import dim_zero_cat
import tqdm
from ssg_tools.utils.metrics_util import triple_recall, is_logits, triple_recall_hits


class TopKMultilabelAccuracy(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self, num_labels, top_k: int = 1, threshold: float = 0.5):
        super().__init__()
        self.num_labels = num_labels
        self.top_k = top_k
        self.threshold = threshold

        # default_state = lambda: torch.zeros(self.num_labels, dtype=torch.long)
        self.add_state("accuracy_hit", default=torch.zeros(1, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("accuracy_count", default=torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum")
        # self.add_state("accuracy_count", default=default_state())

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds_clean, target = _multilabel_stat_scores_format(preds, target, num_labels=self.num_labels, threshold=self.threshold)
        if preds_clean.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        preds = preds[..., None]

        # sort each prediction by confidence
        preds_sorted, preds_sorted_idx = torch.sort(preds, descending=True, dim=1)

        target_mask = target > 0

        # exclusive prefix sum dimension 1
        correction_indices = torch.cat((torch.zeros((target.shape[0], 1), dtype=torch.long), torch.cumsum(target, dim=1)), dim=1)[:, :-1]

        # get the index of the prediction from the sorted values and increment it by one to avoid negative indices
        preds_top_idx = preds_sorted_idx[target_mask] + 1

        no_gt_mask = ~target_mask.any(dim=1)

        if no_gt_mask.sum() > 0:
            # filter only 'none' predictions (below threshold) without ground truth
            none_predictions_mask = torch.logical_and((preds < self.threshold).any(dim=1), no_gt_mask)
            lowest_none_pred = torch.argmin(preds_clean, dim=1)[none_predictions_mask] + 1
            self.accuracy_hit[0] += (lowest_none_pred <= self.top_k).sum()
            self.accuracy_count[0] += lowest_none_pred.shape[0]  # all 'none' predictions without ground truth

        correction_top_idx = correction_indices[target_mask]
        # correct the prefix indices with the appropriate corrections,
        # so multiple labels per prediction are not disadvantaged
        preds_top_idx -= correction_top_idx
        self.accuracy_hit[0] += (preds_top_idx <= self.top_k).sum()
        self.accuracy_count[0] += preds_top_idx.shape[0]  # all actual predictions

    def compute(self) -> torch.Tensor:
        total = dim_zero_cat(self.accuracy_hit).sum().float()
        count = dim_zero_cat(self.accuracy_count).sum()

        # prevent division by zero
        value = total / (count + 1e-12)
        return value


class SceneGraphRecallBase(Metric):
    # Set to True if the metric is differentiable else set to False
    is_differentiable = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def __init__(self, num_labels, k: int = 100, threshold: float = 0.5, multilabel: bool = True):
        super().__init__()
        self.num_labels = num_labels
        self.k = k
        self.threshold = threshold
        self.multilabel = multilabel

        self.add_state("recall_hit", default=lambda: torch.zeros(self.num_labels, dtype=torch.long))
        self.add_state("recall_count", default=lambda: torch.zeros(self.num_labels, dtype=torch.long))

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        if self.multilabel:
            preds_clean, target = _multilabel_stat_scores_format(preds, target, num_labels=self.num_labels, threshold=self.threshold)
        else:
            # TODO: enable single label prediction as well
            preds = preds.max(1)[1]
            target = target.reshape(-1, 1)

        if preds_clean.shape != target.shape:
            raise ValueError("preds and target must have the same shape")

        preds = preds[..., None]

        target_mask = target > 0
        preds = preds[target_mask]

        if preds.size(0) == 0:
            # shortcut if there are no predictions with ground truth in this iteration
            return

        preds_clean = preds_clean[target_mask]
        target = target[target_mask]
        target_labels = target_mask.nonzero()[:, 1]

        # sort the predictions by their score
        preds, sorter = torch.sort(preds, descending=True)

        # take the k most confident predictions
        k = self.k if preds.shape[0] > self.k else preds.shape[0]
        pred_clean_sorted = preds_clean[sorter][:k]
        target_sorted = target[sorter]
        target_sorted_k = target_sorted[:k]
        target_labels_sorted = target_labels[sorter]
        target_labels_sorted_k = target_labels_sorted[:k]

        correct = torch.logical_and(pred_clean_sorted, target_sorted_k).int()
        accum_tensor = torch.zeros_like(self.recall_hit)
        accum_tensor.index_put_((target_labels_sorted_k,), correct.long(), accumulate=True)
        self.recall_hit += accum_tensor
        # print("hits", self.recall_hit)

        accum_tensor = torch.zeros_like(self.recall_count)
        accum_tensor.index_put_((target_labels_sorted,), target[sorter].long(), accumulate=True)
        self.recall_count += accum_tensor
        # print("counts", self.recall_count)


class RecallAtK(SceneGraphRecallBase):
    # Set to True if the metric is differentiable else set to False
    is_differentiable = False

    # Set to True if the metric reaches it optimal value when the metric is maximized.
    # Set to False if it when the metric is minimized.
    higher_is_better = True

    # Set to True if the metric during 'update' requires access to the global metric
    # state for its calculations. If not, setting this to False indicates that all
    # batch states are independent and we will optimize the runtime of 'forward'
    full_state_update: bool = True

    def compute(self) -> torch.Tensor:
        total = dim_zero_cat(self.recall_hit).sum().float()
        count = dim_zero_cat(self.recall_count).sum()
        value = total / (count + 1e-12)
        return value


class MeanRecallAtK(SceneGraphRecallBase):
    def compute(self) -> torch.Tensor:
        total = dim_zero_cat(self.recall_hit).float()
        count = dim_zero_cat(self.recall_count)
        # print("total", total, count)
        values = (total / (count + 1e-12)).mean()  # compute the recall for each class separately and compute the mean afterwards
        # print("mean recall", self.k, values)
        return values  # the recall is the mean of all separate classes value


class TripleRecallAtK(Metric):
    def __init__(self, k=1, average: Literal["micro", "macro"] = "micro", **kwargs):
        super().__init__(**kwargs)
        self.add_state("node_preds", default=[], dist_reduce_fx="cat")
        self.add_state("edge_preds", default=[], dist_reduce_fx="cat")
        self.add_state("edge_index", default=[], dist_reduce_fx="cat")
        self.add_state("node_gt", default=[], dist_reduce_fx="cat")
        self.add_state("edge_gt", default=[], dist_reduce_fx="cat")
        self.add_state("num_nodes", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ptr_max", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("ptr", default=[], dist_reduce_fx="cat")

        self.k = k
        assert average in ["micro", "macro"], f"Average can be `micro` or `macro`, got {average}"
        self.average = average

    def update(
        self,
        node_pred: torch.Tensor,
        edge_pred: torch.Tensor,
        node_gt: torch.Tensor,
        edge_gt: torch.Tensor,
        edge_index: torch.Tensor,
        ptr: torch.Tensor,
    ):
        self.edge_index.append(edge_index + self.num_nodes)
        self.ptr.append(ptr[:-1] + self.ptr_max)
        self.node_preds.append(node_pred)
        self.edge_preds.append(edge_pred)
        self.node_gt.append(node_gt)
        self.edge_gt.append(edge_gt)
        self.num_nodes += node_pred.size(0)
        self.ptr_max = ptr[-1] + self.ptr_max

    def compute(self):
        node_preds = dim_zero_cat(self.node_preds)
        edge_preds = dim_zero_cat(self.edge_preds)
        node_gt = dim_zero_cat(self.node_gt)
        edge_gt = dim_zero_cat(self.edge_gt)
        edge_index = torch.cat(self.edge_index, dim=1)
        self.ptr.append(self.ptr_max.unsqueeze(0))
        ptr = torch.cat(self.ptr)

        if is_logits(node_preds):
            node_preds = torch.nn.functional.softmax(node_preds, dim=1)
        if is_logits(edge_preds, multilabel=True):
            edge_preds = torch.nn.functional.sigmoid(edge_preds)

        hits = []
        total = []

        for _i in range(len(ptr) - 1):
            i = ptr[_i]
            j = ptr[_i + 1]

            edge_mask = torch.all(torch.logical_and(edge_index.t() >= i, edge_index.t() < j), dim=1)

            if not torch.any(edge_mask):
                continue  # some frames don't have edges due to single nodes or distance between nodes

            _edge_index = edge_index.t()[edge_mask].t() - i  # extracted graphs need edge index from 0 to n_nodes
            _hits, _total, _, _, _ = triple_recall(
                node_preds[i:j], edge_preds[edge_mask], node_gt[i:j], edge_gt[edge_mask], _edge_index, self.k
            )
            hits.append(_hits)
            total.append(_total)

        if self.average == "micro":
            return torch.tensor(hits).sum() / torch.tensor(total).sum()
        else:  # "macro"
            return (torch.tensor(hits) / torch.tensor(total)).mean()


class MeanNodeEdgeAccuracy(Metric):

    def __init__(self, num_node_classes, num_edge_classes, average="micro", k=1, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.add_state("node_preds", default=[], dist_reduce_fx="cat")
        self.add_state("edge_preds", default=[], dist_reduce_fx="cat")
        self.add_state("node_gt", default=[], dist_reduce_fx="cat")
        self.add_state("edge_gt", default=[], dist_reduce_fx="cat")
        self.add_state("num_nodes", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_edges", default=torch.tensor(0), dist_reduce_fx="sum")

        self.num_node_classes = num_node_classes
        self.num_edge_classes = num_edge_classes
        self.average = average
        self.k = k
        self.t = threshold

    def update(self, node_pred: torch.Tensor, edge_pred: torch.Tensor, node_gt: torch.Tensor, edge_gt: torch.Tensor) -> torch.Tensor:

        if is_logits(node_pred, multilabel=False):
            node_pred = torch.nn.functional.softmax(node_pred, dim=1)
        if is_logits(edge_pred, multilabel=True):
            edge_pred = torch.nn.functional.sigmoid(edge_pred)

        self.node_preds.append(node_pred)
        self.edge_preds.append(edge_pred)
        self.node_gt.append(node_gt)
        self.edge_gt.append(edge_gt)
        self.num_nodes += node_pred.size(0)
        self.num_edges += edge_pred.size(0)

    def compute(self) -> torch.Tensor:

        node_preds = dim_zero_cat(self.node_preds)
        edge_preds = dim_zero_cat(self.edge_preds)
        node_gt = dim_zero_cat(self.node_gt)
        edge_gt = dim_zero_cat(self.edge_gt)

        # in the multilabel case, micro and macro averaging are the same and there is no topk
        correct_edge = (edge_preds > self.t).int() == edge_gt
        edge_acc = correct_edge.float().mean()

        if self.average == "micro":
            node_acc = torch.any(torch.topk(node_preds, self.k, dim=1).indices == node_gt.unsqueeze(1), dim=1).float().mean()
            return (node_acc + edge_acc) / 2
        elif self.average == "macro":
            # Calculate top-k indices for predictions
            topk_indices_node = torch.topk(node_preds, k=self.k, dim=1).indices
            # Compare top-k indices with ground truth labels
            correct_node = topk_indices_node == node_gt.unsqueeze(1)
            # Initialize an empty list to store accuracies for each class
            node_acc = []
            # Calculate accuracy for each class in a vectorized way
            for i in range(self.num_node_classes):
                # Mask for samples of class i
                mask = node_gt == i
                # Calculate top-k accuracy for class i and append to acc
                class_acc = correct_node[mask].any(dim=1).float().mean().item()
                node_acc.append(class_acc)

            return (torch.tensor(node_acc).mean() + edge_acc) / 2


class TripleRecallWithScores(Metric):
    def __init__(self, k=1, average: Literal["micro", "macro"] = "micro", disable_progress_bar=False, **kwargs):
        super().__init__(**kwargs)
        self.add_state("edges_sorted", default=[], dist_reduce_fx="cat")
        self.add_state("triples_sorted", default=[], dist_reduce_fx="cat")
        self.add_state("triples_gt", default=[], dist_reduce_fx="cat")
        self.add_state("num_graphs", default=torch.tensor(0), dist_reduce_fx="sum")

        self.k = k
        assert average in ["micro", "macro"], f"Average can be `micro` or `macro`, got {average}"
        self.average = average
        self.disable_progress_bar = disable_progress_bar

    def update(
        self,
        edges_sorted: torch.Tensor,
        triples_sorted: torch.Tensor,
        triples_gt: torch.Tensor,
    ):
        # Detach from computation graph, move to CPU, and clone to break any memory views
        self.edges_sorted.append(edges_sorted[: self.k].detach().cpu().clone())
        self.triples_sorted.append(triples_sorted[: self.k].detach().cpu().clone())
        self.triples_gt.append(triples_gt.detach().cpu().clone())
        self.num_graphs += 1

    def compute(self):

        edges_sorted = dim_zero_cat(self.edges_sorted)
        triples_sorted = dim_zero_cat(self.triples_sorted)
        # triples_gt = dim_zero_cat(self.triples_gt)

        hits = []
        total = []

        for _i, gt in tqdm.tqdm(
            zip(range(self.num_graphs.item()), self.triples_gt), desc=f"Recall@{self.k}: ", disable=self.disable_progress_bar
        ):
            i = self.k * _i
            j = self.k * (_i + 1)

            _hits, _total = triple_recall_hits(edges_sorted[i:j], triples_sorted[i:j], gt)

            hits.append(_hits)
            total.append(_total)

        if self.average == "micro":
            return torch.tensor(hits).sum() / torch.tensor(total).sum()
        else:  # "macro"
            return (torch.tensor(hits) / torch.tensor(total)).mean()
