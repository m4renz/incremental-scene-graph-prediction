# TODO: port this to torchmetrics!

import torch
import numpy as np
import torch.nn.functional as F


def get_gt(objs_target, rels_target, edges, multi_rel_outputs):
    gt_edges = []
    for edge_index in range(len(edges)):
        idx_eo = edges[edge_index][0]
        idx_os = edges[edge_index][1]
        target_eo = objs_target[idx_eo]
        target_os = objs_target[idx_os]
        target_rel = []
        if multi_rel_outputs:
            assert rels_target.ndim == 2
            for i in range(rels_target.shape[-1]):
                if rels_target[edge_index][i] == 1:
                    target_rel.append(i)
        else:
            assert rels_target.ndim == 1
            if rels_target[edge_index] > 0:  # not None
                target_rel.append(rels_target[edge_index])
        gt_edges.append((target_eo, target_os, target_rel))
    return gt_edges


def get_gt(objs_target, rels_target, edges, multi_rel_outputs):  # noqa: F811
    gt_edges = []

    for edge_index in range(len(edges)):
        idx_eo = edges[edge_index][0]
        idx_os = edges[edge_index][1]
        target_eo = objs_target[idx_eo]
        target_os = objs_target[idx_os]
        target_rel = []
        if multi_rel_outputs:
            assert rels_target.ndim == 2
            for i in range(rels_target.shape[-1]):
                if rels_target[edge_index][i] == 1:
                    target_rel.append(i)
        else:
            assert rels_target.ndim == 1
            if rels_target[edge_index] > 0:  # not None
                target_rel.append(rels_target[edge_index])
        gt_edges.append((target_eo, target_os, target_rel))
    return gt_edges


def evaluate_topk_object(objs_pred, objs_target, topk):
    res = []
    for obj in range(len(objs_pred)):
        obj_pred = objs_pred[obj]
        sorted_idx = torch.sort(obj_pred, descending=True)[1]
        gt = objs_target[obj]
        index = 1
        for idx in sorted_idx:
            if obj_pred[gt] >= obj_pred[idx] or index > topk:
                break
            index += 1
        res.append(index)
    return np.asarray(res)


def evaluate_topk_predicate(rels_preds, gt_edges, multi_rel_outputs, topk, confidence_threshold=0.5, epsilon=0.02):
    res = []
    for rel in range(len(rels_preds)):
        rel_pred = rels_preds[rel]
        # make the 'none' confidence the highest, if none of the rel classes are bigger than confidence_threshold
        # which means 'none' prediction in the multi binary cross entropy approach.
        # if multi_rel_outputs:
        #     if rel_pred.max() < confidence_threshold:
        #         rel_pred[0] = rel_pred.max() + epsilon

        sorted_conf_matrix, sorted_idx = torch.sort(rel_pred, descending=True)
        temp_topk = []
        rels_target = gt_edges[rel][2]

        if len(rels_target) == 0:  # no gt relation
            indices = torch.where(sorted_conf_matrix < confidence_threshold)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = sorted(indices)[0].item() + 1

            temp_topk.append(index)

        for gt in rels_target:
            index = 1
            for idx in sorted_idx:
                if rel_pred[gt] >= rel_pred[idx] or index > topk:
                    break
                index += 1
            temp_topk.append(index)

        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            res.append(tmp - counter)
            counter += 1
        # res += temp_topk
    return np.asarray(res)


def evaluate_triplet_topk(
    objs_pred,
    rels_pred,
    gt_rel,
    edges,
    multi_rel_outputs,
    topk,
    confidence_threshold=0.5,
    epsilon=0.02,
    use_clip=False,
    obj_topk=None,
):
    res, triplet = [], []
    if not use_clip:
        # convert the score from log_softmax to softmax
        objs_pred = np.exp(objs_pred)
    else:
        # convert the score to softmax
        objs_pred = F.softmax(objs_pred, dim=-1)

    if not multi_rel_outputs:
        rels_pred = np.exp(rels_pred)

    sub_scores, obj_scores, rel_scores = [], [], []

    for edge in range(len(edges)):
        edge_from = edges[edge][0]
        edge_to = edges[edge][1]
        rel_predictions = rels_pred[edge]
        sub = objs_pred[edge_from]
        obj = objs_pred[edge_to]

        if obj_topk is not None:
            sub_pred = obj_topk[edge_from]
            obj_pred = obj_topk[edge_to]

        node_score = torch.einsum("n,m->nm", sub, obj)
        conf_matrix = torch.einsum("nl,m->nlm", node_score, rel_predictions)
        conf_matrix_1d = conf_matrix.reshape(-1)
        sorted_conf_matrix, sorted_args_1d = torch.sort(conf_matrix_1d, descending=True)

        # just take topk
        sorted_conf_matrix = sorted_conf_matrix[:topk]
        sorted_args_1d = sorted_args_1d[:topk]

        sub_gt = gt_rel[edge][0]
        obj_gt = gt_rel[edge][1]
        rel_gt = gt_rel[edge][2]
        temp_topk, tmp_triplet = [], []

        if len(rel_gt) == 0:  # no gt relation

            # check how strongly the model predicts the none-relationship (= all values below threshold).
            # the index is the actual number of predicted relationships
            indices = torch.where(sorted_conf_matrix < confidence_threshold)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = sorted(indices)[0].item() + 1
            temp_topk.append(index)
            if obj_topk is not None:
                tmp_triplet.append([sub_gt.cpu(), sub_pred, obj_gt.cpu(), obj_pred, -1])
            else:
                tmp_triplet.append([sub_gt.cpu(), obj_gt.cpu(), -1])

        for predicate in rel_gt:  # for multi class case
            try:
                gt_conf = conf_matrix[sub_gt, obj_gt, predicate]
            except IndexError:
                print("IndexError: ", sub_gt, obj_gt, predicate)
                raise IndexError
            indices = torch.where(sorted_conf_matrix == gt_conf)[0]
            if len(indices) == 0:
                index = topk + 1
            else:
                index = sorted(indices)[0].item() + 1
            temp_topk.append(index)
            if obj_topk is not None:
                tmp_triplet.append([sub_gt.cpu(), sub_pred, obj_gt.cpu(), obj_pred, predicate])
            else:
                tmp_triplet.append([sub_gt.cpu(), obj_gt.cpu(), predicate])

            sub_scores.append(sub)
            obj_scores.append(obj)
            rel_scores.append(rel_predictions)

        temp_topk = sorted(temp_topk)
        counter = 0
        for tmp in temp_topk:
            res.append(tmp - counter)
            counter += 1
        triplet += tmp_triplet

    return np.asarray(res), np.array(triplet), sub_scores, obj_scores, rel_scores


def get_mean_recall(triplet_rank, cls_matrix, topk=[50, 100]):
    if len(cls_matrix) == 0:
        return np.array([0, 0])

    mean_recall = [[] for _ in range(len(topk))]
    cls_num = int(cls_matrix.max())
    for i in range(cls_num):
        cls_rank = triplet_rank[cls_matrix[:, -1] == i]
        if len(cls_rank) == 0:
            continue
        for idx, top in enumerate(topk):
            mean_recall[idx].append((cls_rank <= top).sum() / len(cls_rank))
    mean_recall = np.array(mean_recall, dtype=np.float32)
    return mean_recall.mean(axis=1)


def remap_hetero_gt_subgraph(batch, rel_type):

    obj_1 = batch.y_dict[rel_type[0]]
    obj_2 = batch.y_dict[rel_type[2]]
    edge_index = batch.edge_index_dict[rel_type]

    n_1 = obj_1.shape[0]

    mapped_index = torch.zeros_like(edge_index, dtype=torch.long)
    mapped_index[0] = edge_index[0]
    mapped_index[1] = edge_index[1] + n_1

    mapped_obj = torch.cat([obj_1, obj_2], dim=0)

    assert torch.max(mapped_index) < mapped_obj.shape[0]

    return mapped_obj, mapped_index


def triple_recall(
    node_pred: torch.Tensor,
    edge_pred: torch.Tensor,
    node_gt: torch.Tensor,
    edge_gt: torch.Tensor,
    edge_index: torch.Tensor,
    k: int = 1,
    num_edge_classes: int = None,
) -> float:
    """
    Calculate the mean triple recall for predicted nodes and edges.
    Args:
        node_pred (torch.Tensor): Predicted node labels with shape (num_nodes, num_classes).
        edge_pred (torch.Tensor): Predicted edge labels with shape (num_edges, num_edge_classes).
        node_gt (torch.Tensor): Ground truth node labels with shape (num_nodes,).
        edge_gt (torch.Tensor): Ground truth edge labels with shape (num_edges, num_edge_classes) or (num_edges,).
        edge_index (torch.Tensor): Edge indices with shape (2, num_edges).
        k (int, optional): Number of top triples to consider for recall calculation. Default is 1.
        num_edge_classes (int, optional): Number of edge classes. Must be provided if edge_gt is not 2D.
            Default is None.
    Returns:
        int: positive hits.
        int: total number of gt triples.
        torch.Tensor: TopK scores.
        torch.Tensor: TopK edge indices.
        torch.Tensor: TopK triples.
    """
    if len(edge_gt) == 0:
        return 0.0
    if edge_gt.ndim == 2:
        num_edge_classes = edge_gt.size(1)
    else:
        if num_edge_classes is None:
            raise ValueError("num_edge_classes must be provided if edge_gt is not 2D")

    assert node_pred.ndim == 2
    num_node_classes = node_pred.size(1)
    assert node_gt.max() < num_node_classes

    assert edge_index.size(0) == 2 and edge_index.size(1) == edge_gt.size(0), "edge_index must be COO format"

    triples = []
    triples_gt = []
    scores = []
    scores_edges = []
    scores_triples = []
    hits = 0

    i = np.arange(num_node_classes)
    e = np.arange(num_edge_classes)
    j = np.arange(num_node_classes)

    # Use meshgrid to generate all combinations
    I, E, J = np.meshgrid(i, e, j, indexing="ij")

    # Stack the arrays and reshape to get the desired triples
    triples = np.stack([I, E, J], axis=-1).reshape(-1, 3)
    triples = torch.tensor(triples, device=node_pred.device)

    for e_i, edge in enumerate(edge_index.t()):
        edge_label = torch.where(edge_gt[e_i])[0]  # get positive indices from multilabel edge_gt

        for e in edge_label:
            # TODO: build gt triples using sparse matrices?
            triples_gt.append((torch.stack([node_gt[edge[0]], e, node_gt[edge[1]]]), edge))

        # TODO: don't iterate through edges, make huge preds vector for all triples and multiply
        node_pred_edge_0 = node_pred[edge[0]]
        node_pred_edge_1 = node_pred[edge[1]]
        edge_pred_e_i = edge_pred[e_i]

        # TODO: nur beste Nodes?
        # TODO: topk pro spo prediction?
        # TODO: top1 node classes, topk edge classes?
        # TODO: threshold classifications?
        score = node_pred_edge_0[:, None, None] * edge_pred_e_i[None, :, None] * node_pred_edge_1[None, None, :]  # wie in Wald et al.
        score = score.flatten()
        scores.append(score)

        scores_edges.extend([edge for i in range(num_node_classes * num_edge_classes * num_node_classes)])
        scores_triples.extend(triples)

    scores = torch.concat(scores)
    scores_edges = torch.stack(scores_edges)
    scores_triples = torch.stack(scores_triples)

    scores_sorted, index_sorted = torch.sort(scores, descending=True)

    topk_scores = scores_sorted[:k]
    topk_edges = scores_edges[index_sorted][:k]
    topk_triples = scores_triples[index_sorted][:k]

    for gt_triple, edge in triples_gt:
        # if edge in topk_edges:
        _topk_triples = topk_triples[torch.where(torch.all(topk_edges == edge, dim=1))[0]]
        for triple in _topk_triples:
            if torch.all(triple == gt_triple):
                hits += 1
                break

    return hits, len(triples_gt), topk_scores, topk_edges, topk_triples


def triple_recall_scores(
    node_pred: torch.Tensor,
    edge_pred: torch.Tensor,
    node_gt: torch.Tensor,
    edge_gt: torch.Tensor,
    edge_index: torch.Tensor,
    num_edge_classes: int = None,
):

    if not torch.any(edge_gt):
        return None, None, None, None

    if edge_gt.ndim == 2:
        num_edge_classes = edge_gt.size(1)
    else:
        if num_edge_classes is None:
            raise ValueError("num_edge_classes must be provided if edge_gt is not 2D")

    assert node_pred.ndim == 2
    num_node_classes = node_pred.size(1)
    assert node_gt.max() < num_node_classes
    assert edge_index.size(0) == 2 and edge_index.size(1) == edge_gt.size(0), "edge_index must be COO format"

    # Generate all (s, p, o) triples
    i = torch.arange(num_node_classes, device=node_pred.device)
    e = torch.arange(num_edge_classes, device=node_pred.device)
    j = torch.arange(num_node_classes, device=node_pred.device)
    I, E, J = torch.meshgrid(i, e, j, indexing="ij")
    triples = torch.stack([I, E, J], dim=-1).reshape(-1, 3)  # (num_triples, 3)

    # Prepare predictions
    src_nodes = edge_index[0]  # (num_edges,)
    dst_nodes = edge_index[1]  # (num_edges,)

    node_pred_src = node_pred[src_nodes]  # (num_edges, num_node_classes)
    node_pred_dst = node_pred[dst_nodes]  # (num_edges, num_node_classes)
    edge_pred_all = edge_pred  # (num_edges, num_edge_classes)

    # Compute scores for all triples per edge: (e, s, p, o)
    score_tensor = (
        node_pred_src[:, :, None, None] * edge_pred_all[:, None, :, None] * node_pred_dst[:, None, None, :]
    )  # shape: (num_edges, s, p, o)

    score_tensor = score_tensor.reshape(edge_pred.size(0), -1)  # (num_edges, num_triples)
    num_triples = score_tensor.shape[1]

    # Repeat edge index for each triple
    edge_index_expanded = edge_index.t().unsqueeze(1).expand(-1, num_triples, -1)  # (num_edges, num_triples, 2)
    scores_edges = edge_index_expanded.reshape(-1, 2)  # (num_edges * num_triples, 2)

    # Repeat triples for each edge
    scores_triples = triples.repeat(edge_pred.size(0), 1)  # (num_edges * num_triples, 3)

    # Flatten scores
    scores = score_tensor.flatten()  # (num_edges * num_triples,)

    # Sort scores
    scores_sorted, index_sorted = torch.sort(scores, descending=True)

    # Ground truth triples: still loop-based due to variable number of positives
    triples_gt = []
    for e_i, edge in enumerate(edge_index.t()):
        edge_label = torch.where(edge_gt[e_i])[0]
        for e in edge_label:
            triples_gt.append(torch.cat((node_gt[edge[0]].unsqueeze(0), e.unsqueeze(0), node_gt[edge[1]].unsqueeze(0), edge)))

    if len(triples_gt) > 0:
        triples_gt = torch.stack(triples_gt)
    else:
        triples_gt = torch.empty((0, 5), dtype=torch.long, device=node_pred.device)

    return scores_sorted, scores_edges[index_sorted], scores_triples[index_sorted], triples_gt


def triple_recall_hits(topk_edges, topk_triples, triples_gt):
    """
    Calculates the number of ground truth triples that are correctly recalled among the top-k predicted triples.

    Args:
        topk_edges (torch.Tensor): Tensor containing the top-k predicted edges.
        topk_triples (torch.Tensor): Tensor containing the top-k predicted triples corresponding to the edges.
        triples_gt (Iterable[Tuple[torch.Tensor, Any]]): Iterable of tuples, each containing a ground truth triple and its associated edge.

    Returns:
        Tuple[int, int]: A tuple containing:
            - hits (int): The number of ground truth triples that are found among the top-k predictions.
            - total (int): The total number of ground truth triples.
    """

    if triples_gt is None:
        return 0, 1e-12

    hits = 0
    triples = triples_gt[:, :3]
    gt_edges = triples_gt[:, 3:]

    for gt_triple, edge in zip(triples, gt_edges):
        # if edge in topk_edges:
        _topk_triples = topk_triples[torch.where(torch.all(topk_edges == edge, dim=1))[0]]
        for triple in _topk_triples:
            if torch.all(triple == gt_triple):
                hits += 1
                break

    return hits, len(triples_gt)


def is_logits(t: torch.Tensor, multilabel=False):

    if torch.all(t >= 0.0) and torch.all(t <= 1.0):
        if multilabel:
            return False
        else:
            s = torch.sum(t, dim=1)
            if torch.allclose(s, torch.ones_like(s)):
                return False
    return True
