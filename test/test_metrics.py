import torchmetrics
from ssg_tools.utils.metrics import TripleRecallAtK, TripleRecallWithScores  # , TripleRecallWithScores

from ssg_tools.utils.metrics_util import triple_recall_hits, triple_recall_scores, is_logits
from ssg_tools.models.ksgn import IncrementalKSGN
import torch
import pytest

from torch_geometric.data import HeteroData, Data, Batch


class MockHeteroData:
    g = HeteroData()
    g["old"].num_nodes = 1
    g["old"].y = torch.tensor([0])
    g["new"].num_nodes = 2
    g["new"].y = torch.tensor([0, 1])

    pred = {"new": torch.tensor([0.0, 1.0])}


class MockHeteroDataLarge:
    g = HeteroData()
    g["old"].num_nodes = 10
    g["old"].y = torch.zeros(g["old"].num_nodes)
    g["new"].num_nodes = 10
    g["new"].y = torch.arange(0, g["new"].num_nodes, dtype=torch.int)

    pred = {"new": torch.arange(0, g["new"].num_nodes, dtype=torch.float)}


class MockGraphA:

    g = Data()
    g.node_gt = torch.tensor([0, 0, 1])
    g.edge_gt = torch.tensor([[1, 0], [0, 1], [1, 1]])
    g.edge_index = torch.tensor([[0, 1, 2], [1, 2, 1]])
    g.node_preds = torch.tensor([[0.9, 0.1], [0.99, 0.01], [0.1, 0.9]])
    g.edge_preds = torch.tensor([[0.99, 0.1], [0.1, 0.99], [0.1, 0.99]])  # mistake in [2][0] leading to 0.75


class MockGraphB:

    g = Data()
    g.node_gt = torch.tensor([0, 1])
    g.edge_gt = torch.tensor([[1, 0], [0, 1]])
    g.edge_index = torch.tensor([[0, 1], [1, 0]])

    g.node_preds = torch.tensor([[0.7, 0.3], [0.3, 0.7]])
    g.edge_preds = torch.tensor([[0.001, 0.01], [0.1, 0.01]])  # mistake in [2][0] and [4][1] and [4][2] leading to 0.5 and 0.375


@pytest.fixture
def get_graph_batch_small():
    return Batch.from_data_list([MockGraphA.g])


@pytest.fixture
def get_graph_batch_large():
    return Batch.from_data_list([MockGraphA.g, MockGraphB.g])


@pytest.fixture
def get_g_with_unseen():
    g = MockHeteroData.g
    g["old", "to", "new"].edge_index = torch.tensor([[0], [0]])
    return g


@pytest.fixture
def get_g_without_unseen():
    g = MockHeteroData.g
    g["old", "to", "new"].edge_index = torch.tensor([[0, 0], [0, 1]])
    return g


@pytest.fixture
def get_g_unseen_complex():
    g = MockHeteroDataLarge.g
    g["old", "to", "old"].edge_index = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    g["new", "to", "new"].edge_index = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 0, 4, 5]])
    g["old", "to", "new"].edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 5, 6]])
    return g


@pytest.fixture
def get_g_with_wrong_pred():
    g = MockHeteroData.g
    g["old", "to", "new"].edge_index = torch.tensor([[0], [0]])

    g.pred = {"new": torch.tensor([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])}
    return g


def test_triple_recall_at_k(get_graph_batch_small):

    g = get_graph_batch_small
    recall = TripleRecallAtK(4)
    recall.update(g.node_preds, g.edge_preds, g.node_gt, g.edge_gt, g.edge_index, g.ptr)
    result = recall.compute().item()

    assert result == 0.75


def test_triple_recall_at_k_batch(get_graph_batch_large):

    g = get_graph_batch_large
    recall_micro = TripleRecallAtK(4, average="micro")
    recall_micro.update(g.node_preds, g.edge_preds, g.node_gt, g.edge_gt, g.edge_index, g.ptr)
    result_micro = recall_micro.compute().item()

    recall_macro = TripleRecallAtK(4, average="macro")
    recall_macro.update(g.node_preds, g.edge_preds, g.node_gt, g.edge_gt, g.edge_index, g.ptr)
    result_macro = recall_macro.compute().item()

    assert result_micro == 0.5
    assert result_macro == 0.375


def test_get_unseen(get_g_with_unseen):

    pred, target = IncrementalKSGN.get_unseen(MockHeteroData.pred, get_g_with_unseen)

    assert torch.equal(pred, torch.tensor([1.0]))
    assert torch.equal(target, torch.tensor([1]))


def test_get_unseen_none(get_g_without_unseen):

    pred, target = IncrementalKSGN.get_unseen(MockHeteroData.pred, get_g_without_unseen)

    assert pred is None
    assert target is None


def test_get_unseen_complex(get_g_unseen_complex):

    pred, target = IncrementalKSGN.get_unseen(MockHeteroDataLarge.pred, get_g_unseen_complex)

    assert torch.equal(pred, torch.tensor([4.0, 7.0, 8.0, 9.0]))
    assert torch.equal(target, torch.tensor([4, 7, 8, 9]))


def test_unseen_acc(get_g_with_wrong_pred):

    g = get_g_with_wrong_pred
    acc1 = torchmetrics.Accuracy("multiclass", num_classes=3, top_k=1)
    acc2 = torchmetrics.Accuracy("multiclass", num_classes=3, top_k=2)

    pred, target = IncrementalKSGN.get_unseen(g.pred, g)
    acc1.update(pred, target)
    acc2.update(pred, target)
    res1 = acc1.compute()
    res2 = acc2.compute()

    assert res1 == 0.0
    assert res2 == 1.0


def test_recall_scores(get_graph_batch_small):

    g = get_graph_batch_small

    target_scores = torch.tensor(
        [
            0.88209003,
            0.88209,
            0.88209,
            0.09801,
            0.09801,
            0.09801,
            0.089099996,
            0.089099996,
            0.089099996,
            0.009900001,
            0.009900001,
            0.0099,
            0.008909999,
            0.008909999,
            0.008909999,
            0.00099,
            0.00099,
            0.00099,
            0.00090000004,
            0.0008999999,
            0.0008999999,
            0.000100000005,
            0.000100000005,
            0.000100000005,
        ]
    )

    target_edges = torch.tensor(
        [
            [1, 2],
            [0, 1],
            [2, 1],
            [0, 1],
            [1, 2],
            [2, 1],
            [0, 1],
            [2, 1],
            [1, 2],
            [0, 1],
            [2, 1],
            [1, 2],
            [0, 1],
            [2, 1],
            [1, 2],
            [1, 2],
            [2, 1],
            [0, 1],
            [1, 2],
            [2, 1],
            [0, 1],
            [2, 1],
            [1, 2],
            [0, 1],
        ]
    )

    target_triples = torch.tensor(
        [
            [0, 1, 1],
            [0, 0, 0],
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 0],
            [1, 1, 1],
        ]
    )

    target_gt = torch.tensor([[0, 0, 0, 0, 1], [0, 1, 1, 1, 2], [1, 0, 0, 2, 1], [1, 1, 0, 2, 1]])
    s, e, t, gt = triple_recall_scores(g.node_preds, g.edge_preds, g.node_gt, g.edge_gt, g.edge_index)

    assert torch.all(torch.isclose(s, target_scores))
    assert torch.equal(e, target_edges)
    assert torch.equal(t, target_triples)
    assert torch.equal(gt, target_gt)


def test_recall_split_functions(get_graph_batch_small):

    g = get_graph_batch_small
    k = 4

    _, e, t, gt = triple_recall_scores(g.node_preds, g.edge_preds, g.node_gt, g.edge_gt, g.edge_index)
    hits, n = triple_recall_hits(e[:k], t[:k], gt[:k])

    assert hits / n == 0.75


def test_is_logits():

    a = torch.tensor([[-1, 5, 3, 2], [1, 2, 3, 4]], dtype=float)

    assert is_logits(a)
    assert not is_logits(a.softmax(-1))
    assert not is_logits(torch.nn.functional.sigmoid(a), multilabel=True)


def test_triple_recall_with_scores_small(get_graph_batch_small):

    data_list = get_graph_batch_small.to_data_list()
    recall_micro = TripleRecallWithScores(4, average="micro")
    recall_macro = TripleRecallWithScores(4, average="macro")

    for g in data_list:
        _, edges, triples, ground_truth = triple_recall_scores(g.node_preds, g.edge_preds, g.node_gt, g.edge_gt, g.edge_index)
        recall_micro.update(edges, triples, ground_truth)
        recall_macro.update(edges, triples, ground_truth)

    result_micro = recall_micro.compute().item()
    result_macro = recall_macro.compute().item()

    assert result_micro == 0.75
    assert result_macro == 0.75


def test_triple_recall_with_scores_large(get_graph_batch_large):

    data_list = get_graph_batch_large.to_data_list()
    recall_micro = TripleRecallWithScores(4, average="micro")
    recall_macro = TripleRecallWithScores(4, average="macro")

    for g in data_list:
        _, edges, triples, ground_truth = triple_recall_scores(g.node_preds, g.edge_preds, g.node_gt, g.edge_gt, g.edge_index)
        recall_micro.update(edges, triples, ground_truth)
        recall_macro.update(edges, triples, ground_truth)

    result_micro = recall_micro.compute().item()
    result_macro = recall_macro.compute().item()

    assert result_micro == 0.5
    assert result_macro == 0.375
