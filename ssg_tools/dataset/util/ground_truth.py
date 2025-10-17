import torch
import torch.nn.functional as F
import torch_geometric
from typing import Iterable

def get_onehot_gt(class_id: torch.Tensor | Iterable, num_classes: int):
    """
    Converts class IDs to one-hot encoded ground truth.

    Args:
        class_id (torch.Tensor | Iterable): The class IDs to be converted.
        num_classes (int): The total number of classes.

    Returns:
        torch.Tensor: The one-hot encoded ground truth tensor.
    """
    return F.one_hot(torch.tensor(class_id), num_classes=num_classes).float()

def get_neighbor_classes(edge_index: torch.Tensor, y: torch.Tensor):
    """
    Computes the neighbor classes for each node in a graph.

    Args:
        edge_index (torch.Tensor): The edge index tensor representing the graph structure.
        y (torch.Tensor): The label tensor for each node.

    Returns:
        torch.Tensor: The tensor containing the neighbor classes for each node.
    """
    num_nodes = y.size(0)
    num_edges = edge_index.size(1)

    # Initialize the output tensor with zeros
    output = torch.zeros((num_nodes, y.size(1)), dtype=torch.long)

    # Iterate over each edge and update the output tensor
    for i in range(num_edges):
        source, target = edge_index[0, i].item(), edge_index[1, i].item()
        output[source] = torch.max(output[source], y[target])

    return output

def get_class_graph_edge_index(edge_index: torch.Tensor, y: torch.Tensor):
    """
    Returns the edge index of the class graph based on the given edge index and class labels.

    Args:
        edge_index (torch.Tensor): The edge index of the graph.
        y (torch.Tensor): The class labels of the nodes, either one-hot encoded or index based.

    Returns:
        torch.Tensor: The edge index of the class graph.
    """

    if y.dim() > 1:
        y = y.argmax(dim=1)

    return torch.stack([edge_index[0], y[edge_index[1]]], dim=0).t().unique(dim=0).t()

def create_class_edge_index(y):
    """
    Create edge index for to class label graph.

    Args:
        y (torch.Tensor): Tensor containing node's class labels.

    Returns:
        torch.Tensor: Edge index tensor.
    """
    return torch.stack(torch.where(y > 0))

def sample_negative_class_edges(class_edge_index : torch.Tensor, num_samples: int | float = 1 , num_nodes: int | tuple | list | None = None):
    """
    Sample negative class edges using negative sampling.

    Args:
        class_edge_index (torch.Tensor): The edge indices of the class.
        num_samples (int | float, optional): The number of negative samples to generate. Defaults to 1.
        num_nodes (int | tuple | list | None, optional): The number of nodes in the graph. Defaults to None.

    Returns:
        torch.Tensor: The negative class edges.

    """
    return torch_geometric.utils.negative_sampling(class_edge_index, num_nodes=num_nodes, num_neg_samples=class_edge_index.size(1)*num_samples)

def create_linkpred_gt(data, num_neg_samples=2):
    """
    Create ground truth for link prediction task.

    Args:
        data (torch_geometric.data.Data): Input graph data.
        num_neg_samples (int, optional): Number of negative samples per positive sample to generate. Defaults to 2.

    Returns:
        tuple: A tuple containing the edge label index and the corresponding edge labels.
    """
    class_edge_index = create_class_edge_index(data.y)
    neg_samples = sample_negative_class_edges(class_edge_index, num_samples=num_neg_samples, num_nodes=(data.num_nodes, data.y.size(1)))
    edge_label_index = torch.cat([class_edge_index, neg_samples], dim=-1)
    edge_label = torch.zeros(edge_label_index.size(1))
    edge_label[:class_edge_index.size(1)] = 1
    return edge_label_index, edge_label