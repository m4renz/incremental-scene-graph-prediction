import networkx
from ssg_tools.loader.expectation_loader import ExpectationLoader
import torch
import matplotlib.pyplot as plt

def quick_plot(x, edge_index, y=None, layout_fct=networkx.spring_layout, directed=False):
    """
    Plots a graph using the given node features, edge indices, and optional node labels and layout.

    Parameters:
        x (Tensor): Node features.
        edge_index (Tensor): Edge indices.
        y (Tensor, optional): Node labels. Defaults to None.
        layout (function, optional): Node layout function. Defaults to networkx.spring_layout.
        directed (bool, optional): Whether the graph is directed. Defaults to False.
    """
    
    G = networkx.Graph()
    if directed:
        G = networkx.DiGraph()
    G.add_nodes_from(range(len(x)))
    G.add_edges_from(edge_index.T.numpy())

    layout = layout_fct(G)

    if y is not None:
        if len(y.shape) > 1:
            y = y.argmax(dim=1)
        networkx.draw(G, with_labels=True, pos=layout, node_color=y, cmap='coolwarm')
    else:
        networkx.draw(G, with_labels=True, pos=layout)



def plot_dataset_metrics(data: ExpectationLoader):
    """
    Plots and prints various metrics of the dataset.

    Args:
        data (ThreeDSSG | ThreeDSSGSubset): The dataset to analyze.

    Returns:
        None
    """
    def get_dataset_numbers(data):
        """
        Calculates the number of nodes, edges, and ground truth nodes for each graph in the dataset.

        Args:
            data (ThreeDSSG or ThreeDSSGSubset): The input dataset.

        Returns:
            tuple: A tuple containing the lists of number of nodes, number of edges, number of ground truth nodes,
                and the total number of ground truth nodes in the dataset.
        """
        n_nodes = []
        n_edges = []

        for sg in data:
            n = sg.x.size(0)
            e = sg.edge_index.size(1)
            

            n_nodes.append(n)
            n_edges.append(e)

        return n_nodes, n_edges

    n_nodes, n_edges = get_dataset_numbers(data)

    if data.y.dim() > 1:
        y = torch.unique(data.y.argmax(dim=1), return_counts=True)[1]
    else:
        y = torch.unique(data.y, return_counts=True)[1]

    avg_nodes = sum(n_nodes) / len(n_nodes)
    avg_edges = sum(n_edges) / len(n_edges)

    print(f'Average number of nodes: {avg_nodes:.2f}')
    print(f'Average number of edges: {avg_edges:.2f}')
    print(f'Min/Max number of nodes: {min(n_nodes)}/{max(n_nodes)}')
    print(f'Min/Max number of edges: {min(n_edges)}/{max(n_edges)}')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.bar(range(len(y)), y)
    plt.title('Ground Truth Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.subplot(1, 3, 2)
    plt.hist(n_nodes)
    plt.title('Node Distribution')
    plt.xlabel('Number of Nodes')

    plt.subplot(1, 3, 3)
    plt.hist(n_edges, bins=20)
    plt.title('Edge Distribution')
    plt.xlabel('Number of Edges')
    plt.show()