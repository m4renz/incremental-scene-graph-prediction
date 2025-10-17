from typing import Sequence, Any
import numpy as np
import networkx as nx
import itertools
import random


def make_complete(graph: nx.Graph):
    new_graph = nx.empty_graph(default=graph.__class__)
    # copy nodes and data from the original graph
    new_graph.add_nodes_from(graph.nodes(data=True))
    if len(graph.nodes()) > 1:
        # if there is more than one node in the graph add a fully connected set of edges
        if new_graph.is_directed():
            edges = itertools.permutations(new_graph.nodes(), 2)
        else:
            edges = itertools.combinations(new_graph.nodes(), 2)
        new_graph.add_edges_from(edges)
    return new_graph


def sample_neighbors(scene_graph: nx.MultiDiGraph, selected_nodes: Sequence[int], nhops:int, nsample_initial: int = 1) -> set[int]:
    """Selects nsample nodes from selected_nodes and captures all neighbors over nhops hops in the scene graph

    Args:
        scene_graph_nodes: the nodes of the scene graph
        selected_nodes: the nodes to consider for sampling
        nhops: the number of hops to include the neighbors
        nsample: The number of nodes to sample from selected_nodes. Defaults to 1.

    Returns:
        set containing the nodes sampled at every hop
    """

    if nsample_initial > len(selected_nodes): nsample_initial = len(selected_nodes)

    # sample nsample unique values from the selected node ids
    sampled_nodes = np.random.choice(np.unique(selected_nodes), nsample_initial, replace=False).tolist()
    
    filtered_nodes = set(sampled_nodes)
    #filtered_nodes = filtered_nodes.union(sampled_nodes)

    n_seletected_nodes = dict() # this save the neighbor with level n. level n+1 is the neighbors from level n.
    n_seletected_nodes[0] = sampled_nodes # first layer is the selected node.

    # loop over the specified number of hops
    for n in range(nhops):
        # collect neighbors
        unique_nn = set()
        for node_id in n_seletected_nodes[n]:
            found = set(scene_graph.nodes[node_id]['neighbors'])
            found = found.intersection(selected_nodes) # only choose the node within our selections
            found = found.difference([0])# ignore 0
            if len(found) == 0: 
                continue

            unique_nn = unique_nn.union(found)
            filtered_nodes = filtered_nodes.union(found)
        n_seletected_nodes[n+1] = unique_nn
    
    return filtered_nodes


def sample_edges_from_nodes(graph: nx.MultiDiGraph, max_edges_per_node: int):
    '''
    flow: an edge passes message from i to j is denoted as  [i,j]. 
    '''

    keep_edges = set()
    graph = graph.copy()
    # remove all self-references just in case
    graph.remove_edges_from(nx.selfloop_edges(graph))

    for node in graph.nodes():
        edges = list(graph.edges(node))
        #neighbors = set(scene_graph.nodes("neighbors")[instance_id])
        #neighbors = neighbors.intersection(selected_nodes) # only the nodes within node_ids are what we want
        #if instance_id in neighbors: neighbors.remove(instance_id) # remove itself

        if max_edges_per_node > 0:
            if len(edges) > max_edges_per_node:
                edges = random.sample(edges, max_edges_per_node)
                #edges = list(np.random.choice(edges, max_edges_per_node))
        keep_edges = keep_edges.union(edges)
    graph = graph.edge_subgraph(keep_edges).copy()
        #edge_indices.extend([(instance_id, neighbor) for neighbor in neighbors])
    return graph


def visualize_graph(G):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, 
                           cmap=plt.get_cmap('jet'),
                           node_size = 500, ax=ax)
    nx.draw_networkx_labels(G, pos, {node: label for node, label in G.nodes(data="rio27_name")}, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(s, o): d for s,o,d in G.edges(data="name")}, ax=ax)
    fig.show()