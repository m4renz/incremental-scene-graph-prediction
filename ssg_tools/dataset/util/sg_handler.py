import json
from pathlib import Path

import networkx as nx
import numpy as np


class SGHandler:

    def __init__(self, path: str | Path, scan: str) -> None:

        self.path = path
        self.scan = scan

        self.scene = None
        self.node_classes = None
        self.edge_classes = None

        self.num_edge_classes = None
        self.num_node_classes = None

        self.sg = nx.MultiDiGraph()
        self.neighbor_graph = nx.Graph()
        self.load_sg()

    def load_sg(self):

        with open(self.path, "r") as f:
            sg_json = json.load(f)

        self.scene = sg_json["scenes"][self.scan]
        self.node_classes = sg_json["node_classes"]
        self.edge_classes = sg_json["edge_classes"]
        # self.edge_classes.append('none')

        self.num_node_classes = len(self.node_classes)
        self.num_edge_classes = len(self.edge_classes)

        for node, attr in self.scene["nodes"].items():
            self.sg.add_node(int(node), **attr)
            self.neighbor_graph.add_node(int(node))
            if "neighbors" in attr and attr["neighbors"]:
                adj = np.stack([np.full_like(attr["neighbors"], fill_value=int(node)), attr["neighbors"]], dtype=np.int64)
                self.neighbor_graph.add_edges_from(adj.T)

        for edge in self.scene["edges"]:
            self.sg.add_edge(edge[0], edge[1], label_idx=edge[2], label=edge[3])

    def print_details(self):

        print(f"Scene: {self.scan}")
        print(f"Nodes: {len(self.sg.nodes)}")
        print(f"Edges: {len(self.sg.edges)}")

        print(f"Node Classes: {set(nx.get_node_attributes(self.sg, 'rio27_name').values())}")
        print(f"Edge Classes: {set(nx.get_edge_attributes(self.sg, 'label').values())}")

    def __str__(self):
        return f"SceneGraph with {len(self.sg.nodes)} nodes and {len(self.sg.edges)} edges"


def get_rio27_list(scenes_path: Path | str, remove_dash: bool = True):
    """
    Get the list of classes from the scenegraph.json file
    """

    with open(scenes_path) as f:
        _scenes = json.load(f)

    classes = _scenes["node_classes"]
    if remove_dash:
        classes.remove("-")
    return classes
