from __future__ import annotations
from collections import defaultdict
import numpy as np
from typing import Sequence, Any, Literal, Optional

from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG


def _compute_weights(labels: Sequence[Any],
                     occurrences: Sequence[int],
                     normalize: bool = True,
                     bce: bool = False,
                     num_samples: Optional[int] = None) -> np.ndarray:
    """Compute the weights of the given labels with occurrences.

    Args:
        labels: The labels to compute the weight for
        occurrences: The occurrence count of each label in labels
        normalize: Normalize the weights. Defaults to True.
        bce: Compute the weights for binary cross entroy. Defaults to False.
        num_samples: Number of total samples to consider for bce. Defaults to None. Required if bce=True.

    Returns:
        Weights for each label
    """
    occurrences = np.asarray(occurrences)
    
    if not bce:
        factor = occurrences.sum()
    else:
        factor =  np.subtract(num_samples, occurrences, where=occurrences > 0)

    weights = np.divide(factor, occurrences, where=occurrences > 0)

    if normalize:
        weights /= weights.sum()
        weights *= len(labels)
    
    # set the uninitialized weights to the minimal value of the initialized ones
    weights[weights == 0] = weights[weights >0].min()

    return weights.astype(np.float32)


def compute_weights(dataset: DatasetInterface3DSSG, 
                    edge_mode: Literal["gt", "fully_connected", "nn"] = 'gt',
                    normalize: bool = True,
                    use_bce: bool = False):
    """Compute the weights for the node and edge classes of the given dataset

    Args:
        dataset: The dataset to compute the weights for
        fully_connected: Use the fully connected edges. Defaults to False.
        normalize: Normalize the weights. Defaults to True.
        use_bce: Compute the edge class weights for binary cross entropy. Defaults to False.
    """
    occurrences_node_classes = np.zeros(len(dataset.node_classes()), dtype=np.int64)
    occurrences_edge_relationships = np.zeros(len(dataset.edge_classes()), dtype=np.int64)

    #scene_stats = {}
    n_edge_with_gt = 0
    n_edges_fully_connected = 0


    for scan in dataset.scans():
        scene_graph = scan.scene_graph(raw=False)
        # number of objects in the scene
        n_obj = scene_graph.number_of_nodes()

        # count how often each time the label appears in the scene
        node_labels = np.array(scene_graph.nodes.data("rio27_enc"))[:, 1]
        # TODO: is the extra weight for label 0 required?
        occurrences_node_classes += np.bincount(node_labels, minlength=occurrences_node_classes.shape[0])

        # number of fully connected edges in the scene.
        n_edges_fully_connected += n_obj * n_obj - n_obj
        #n_rel = scene_graph.number_of_edges()

        #nnk = defaultdict(int)
        edge_labels = np.array(scene_graph.edges(keys=True))
        # TODO: is the extra weight for label 0 required?
        occurrences_edge_relationships += np.bincount(edge_labels[:, 2], minlength=occurrences_edge_relationships.shape[0])

        uniques_edges = np.unique(edge_labels[:, :2], axis=1)
        #for subject, object, relation in scene_graph.edges(keys=True):
        #    occurrences_edge_relationships[relation - 1] += 1
        #    nnk[f"{object}_{subject}"] += 1
        n_edge_with_gt += uniques_edges.shape[0]
        #scene_stats[scan.scan_id] = scan_stats = {}
        #scan_stats["num_objects"] = n_obj
        #scan_stats["num_relationsships"] = n_rel

    node_class_weights = _compute_weights(dataset.node_classes(), occurrences_node_classes, normalize=normalize)

    if use_bce:
        if edge_mode == 'gt':
            total_n_edges = n_edge_with_gt
        elif edge_mode == 'fully_connected':
            total_n_edges = n_edges_fully_connected
        elif edge_mode == 'nn':
            # number of edges in the neighbors graph. Each edge is counted only once
            total_n_edges = sum(scan.neighbors_graph().number_of_edges() for scan in dataset.scans())
        else:
            raise ValueError(f"Invalid edge mode {edge_mode}.")

        edge_class_weights = _compute_weights(dataset.edge_classes(), occurrences_edge_relationships, normalize=normalize, bce=True, num_samples=total_n_edges)
    else:
        edge_class_weights = _compute_weights(dataset.edge_classes(), occurrences_edge_relationships, normalize=normalize)

    return node_class_weights, edge_class_weights, occurrences_node_classes, occurrences_edge_relationships