from ssg_tools.dataset.preprocessing.features import save_class_embeddings, get_numberbatch_embeddings, get_clip_text_features
from ssg_tools.dataset.util.sg_handler import get_rio27_list
from pathlib import Path
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the embeddings")
    parser.add_argument("--dataset_path", required=True, help="The path to the dataset.")
    parser.add_argument("--numberbatch_path", help="The path to the numberbatch file.", default="/tmp")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    hetero_scene_graph_path = dataset_path / "hetero_scene_graph"

    if not hetero_scene_graph_path.exists():
        hetero_scene_graph_path.mkdir(parents=True)

    classes = get_rio27_list(dataset_path / "scenegraph.json")
    numberbatch = get_numberbatch_embeddings(classes, args.numberbatch_path)
    save_class_embeddings(numberbatch, hetero_scene_graph_path / "rio27_numberbatch_embeddings.npy")
    clip = get_clip_text_features(classes)
    np.save(hetero_scene_graph_path / "rio27_clip_embeddings.npy", clip, allow_pickle=False)
