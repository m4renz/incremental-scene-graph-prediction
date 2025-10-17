from ssg_tools.dataset.preprocessing.scene_graph_generation import scene_graph_remapping
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG
import argparse

# recreate the interface to parse the downloaded scans

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate the sampled points from the original mesh data")
    parser.add_argument("--dataset_path", required=True, help="The path to the dataset.")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing files.")
    args = parser.parse_args()

    dataset = DatasetInterface3DSSG(args.dataset_path)
    scene_graph_remapping(dataset, overwrite=args.overwrite)
