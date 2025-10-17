from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG
from ssg_tools.dataset.preprocessing.splits import splits
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the splits for the 3dssg dataset")
    parser.add_argument("--dataset_path", required=True, help="The path to the dataset.")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing files.")
    parser.add_argument(
        "--train_validate_percentage", type=float, default=0.8, help="The percentage of the dataset to use for training and validation."
    )
    args = parser.parse_args()

    dataset = DatasetInterface3DSSG(args.dataset_path)
    splits(dataset, train_validate_percentage=args.train_validate_percentage, overwrite=args.overwrite)
