from ssg_tools.dataset.preprocessing.download_3dssg import download_3dssg
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the 3dssg dataset")
    parser.add_argument("--dataset_path", required=True, help="The path to the dataset.")
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing files.")
    parser.add_argument("--download_rendered_views", action="store_true", help="download rendered views.")
    args = parser.parse_args()

    dataset = DatasetInterface3DSSG(args.dataset_path)
    download_3dssg(
        dataset,
        rendered_views=args.download_rendered_views,
        overwrite=args.overwrite,
    )
