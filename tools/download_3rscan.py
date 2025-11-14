from ssg_tools.dataset.preprocessing.download_3rscan import download_3rscan_data
from pathlib import Path
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the 3RScan dataset")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset root where 3RScan will be stored.")
    parser.add_argument("--download_script", required=True, help="Path to 3RScan download.py obtained from RIO/3RScan.")
    parser.add_argument("--id", required=False, help="Optional single scan UUID to download.")
    parser.add_argument("--sequences", action="store_true", help="Download sequences.zip for each scan and extract.")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers for sequence extraction.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted sequences when present.")
    args = parser.parse_args()

    download_3rscan_data(
        download_script=Path(args.download_script),
        output_path=Path(args.dataset_path) / "3RScan",
        nworkers=args.workers,
        sequences=args.sequences,
        overwrite=args.overwrite,
        id=args.id,
    )
