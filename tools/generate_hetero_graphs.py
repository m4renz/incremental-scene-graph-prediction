from ssg_tools.dataset.hetero_dataset import HeteroSceneGraphDataset
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the hetero graphs")
    parser.add_argument("--dataset_path", required=True, help="The path to the dataset.")
    parser.add_argument("--id", help="The id of the scan to generate the hetero graphs for.")
    parser.add_argument("--num_feature_points", type=int, default=256, help="The number of feature points to use.")
    parser.add_argument("--reprocess", action="store_true", help="Whether to reprocess the data.")
    parser.add_argument("--num_workers", type=int, default=1, help="The number of workers to use.")
    parser.add_argument("--num_processes", type=int, default=1, help="The number of processes to use.")
    args = parser.parse_args()

    dataset = HeteroSceneGraphDataset(
        root=args.dataset_path,
        num_feature_points=args.num_feature_points,
        scans=[args.id] if args.id else None,
        reprocess=args.reprocess,
        num_workers=args.num_workers,
        num_processes=args.num_processes,
    )  # torch_geometric datasets process automatically when initialized
