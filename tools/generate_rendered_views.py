import os, sys

sys.path.append(os.path.dirname(__file__))
from ssg_tools.rendering.scan_renderer import ScanRenderer
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG, ScanInterface
import argparse
from pathlib import Path
import warnings
from tqdm.contrib.concurrent import process_map
import logging

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)

script_path = Path(__file__).parent

parser = argparse.ArgumentParser(description="Renders the ground truth of a 3dssg scan for each pose")
parser.add_argument("--dataset_path", required=True, help="The path to the dataset.")
parser.add_argument("--id", required=False, nargs=1, default=None, help="specific scan id to render. By default all scans will be rendered")
parser.add_argument("--workers", type=int, default=0, help="The number of parallel processes to be used.")
args = parser.parse_args()

if __name__ == "__main__":

    dataset_path = Path(args.dataset_path)

    dataset = DatasetInterface3DSSG(dataset_path)

    # process all scans if id is not given
    if args.id is None:
        args.id = dataset.scan_ids
    print(args.id)

    def process_scan(scan_id):
        scan = dataset.scan(scan_id)
        # create the output dir if missing
        outp = scan.scan_path / "sequence"
        outp.mkdir(parents=True, exist_ok=True)
        renderer = ScanRenderer(scan)

        for pose_index in range(scan.nimages):
            res = renderer.render(pose_index)
            res.save(outp)

    log.info("Rendering images...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if args.workers > 0:
            process_map(process_scan, args.id, max_workers=args.workers, chunksize=1)
        else:
            for id in args.id:
                process_scan(id)
    log.info("done.")
