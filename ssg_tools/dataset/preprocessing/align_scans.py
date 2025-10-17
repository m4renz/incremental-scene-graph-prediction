from pathlib import Path
import numpy as np
import json
from shutil import copyfile
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map 
from functools import partial
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG, ScanInterface
import logging

log = logging.getLogger(__name__)

__all__ = ["align_scans"]


def read_transform_matrices(dataset_path: Path) -> dict[str, np.ndarray]:
    rescan2ref = {}
    path = Path(dataset_path) / "3RScan.json"
    with open(path, "r") as f:
        data = json.load(f)
        for scene in data:
            for scans in scene["scans"]:
                if "transform" in scans:
                    rescan2ref[scans["reference"]] = np.asarray(scans["transform"], dtype=np.float32).reshape(4,4)
    return rescan2ref

def process(scan: ScanInterface, 
            *,
            overwrite: bool = False) -> None:
    """Process a single scan
    """
    filename_in = scan.label_mesh_filename(aligned=False)
    filename_out = scan.label_mesh_filename(aligned=True)

    if not filename_in.is_file():
        log.debug("Skipping file %s. File does not exist", filename_in)
        return # file does not exist in this scan  
    
    if filename_out.is_file() and not overwrite:
        log.debug("Skipping processed file %s.", filename_out)
        return # skip if file is processed already
    
    transform_matrix, valid = scan.transform_matrix
    if valid:
        plyfile = scan.label_mesh(aligned=False)
        plyfile.transform(transform_matrix)
        plyfile.write(filename_out, ascii=None)
    else:
        copyfile(filename_in, filename_out)

def align_scans(dataset: DatasetInterface3DSSG,
                nworkers: int = 0,
                overwrite: bool = False) -> None:
    
    process_func = partial(process, overwrite=overwrite)
    
    if nworkers < 0 or nworkers > 0:
        process_map(process_func, dataset.scans(), max_workers=nworkers, chunksize=1)
    else:
        pbar = tqdm(dataset.scans())
        for scan in pbar:
            process_func(scan)