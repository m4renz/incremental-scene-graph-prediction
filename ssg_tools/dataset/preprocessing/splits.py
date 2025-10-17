from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG
import logging

log = logging.getLogger(__name__)

def gen_splits(dataset: DatasetInterface3DSSG, train_validate_percent: float = 0.8) -> tuple[list[str], list[str], list[str]]:    
    path_3rscan_json = dataset.path / "3RScan.json"
    with open(path_3rscan_json,'r') as f:
        data = json.load(f)
    
    # get all valid scan ids
    scan_ids = set(dataset.scan_ids)

    train_scans = []
    test_scans  = []

    for scan in data:
        ref_id = scan['reference']
        type = scan['type']
        if type == 'train':
            l = train_scans
        elif type in ('validation', 'test'):
            l = test_scans
        else:
            log.warning("Invalid scan type '%s' found in dataset.", type)
            continue

        if ref_id in scan_ids:
            l.append(ref_id)

        for sscan in scan['scans']:
            scan_id = sscan["reference"]
            if scan_id in scan_ids:
                l.append(scan_id)
    
    ntrain = int(max(1, math.floor(train_validate_percent * len(train_scans))))
    
    sample_train = np.random.choice(range(len(train_scans)),ntrain, replace=False).tolist()
    sample_valid = set(range(len(train_scans))).difference(sample_train)
    assert len(sample_train) + len(sample_valid) == len(train_scans)
    
    sample_train = [train_scans[i] for i in sample_train]
    sample_valid = [train_scans[i] for i in sample_valid]
    
    log.info('train: %d, validation: %d, test: %d', len(sample_train), len(sample_valid), len(test_scans))
            
    return sample_train, sample_valid, test_scans


def save(path: Path, scans: list[str]) -> None:
    with open(path,'w') as f:
        json.dump(scans, f, indent=4)   

def splits(dataset: DatasetInterface3DSSG, train_validate_percentage: float = 0.8, overwrite: bool = False):
    # from ssg_tools import ConfigArgumentParser, init_logging
    # #from ssg_tools.util import set_random_seed

    # parser = ConfigArgumentParser(
    #     description='Generate the train, validate, test splits from the raw 3dssg dataset.')
    # parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite existing files.')
    # parser.add_argument('-p', '--percentage', type=float, default=0.9, help='split percentage between train and validation')

    # args = parser.parse_args()

    # config = args.config
    # init_logging(config)

    train_path = dataset.filename_split("train")
    validate_path = dataset.filename_split("validate")
    test_path = dataset.filename_split("test")
    # set the random seed here
    #set_random_seed(config.get("random_seed", None))

    train_scans, validation_scans, test_scans = gen_splits(dataset, train_validate_percentage)


    if train_path.exists() and not overwrite:
        log.info("%s exists. Skipping...", train_path)
    save(train_path, train_scans)
    if validate_path.exists() and not overwrite:
        log.info("%s exists. Skipping...", validate_path)
    save(validate_path, validation_scans)
    if test_path.exists() and not overwrite:
        log.info("%s exists. Skipping...", test_path)
    save(test_path, test_scans)
    log.info("done.")

# if __name__ == '__main__':
#     main()