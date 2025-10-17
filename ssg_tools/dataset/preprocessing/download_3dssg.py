from pathlib import Path

import logging
from ssg_tools.dataset.dataset_interface import DatasetInterface3DSSG
from ssg_tools.dataset.preprocessing.download_util import download_file, unzip_file

log = logging.getLogger(__name__)

__all__ = ["download_3dssg"]

_urls = ["http://campar.in.tum.de/public_datasets/3DSSG/3DSSG.zip", "https://www.campar.in.tum.de/public_datasets/3RScan/3RScan.json"]

_url_rendered_views = "https://www.campar.in.tum.de/public_datasets/2023_cvpr_wusc/rendered.zip"


def download_3dssg(
    dataset_interface: DatasetInterface3DSSG,
    overwrite: bool = False,
    rendered_views: bool = False,
    remove_zips: bool = True,
) -> None:

    dataset_path = dataset_interface.path
    # download all files specified in the configuration
    for url in _urls:
        filename = Path(url).name
        filepath = dataset_path / filename
        log.info("Downloading %s from %s...", filename, url)
        download_file(filepath, url, overwrite=overwrite)
        if Path(url).suffix == ".zip":
            log.info("Extracting %s", filepath)
            unzip_file(filepath, extract_pattern="3DSSG/*", overwrite=overwrite)
            if remove_zips:
                filepath.unlink()

    if rendered_views:
        log.info("Downloading rendered views...")
        rendered_views_path = dataset_path / "rendered.zip"
        download_file(
            rendered_views_path,
            _url_rendered_views,
            overwrite=overwrite,
        )
        log.info("Extracting rendered views")
        unzip_file(
            rendered_views_path,
            overwrite=overwrite,
        )
        if remove_zips:
            rendered_views_path.unlink()


# CLI for downloads is provided in tools/download_3dssg.py
