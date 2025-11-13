from pathlib import Path
import subprocess
from tqdm_loggable.auto import tqdm
import logging
from ssg_tools.dataset.preprocessing.subprocess import run

log = logging.getLogger(__name__)

__all__ = ["download_3rscan_data"]


def download_3rscan_data(
    download_script: Path, output_path: Path, nworkers: int = 0, sequences: bool = False, overwrite: bool = True, id: str | None = None
) -> None:

    def run_download(cmd, cwd):
        sp = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
        # send something to skip input()
        sp.communicate("1".encode())[0].rstrip()
        sp.stdin.close()  # close so that it will proceed

    download_script = Path(download_script).absolute()
    output_path = Path(output_path).absolute()

    # create the output path if it doesnt exist
    output_path.mkdir(parents=True, exist_ok=True)
    # check if download.py exist
    assert (
        download_script.is_file()
    ), f"The 3RScan download script 'download.py' must be present at {download_script} in order to download the 3RScan dataset.\
        Please fill the term of use in this page: https://waldjohannau.github.io/RIO/ to obtain the download script."

    types = ["semseg.v2.json", "labels.instances.annotated.v2.ply", "mesh.refined.v2.obj", "mesh.refined.mtl", "mesh.refined_0.png"]
    if sequences:
        types.append("sequence.zip")

    log.debug("Downloading 3rscan dataset...")
    pbar = tqdm(types)
    for type in pbar:
        pbar.set_description(f"downloading file type {type}...")
        cmd = ["python", download_script, "-o", output_path, "--type", type]
        if id:
            cmd.append("--id")
            cmd.append(id)
        run_download(cmd, "./")

    if sequences:
        nworkers = 1 if nworkers == 0 else nworkers
        log.info("Extracting sequence data...")
        # unzip all sequences
        cmd = (
            r"""find . -name 'sequence.zip' -print0 | xargs -0 -I {{}} -P {nworkers} sh -c '"""
            r"""base="$1"; """
            r"""filename="${{base%.*}}"; """
            r"""unzip -{unzip_flag} -d "$filename" "$1"' sh {{}}"""
        ).format(nworkers=nworkers, unzip_flag=("o" if overwrite else "n"))
        # cmd = r"""find . -name 'sequence.zip' -exec sh -c 'base={};filename="${base%.*}"; unzip -o -d $filename {};' ';'   """
        run(cmd, output_path)
    log.info("done.")
