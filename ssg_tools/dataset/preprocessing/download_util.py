import tempfile
import urllib
import os
import logging
from pathlib import Path
import subprocess
import urllib.request

log = logging.getLogger(__name__)

def download_file(output_path: Path, url: str, overwrite: bool = False):
    output_path = Path(output_path).absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file() and not overwrite:
        log.info("Skipping download of existing file \"%s\"", output_path)
    else:
        #  download the file into a temporary file and rename it after it is complete
        fh, out_file_tmp = tempfile.mkstemp(dir=output_path.parent)
        f = os.fdopen(fh, 'w')
        f.close()
        urllib.request.urlretrieve(url, out_file_tmp) 
        Path(out_file_tmp).rename(output_path)
        #os.rename(out_file_tmp, output_path)
    

def unzip_file(file: Path, destination=None, extract_pattern: str | None = None, overwrite: bool = False) -> None:
    file = Path(file)
    destination = destination or file.parent
    # Unzip
    cmd = [
        "unzip", str(file), 
    ]
    if extract_pattern is not None:
        cmd.extend([extract_pattern, "-j"])
    
    if destination is not None:
        cmd.extend(["-d", str(destination)+ "/"])
    
    log.info("Unzip command: %s", " ".join(cmd))
    
    sp = subprocess.Popen(cmd, cwd=destination, stdout=subprocess.PIPE,
                          stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    if overwrite:
        # send something to skip input()
        sp.communicate('A'.encode())[0].rstrip()
    else:
        # send something to skip input()
        sp.communicate('N'.encode())[0].rstrip()

