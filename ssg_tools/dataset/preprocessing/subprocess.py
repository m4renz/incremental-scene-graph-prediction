import subprocess, os, sys
from pathlib import Path

def execute(cmd,cwd,):
    '''
    Executate something with realtime stdout catch
    '''
    shell = isinstance(cmd,str)
    popen = subprocess.Popen(cmd,cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                             shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        print(cmd)
        raise subprocess.CalledProcessError(return_code, cmd)

def run(bashCommand,cwd='./',verbose:bool=True):
    '''
    execute an exe with realtime stdout catch
    '''
    for path in execute(bashCommand,cwd):
        if verbose:
            print(path, end="")
        else:
            pass
        
def run_python(bashCommand,cwd="./", pythonpath:str="",verbose:bool=True):
    ''' 
    execute a python script with realtime stdout catch
    '''
    if len(pythonpath)>0: os.environ['PYTHONPATH'] = pythonpath
    bashCommand=[sys.executable]+bashCommand
    run(bashCommand,cwd,verbose)


def download_unzip(url: str, output_dir: Path, unzip_args: list[str] | None = None, overwrite: bool = False) -> None:
    filename = os.path.basename(url)
    output_dir = Path(output_dir)
    if not output_dir / filename or overwrite:
        cmd = [
            "wget", url
        ]
        run(cmd, output_dir)

    # Unzip
    cmd = [
        "unzip", filename, 
    ]

    if unzip_args is not None:
        cmd.extend(unzip_args)

    sp = subprocess.Popen(cmd, cwd=output_dir, stdout=subprocess.PIPE,
                          stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    if overwrite:
        # send something to skip input()
        sp.communicate('A'.encode())[0].rstrip()
    else:
        # send something to skip input()
        sp.communicate('N'.encode())[0].rstrip()
    sp.stdin.close()  # close so that it will proceed