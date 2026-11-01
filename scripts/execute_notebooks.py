"""Execute all project notebooks with the current Python interpreter in local Jupyter kernels."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
R=Path(__file__).resolve().parents[1]

def main():
    for path in sorted((R/'notebooks').glob('*.ipynb')):
        nb=nbformat.read(path,as_version=4)
        km=KernelManager(kernel_name='python3')
        km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        client=NotebookClient(nb,timeout=240,resources={'metadata':{'path':str(R)}},km=km)
        try:
            client.execute()
            nbformat.validate(nb)
            nbformat.write(nb,path)
            print('Executed and validated',path.name,flush=True)
        finally:
            if km.has_kernel:km.shutdown_kernel(now=True)

if __name__=='__main__':main()
