# Put this directory first on PYTHONPATH to load the KernelForge overlay (infera_kernels) in every python process.
# This file shadows the distribution's sitecustomize, so that one is executed first.
import os
import sys

_sys_sc = "/usr/lib/python3.10/sitecustomize.py"
if os.path.isfile(_sys_sc):
    try:
        exec(compile(open(_sys_sc).read(), _sys_sc, "exec"), {"__name__": "_dist_sitecustomize"})
    except Exception as e:
        print(f"[infera-kernels] distribution sitecustomize failed: {e}", file=sys.stderr)

if any(os.environ.get(f"INFERA_KERNEL_{k}") == "1" for k in ("SPARSE", "INDEXER", "MOE", "DCP_SPARSE", "DCP_INDEXER")):
    import infera_kernels
    infera_kernels.install()
