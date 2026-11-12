# Put this directory first on PYTHONPATH to load the KernelForge overlay (yihou_kf) in every python process.
# This file shadows the distribution's sitecustomize, so that one is executed first.
import os
import sys

_sys_sc = "/usr/lib/python3.10/sitecustomize.py"
if os.path.isfile(_sys_sc):
    try:
        exec(compile(open(_sys_sc).read(), _sys_sc, "exec"), {"__name__": "_dist_sitecustomize"})
    except Exception as e:
        print(f"[yihou-kf] distribution sitecustomize failed: {e}", file=sys.stderr)

if any(os.environ.get(f"YIHOU_KF_{k}") == "1" for k in ("SPARSE", "INDEXER", "MOE", "DCP_SPARSE", "DCP_INDEXER", "FUSED_INDEXER")) or os.environ.get("YIHOU_KF_DUMP_DCP_IDX"):
    import yihou_kf
    yihou_kf.install()

if os.environ.get("KOPT_MEGATP") == "1":
    import kopt_megatp
    kopt_megatp.install()

if os.environ.get("KOPT_GEMM_MERGE") == "1":
    sys.path.insert(0, "/kopt_overlay_extra")
    import kopt_gemm_merge
    kopt_gemm_merge.install()
