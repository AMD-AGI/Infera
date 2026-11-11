#!/usr/bin/env python3
"""Tapered GEMM2 column groups for MegaMoeTP's large-batch schedule (in place, inside a throw-away container).
GEMM2 units are claimed column group by column group, and a group's reduce-scatter starts only when all its units are
done, so the last groups' units plus their reduction are the kernel's tail (per-CTA timestamps 11-11: with q 2 the CTAs
finish GEMM 13-18 us apart and the roles end 16-30 us after the median CTA). aiter tapers the widths for one shape
only (H 3072, q 3: [3, 3, 3, 2, 1]); this lets AITER_MEGAMOE_TP_LB_CGW="w1,w2,..." set them (sum = H / 256 chunks,
each width a legal q), e.g. "2,2,2,2,2,2,2,2,2,2,1,1,1,1". The kernel name gets the "_cg..." suffix as in aiter's case.

Usage (in the container): python3 megatp_cgw_patch.py
"""
import importlib.util
import os

path = os.path.dirname(importlib.util.find_spec("aiter").origin) + "/ops/flydsl/kernels/mega_moe_tp/mega_moe_tp_config.py"
s = open(path).read()
if "AITER_MEGAMOE_TP_LB_CGW" in s:
    print("already patched")
else:
    old = "    if _cg:\n        CGW = [3, 3, 3, 2, 1]\n        NCG = len(CGW)\n"
    new = old + (
        "    _cgw_env = __import__(\"os\").environ.get(\"AITER_MEGAMOE_TP_LB_CGW\", \"\")\n"
        "    if lb and _cgw_env:\n"
        "        _w = [int(v) for v in _cgw_env.split(\",\")]\n"
        "        assert sum(_w) == NCK, f\"AITER_MEGAMOE_TP_LB_CGW must sum to {NCK}\"\n"
        "        assert all(w * GPC % gemm2_group_step(H, I) == 0 for w in _w), \"illegal width\"\n"
        "        CGW, NCG, _cg = _w, len(_w), True\n"
    )
    assert s.count(old) == 1, "anchor not found once"
    open(path, "w").write(s.replace(old, new))
    print("patched", path)
