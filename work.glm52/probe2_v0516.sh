#!/usr/bin/env bash
# Probe v0.5.16 image WITH gpu devices (aiter needs rocminfo at import).
set -uo pipefail
IMAGE=lmsysorg/sglang:v0.5.16-rocm720-mi35x
SG=/sgl-workspace/sglang/python/sglang
docker run --rm --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --entrypoint bash "$IMAGE" -lc '
echo "===== versions ====="
python3 - <<"PY"
import importlib.metadata as md
for p in ("sglang","aiter","torch","triton","tilelang"):
    try: print(f"{p:10s} =", md.version(p))
    except Exception as e: print(f"{p:10s} = <{type(e).__name__}>")
import aiter, os
print("aiter.__version__ =", getattr(aiter,"__version__","?"))
from aiter.jit.utils.chip_info import get_gfx, get_cu_num
print("gfx =", get_gfx(), " cu_num =", get_cu_num())
PY
echo
echo "===== gfx950 mla_asm.csv : rows relevant to GLM-5.2 ====="
CSV=/sgl-workspace/aiter/hsa/gfx950/mla/mla_asm.csv
head -1 "$CSV"
echo "-- bf16 q / fp8 kv (what sglang feeds when --kv-cache-dtype fp8_e4m3):"
awk -F, "\$1==\"bf16\" && \$2==\"fp8\"" "$CSV"
echo "-- gqa 8 (native, TP8 unpadded) and gqa 16 (after sglang pad):"
awk -F, "\$3==8 || \$3==16" "$CSV" | cut -d, -f1-8
echo "-- total rows:"; wc -l < "$CSV"
echo
echo "===== is_experimental_enabled ====="
python3 -c "from aiter.jit.core import is_experimental_enabled as f; print(f())"
echo
echo "===== sglang: DSA choices + need_pad_heads + aiter dispatch ====="
python3 -c "from sglang.srt.server_args import DSA_CHOICES; print(DSA_CHOICES)"
grep -n "need_pad_heads" -A 5 '"$SG"'/srt/layers/attention/dsa_backend.py | head -14
echo "-- ROCm DSA default backend rule:"
grep -n "is_hip()" -B 3 -A 4 '"$SG"'/srt/arg_groups/overrides.py | grep -n "tilelang" -B 5 | head -20
echo
echo "===== PR30506 topk_v2 auto-disable sites ====="
sed -n "4510,4575p" '"$SG"'/srt/server_args.py
'
