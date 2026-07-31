#!/usr/bin/env bash
# Zero-GPU probe of the v0.5.16 image: what aiter version, which DSA kernels exist,
# and would GLM-5.2's shape (bf16 q / fp8 kv / gqa16-after-pad / persistent) find a kernel.
set -uo pipefail
IMAGE=lmsysorg/sglang:v0.5.16-rocm720-mi35x
docker run --rm --network host --entrypoint bash "$IMAGE" -lc '
python3 - <<"PY"
import importlib.metadata as md, os, glob, sys
def v(p):
    try: return md.version(p)
    except Exception as e: return f"<{e.__class__.__name__}>"
print("sglang      =", v("sglang"))
print("aiter       =", v("aiter"))
print("torch       =", v("torch"))
print("triton      =", v("triton"))
import aiter, sglang
print("aiter path  =", os.path.dirname(aiter.__file__))
print("sglang path =", os.path.dirname(sglang.__file__))
try:
    print("aiter.__version__ =", aiter.__version__)
except Exception as e:
    print("aiter.__version__ err", e)
PY
echo "===== gfx950 mla_asm.csv ====="
CSV=$(python3 -c "import aiter,os;print(os.path.dirname(aiter.__file__))")/../hsa/gfx950/mla/mla_asm.csv
ls -la "$CSV" 2>&1 || find / -path /proc -prune -o -name mla_asm.csv -print 2>/dev/null | head
for f in $(find / -path /proc -prune -o -name mla_asm.csv -print 2>/dev/null); do
  echo "--- $f"; head -1 "$f"; grep -E "^(bf16,fp8|fp8,fp8|bf16,bf16),.*,(8|16),1," "$f" | head -20
done
echo "===== sglang DSA_CHOICES ====="
python3 -c "from sglang.srt.server_args import DSA_CHOICES; print(DSA_CHOICES)"
echo "===== topk_v2 auto-disable (PR 30506) ====="
grep -rn "OPT_USE_TOPK_V2" $(python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__))") 2>/dev/null | head -10
echo "===== need_pad_heads block ====="
grep -n "need_pad_heads" -A 6 $(python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__))")/srt/layers/attention/dsa_backend.py | head -20
'
