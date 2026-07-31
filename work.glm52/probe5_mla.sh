#!/usr/bin/env bash
set -uo pipefail
IMAGE=lmsysorg/sglang:v0.5.16-rocm720-mi35x
docker run --rm --network host -v /mnt/vast:/mnt/vast --entrypoint bash "$IMAGE" -lc '
SG=/sgl-workspace/sglang/python/sglang
echo "===== DSA/MLA attention layer: RadixAttention with kv_lora_rank+qk_rope ====="
grep -rn "kv_lora_rank + self.qk_rope_head_dim\|kv_lora_rank + qk_rope_head_dim" $SG/srt/models/*.py | head -20
echo
echo "===== deepseek_v2.py RadixAttention (MLA absorbed) ====="
DS=$SG/srt/models/deepseek_v2.py
grep -n "RadixAttention(" -A 14 "$DS" | grep -nE "RadixAttention|head_dim|v_head_dim|num_kv_heads|kv_lora_rank" | head -40
echo
echo "===== where does GlmMoeDsa map to? ====="
grep -rn "GlmMoeDsa" $SG/srt/ --include=*.py | head -20
echo
echo "===== kv_cache_dim calc (find real file) ====="
F=$(grep -rln "TileLang and AITER DSA kernels consume\|calculate_mla_kv_cache_dim" $SG/srt/ | head -3)
echo "files: $F"
for f in $F; do echo "--- $f"; grep -n "calculate_mla_kv_cache_dim" -A 50 "$f" | head -60; done
'
