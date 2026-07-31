#!/usr/bin/env bash
# Find what head_dim the GLM-5.2 DSA MLA RadixAttention layer is constructed with,
# inside the v0.5.16 image. Static source read + config math. No GPU alloc.
set -uo pipefail
IMAGE=lmsysorg/sglang:v0.5.16-rocm720-mi35x
docker run --rm --network host -v /mnt/vast:/mnt/vast --entrypoint bash "$IMAGE" -lc '
SG=/sgl-workspace/sglang/python/sglang
echo "===== which model file serves GlmMoeDsaForCausalLM ====="
grep -rln "GlmMoeDsaForCausalLM\|glm_moe_dsa" $SG/srt/models/ | head
echo
F=$(grep -rln "GlmMoeDsaForCausalLM" $SG/srt/models/ | head -1)
echo "model file = $F"
echo
echo "===== RadixAttention construction in that file ====="
grep -n "RadixAttention(" -A 16 "$F" | head -60
echo
echo "===== head_dim / v_head_dim assignments ====="
grep -nE "self\.(head_dim|v_head_dim|qk_head_dim|kv_lora_rank|qk_rope_head_dim|qk_nope_head_dim) *=" "$F" | head -30
echo
echo "===== the aiter view line in dsa_backend ====="
grep -n "view(-1, 1, 1, layer.head_dim)" -B 4 -A 2 $SG/srt/layers/attention/dsa_backend.py
echo
echo "===== calculate_mla_kv_cache_dim HIP branch ====="
grep -n "TileLang and AITER DSA kernels consume" -B 4 -A 8 $SG/srt/model_executor/model_runner_kv_cache_mixin.py
echo
echo "===== config math ====="
python3 - <<"PY"
import json
c=json.load(open("/mnt/vast/xiaobo/models/GLM-5.2-MXFP4/config.json"))
kl,qr=c["kv_lora_rank"],c["qk_rope_head_dim"]
print("  kv_lora_rank+qk_rope_head_dim (pool kv_cache_dim) =", kl+qr)
print("  config head_dim    =", c.get("head_dim"))
print("  config qk_head_dim =", c.get("qk_head_dim"))
print("  config v_head_dim  =", c.get("v_head_dim"))
print("  num_attention_heads=", c["num_attention_heads"], "-> TP8 =", c["num_attention_heads"]//8)
PY
'
