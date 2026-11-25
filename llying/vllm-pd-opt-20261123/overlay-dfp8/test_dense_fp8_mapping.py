# In-image CPU check (VLLM_TARGET_DEVICE=cpu): which layers the online FP8 config would quantize.
# usage: python3 test_dense_fp8_mapping.py <model_dir> dense_fp8_qcfg.json
import json, sys, re
from vllm.config.quantization import QuantizationConfigArgs
from vllm.model_executor.layers.quantization.online.base import OnlineQuantizationConfig
from vllm.model_executor.layers.linear import (ColumnParallelLinear, MergedColumnParallelLinear,
                                               ReplicatedLinear, RowParallelLinear)
cfg = json.load(open(sys.argv[1] + "/config.json")); args = json.load(open(sys.argv[2]))
oc = OnlineQuantizationConfig(QuantizationConfigArgs(**args))
oc.packed_modules_mapping = {"fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
                             "gate_up_proj": ["gate_proj", "up_proj"], "wk_weights_proj": ["wk", "weights_proj"]}
L = cfg["num_hidden_layers"] + cfg.get("num_nextn_predict_layers", 0)
mods = [("self_attn.fused_qkv_a_proj", MergedColumnParallelLinear), ("self_attn.q_b_proj", ColumnParallelLinear),
        ("self_attn.kv_b_proj", ColumnParallelLinear), ("self_attn.o_proj", RowParallelLinear),
        ("self_attn.indexer.wq_b", ReplicatedLinear), ("mlp.gate_up_proj", MergedColumnParallelLinear),
        ("mlp.down_proj", RowParallelLinear), ("mlp.gate", ReplicatedLinear),
        ("mlp.shared_experts.gate_up_proj", MergedColumnParallelLinear), ("mlp.shared_experts.down_proj", RowParallelLinear),
        ("eh_proj", ReplicatedLinear)]
bad = []; n = 0
for i in range(L):
    for m, cls in mods:
        p = f"model.layers.{i}.{m}"; r = oc.resolve_quant_method_cls(cls, p)
        want = (i < cfg["num_hidden_layers"] and (m in ("self_attn.fused_qkv_a_proj", "self_attn.q_b_proj", "self_attn.o_proj", "self_attn.indexer.wq_b")
                or (i < cfg["first_k_dense_replace"] and m in ("mlp.gate_up_proj", "mlp.down_proj"))))
        n += r is not None
        if (r is not None) != want: bad.append((p, r))
print("quantized prefixes:", n, "mismatches:", bad[:10]); assert not bad
assert all(oc.resolve_quant_method_cls(c, f"model.layers.{L-1}.{m}") is None for m, c in mods), "MTP layer touched"
print("OK: MTP layer", L - 1, "untouched")
