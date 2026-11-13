"""A tensor-parallel Dream-7B refinement step, measured and simulated.

torchrun --nproc_per_node TP tp_step.py out.json

Megatron sharding by hand on the HF model: q/k/v, gate and up by output
rows, o and down by input columns with an all-reduce after each, the LM head
by vocabulary. With more ranks than KV heads each KV head is replicated.
"""

import json
import os
import sys
import time

import torch
import torch.distributed as dist
from transformers import AutoModel

from infera.projection.core.projection.inference_projection.dllm import (
    DLLMHardware,
    DLLMStepModel,
    Parallel,
    SeqShape,
    load_model,
)

dist.init_process_group("nccl")
r, tp = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(r)
model = (
    AutoModel.from_pretrained(
        "Dream-org/Dream-v0-Instruct-7B", trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    .cuda()
    .eval()
)
cfg = model.config
H, KVH, HD = (
    cfg.num_attention_heads,
    cfg.num_key_value_heads,
    cfg.hidden_size // cfg.num_attention_heads,
)
hq = H // tp
kvl = max(1, KVH // tp)
kv0 = (r * KVH // tp) if tp <= KVH else r // (tp // KVH)


def rows(lin, start, n):
    lin.weight = torch.nn.Parameter(lin.weight.data[start : start + n].clone())
    if lin.bias is not None:
        lin.bias = torch.nn.Parameter(lin.bias.data[start : start + n].clone())
    lin.out_features = n


def cols(lin, start, n):
    lin.weight = torch.nn.Parameter(lin.weight.data[:, start : start + n].clone())
    lin.in_features = n


def reduce_hook(_m, _i, out):
    dist.all_reduce(out)
    return out


if tp > 1:
    for layer in model.model.layers:
        a = layer.self_attn
        rows(a.q_proj, r * hq * HD, hq * HD)
        rows(a.k_proj, kv0 * HD, kvl * HD)
        rows(a.v_proj, kv0 * HD, kvl * HD)
        cols(a.o_proj, r * hq * HD, hq * HD)
        a.num_heads, a.num_key_value_heads = hq, kvl
        a.num_key_value_groups = hq // kvl
        a.hidden_size = hq * HD
        a.o_proj.register_forward_hook(reduce_hook)
        m = layer.mlp
        f = m.gate_proj.out_features // tp
        rows(m.gate_proj, r * f, f)
        rows(m.up_proj, r * f, f)
        cols(m.down_proj, r * f, f)
        m.down_proj.register_forward_hook(reduce_hook)
    v = model.lm_head.out_features // tp
    rows(model.lm_head, r * v, v)
    torch.cuda.empty_cache()

hw = DLLMHardware.from_arch("mi325x", gpu_clock_mhz=1510)
sim = DLLMStepModel(load_model("dream_7B"), Parallel(tp=tp), hw, runtime="hf_eager")
vocab = model.lm_head.out_features


def step(ids):
    logits = model(ids).logits
    p = torch.softmax(logits.float(), dim=-1)
    x0 = p.argmax(-1)
    return torch.gather(p, -1, x0.unsqueeze(-1))


rows_out = []
with torch.no_grad():
    for b in (1, 8):
        for s in (512, 1024, 2048):
            ids = torch.randint(0, vocab, (b, s), device="cuda")
            for _ in range(2):
                step(ids)
            torch.cuda.synchronize()
            dist.barrier()
            t0 = time.perf_counter()
            n = 5
            for _ in range(n):
                step(ids)
            torch.cuda.synchronize()
            t = torch.tensor([(time.perf_counter() - t0) / n * 1e3], device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            bd = sim.step([SeqShape(s, s, s)] * b)
            d = bd.as_dict()
            row = {
                "tp": tp,
                "batch": b,
                "seq": s,
                "meas_ms": t.item(),
                "sim_ms": bd.total_ms,
                "ratio": bd.total_ms / t.item(),
                "sim_comm_ms": d["exposed_comm_ms"],
                "sim_compute_ms": d["compute_ms"],
            }
            if r == 0:
                rows_out.append(row)
                print(json.dumps(row), flush=True)
if r == 0:
    path = sys.argv[1]
    old = json.load(open(path)) if os.path.exists(path) else []
    json.dump(old + rows_out, open(path, "w"), indent=1)
dist.destroy_process_group()
