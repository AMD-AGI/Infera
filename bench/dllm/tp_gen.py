"""Whole Dream-7B generations without a cache under tensor parallelism.

torchrun --nproc_per_node TP tp_gen.py <out.json> <replica-id>

The Megatron sharding of tp_step.py; each generation is STEPS full
refreshes of the P + G canvas and the FP32 sampler. Several replicas of
the group run at once on disjoint GPUs, as a node serving 8 / TP replicas
does; each writes its own rows.
"""

import json
import os
import sys
import time

import torch
import torch.distributed as dist
from transformers import AutoModel

out, replica = sys.argv[1], int(sys.argv[2])
P = G = 256
STEPS = 64
BATCHES = [int(x) for x in os.getenv("BATCHES", "1,2,4,8,16,32").split(",")]

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
H, KVH, HD = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
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
vocab = model.lm_head.out_features


def step(ids):
    p = torch.softmax(model(ids).logits.float(), dim=-1)
    x0 = p.argmax(-1)
    return torch.gather(p, -1, x0.unsqueeze(-1))


def generation(ids, steps):
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(steps):
        step(ids)
    torch.cuda.synchronize()
    t = torch.tensor([(time.perf_counter() - t0) * 1e3], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


rows_out = []
with torch.inference_mode():
    for b in BATCHES:
        ids = torch.randint(0, vocab, (b, P + G), device="cuda")
        row = dict(tp=tp, replica=replica, batch=b)
        try:
            generation(ids, 3)
            ms = [generation(ids, STEPS) for _ in range(2)]
            row.update(meas_ms=sum(ms) / 2, runs_ms=ms)
        except torch.OutOfMemoryError:
            row.update(oom=True)
        if r == 0:
            rows_out.append(row)
            print(json.dumps(row), flush=True)
if r == 0:
    json.dump(rows_out, open(out, "w"), indent=1)
dist.destroy_process_group()
