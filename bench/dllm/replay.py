"""Replay the simulator's refinement-step schedule on the real model, and simulate it.

python replay.py <hf_repo> <preset> <out.json>
Every Step(q, kv, logits) runs as: a forward of q positions against kv - q
cached ones (a full refresh when q == kv), then the eager sampler over the
logits positions. The cost is the schedule's; the tokens are not decoded.
"""

import json
import os
import sys
import time

import torch
from transformers import AutoModel, AutoModelForCausalLM

from infera.projection.core.projection.inference_projection.dllm import (
    DLLMHardware,
    DLLMStepModel,
    Parallel,
    load_model,
)
from infera.projection.core.projection.inference_projection.dllm.simulator import simulate_static
from infera.projection.core.projection.inference_projection.dllm.workload import DLLMWorkload

repo, preset, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
P, G = int(os.getenv("P", "256")), int(os.getenv("G", "256"))
BLOCK = int(os.getenv("BLOCK", "32"))
dream = "Dream" in repo
cls = AutoModel if dream else AutoModelForCausalLM
model = cls.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
vocab = model.config.vocab_size
spec = load_model(preset)
hw = DLLMHardware.from_arch("mi325x", gpu_clock_mhz=1510)
sim = DLLMStepModel(spec, Parallel(), hw, runtime="hf_eager")


def legacy(pkv):
    if hasattr(pkv, "to_legacy_cache"):
        return pkv.to_legacy_cache()
    return pkv


def forward(ids, past):
    if dream:
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache() if past is None else DynamicCache.from_legacy_cache(past)
        o = model.model(ids, past_key_values=cache, use_cache=True)
        return model.lm_head(o.last_hidden_state), cache.to_legacy_cache()
    o = model(ids, past_key_values=past, use_cache=True)
    return o.logits, o.past_key_values


def sampler(logits, k):
    p = torch.softmax(logits.float(), dim=-1)
    x0 = p.argmax(-1)
    conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
    return x0, torch.topk(conf, max(1, k), dim=-1).indices


def run(wl, batch):
    sched = wl.schedule()
    canvas = torch.randint(0, vocab, (batch, wl.canvas), device="cuda")
    cache = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for st in sched:
        q, kv, lg = st.shape.q, st.shape.kv, st.shape.logits
        if q == kv:
            logits, full = forward(canvas[:, :q], None)
            if wl.cache != "none":
                cache = full
        else:
            past = tuple((k[:, :, : kv - q], v[:, :, : kv - q]) for k, v in cache)
            logits, new = forward(canvas[:, kv - q : kv], past)
            if wl.cache == "block" and st.finishes_block >= 0:
                cache = new
        if lg:
            sampler(logits[:, -lg:], max(1, wl.block_len // max(1, wl.steps // wl.num_blocks)))
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


rows = []
cases = [("none", 0, s) for s in (256, 64)] + [
    (c, BLOCK, s) for c in ("none", "prefix", "dual", "block") for s in (256, 128, 64)
]
if os.getenv("SMOKE"):
    cases = [("none", BLOCK, 8), ("dual", BLOCK, 16), ("block", BLOCK, 16)]
for cache, block, steps in cases:
    wl = DLLMWorkload(prompt_len=P, gen_len=G, block_len=block, steps=steps, cache=cache)
    for batch in (1, 8):
        warm = 2 * wl.num_blocks if cache != "none" else wl.num_blocks
        run(DLLMWorkload(prompt_len=P, gen_len=G, block_len=block, steps=warm, cache=cache), batch)
        ms = run(wl, batch)
        s = simulate_static(sim, wl, batch)
        row = {
            "cache": cache,
            "block": block or G,
            "steps": steps,
            "batch": batch,
            "meas_ms": ms,
            "sim_ms": s.latency_ms,
            "ratio": s.latency_ms / ms,
            "meas_tok_s": batch * G / ms * 1e3,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
json.dump({"repo": repo, "P": P, "G": G, "rows": rows}, open(out_path, "w"), indent=1)
