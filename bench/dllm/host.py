"""Host floor of an eager HF step: wall vs GPU-busy at small q against a cache.

python host.py <hf_repo> <out.json>
"""

import json
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModel, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

repo, out_path = sys.argv[1], sys.argv[2]
dream = "Dream" in repo
llada = "LLaDA" in repo
cls = AutoModel if dream else AutoModelForCausalLM
model = cls.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
vocab = model.config.vocab_size
KV = 512


def make_cache(b):
    ids = torch.randint(0, vocab, (b, KV), device="cuda")
    with torch.no_grad():
        if llada:
            return model(ids, use_cache=True).past_key_values
        c = DynamicCache()
        (model.model if dream else model)(ids, past_key_values=c, use_cache=True)
        return c.to_legacy_cache()


def step(ids, past):
    if llada:
        return model(ids, past_key_values=past, use_cache=False).logits
    c = DynamicCache.from_legacy_cache(past)
    if dream:
        return model.lm_head(model.model(ids, past_key_values=c, use_cache=True).last_hidden_state)
    return model(ids, past_key_values=c, use_cache=True).logits


rows = []
with torch.no_grad():
    for b in (1, 8):
        past = make_cache(b)
        for q in (1, 8, 32, 128):
            ids = torch.randint(0, vocab, (b, q), device="cuda")
            for _ in range(3):
                step(ids, past)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            n = 10
            for _ in range(n):
                step(ids, past)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t0) / n * 1e3
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                step(ids, past)
                torch.cuda.synchronize()
            ev = [e for e in prof.events() if e.device_type.name == "CUDA"]
            busy = sum(e.device_time for e in ev) / 1e3 if ev else 0.0
            row = {
                "batch": b,
                "q": q,
                "kv": KV + q,
                "wall_ms": wall,
                "busy_ms": busy,
                "kernels": len(ev),
            }
            if (b, q) in ((1, 1), (8, 32)):
                agg = {}
                for e in ev:
                    a = agg.setdefault(e.name[:70], [0, 0.0])
                    a[0] += 1
                    a[1] += e.device_time / 1e3
                row["top"] = sorted(
                    ([k, n, round(ms, 3)] for k, (n, ms) in agg.items()), key=lambda x: -x[2]
                )[:18]
            rows.append(row)
            print(json.dumps(row), flush=True)
json.dump({"repo": repo, "rows": rows}, open(out_path, "w"), indent=1)
