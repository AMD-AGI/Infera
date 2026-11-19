"""Replay the simulator's generation schedules on a real dLLM and time them.

usage: python wreplay.py <family> <hf_repo> <preset> <out.json>

Every Step(q, kv, logits) runs as a forward of q positions against kv - q
cached ones (a full refresh when q == kv), then the eager sampler over the
logits positions; a step that finishes a block commits its KV. The cost is
the schedule's; the tokens are not decoded. family: dream | sdar.
"""

import json
import os
import sys
import time

import torch
from transformers import AutoModel, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from infera.projection.core.projection.inference_projection.dllm.workload import DLLMWorkload

family, repo, preset, out_path = sys.argv[1:5]
P, G = int(os.getenv("P", "256")), int(os.getenv("G", "256"))
cls = AutoModel if family == "dream" else AutoModelForCausalLM
model = cls.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
vocab = model.config.vocab_size


def forward(ids, past, commit):
    """Logits for ids against the legacy cache ``past``; the cache with ids appended if commit."""
    cache = DynamicCache() if past is None else DynamicCache.from_legacy_cache(past)
    if family == "dream":
        o = model.model(ids, past_key_values=cache, use_cache=True)
        return model.lm_head(o.last_hidden_state), cache.to_legacy_cache()
    kv = ids.shape[1] + (0 if past is None else past[0][0].shape[2])
    mask = torch.ones(1, 1, ids.shape[1], kv, dtype=torch.bool, device=ids.device)
    o = model(ids, attention_mask=mask, past_key_values=cache, use_cache=True, store_kv=commit)
    return o.logits, (cache.to_legacy_cache() if commit else None)


def sampler(logits, k):
    p = torch.softmax(logits.float(), dim=-1)
    x0 = p.argmax(-1)
    conf = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
    return x0, torch.topk(conf, max(1, k), dim=-1).indices


def run(wl, batch):
    canvas = torch.randint(0, vocab - 1000, (batch, wl.canvas), device="cuda")
    cache = None
    per_step = max(1, wl.block_len // max(1, wl.steps // wl.num_blocks))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for st in wl.schedule():
        q, kv, lg = st.shape.q, st.shape.kv, st.shape.logits
        if q == kv:
            logits, full = forward(canvas[:, :q], None, wl.cache != "none")
            if wl.cache != "none":
                cache = full
        else:
            past = tuple((k[:, :, : kv - q], v[:, :, : kv - q]) for k, v in cache)
            commit = wl.cache == "block" and st.finishes_block >= 0
            logits, new = forward(canvas[:, kv - q : kv], past, commit)
            if commit:
                cache = new
        if lg:
            sampler(logits[:, -lg:], per_step)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


if family == "dream":
    cases = [("none", 0, s) for s in (256, 64)] + [
        (c, 32, s) for c in ("prefix", "dual", "block") for s in (256, 128, 64)
    ]
else:
    cases = [("block", b, s) for b in (4, 32) for s in (256, 128, 64)]
rows = []
with torch.inference_mode():
    for cache, block, steps in cases:
        wl = DLLMWorkload(prompt_len=P, gen_len=G, block_len=block, steps=steps, cache=cache)
        for batch in (1, 8, 32):
            warm = DLLMWorkload(
                prompt_len=P,
                gen_len=G,
                block_len=block,
                steps=wl.num_blocks * (2 if cache != "none" else 1),
                cache=cache,
            )
            run(warm, batch)
            ms = run(wl, batch)
            row = dict(
                cache=cache,
                block=block or G,
                steps=steps,
                batch=batch,
                meas_ms=ms,
                meas_tok_s=batch * G / ms * 1e3,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
json.dump(
    {"family": family, "repo": repo, "preset": preset, "P": P, "G": G, "rows": rows},
    open(out_path, "w"),
    indent=1,
)
