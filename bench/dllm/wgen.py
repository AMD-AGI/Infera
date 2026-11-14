"""Run predict-then-run candidates: whole generations of one dLLM on one GPU.

usage: python wgen.py <family> <hf_repo> <preset> <predictions.json> <out.json>

Every Study A candidate of the preset is replayed from the simulator's
schedule, as wreplay.py does, for every family: full refreshes for the
models without a cache (LLaDA2.0 with its block-causal mask), Dream's
prefix and dual caches, and SDAR's exact block cache. Each candidate is
warmed up and then timed twice; an out-of-memory candidate is recorded.
"""

import json
import sys
import time

import torch
from transformers import AutoModel, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from infera.projection.core.projection.inference_projection.dllm.workload import DLLMWorkload

family, repo, preset, pred_path, out_path = sys.argv[1:6]
pred = json.load(open(pred_path))
P, G, STEPS = pred["P"], pred["G"], pred["steps"]
cands = pred["study_a"][preset]["candidates"]
t_load = time.perf_counter()
cls = AutoModel if family == "dream" else AutoModelForCausalLM
model = cls.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
load_s = time.perf_counter() - t_load
vocab = model.config.vocab_size
_masks = {}


def block_causal(b, s, block=32):
    if s not in _masks:
        n = -(-s // block)
        m = torch.tril(torch.ones(n, n, device="cuda"))
        m = m.repeat_interleave(block, 0).repeat_interleave(block, 1)
        _masks[s] = m[:s, :s].log().to(torch.bfloat16)[None, None]
    return _masks[s].expand(b, 1, s, s)


def forward(ids, past, commit):
    """Logits for ids against the legacy cache ``past``; the extended cache if commit."""
    if family in ("llada", "lladamoe"):
        return model(ids).logits, None
    if family == "llada2":
        return model(ids, attention_mask=block_causal(*ids.shape)).logits, None
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
    return x0, torch.topk(conf, max(1, min(k, conf.shape[-1])), dim=-1).indices


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


rows = []
t_all = time.perf_counter()
with torch.inference_mode():
    for c in cands:
        wl = DLLMWorkload(prompt_len=P, gen_len=G, block_len=c["block"], steps=STEPS, cache=c["cache"])
        warm = DLLMWorkload(
            prompt_len=P,
            gen_len=G,
            block_len=c["block"],
            steps=min(STEPS, wl.num_blocks * (2 if c["cache"] != "none" else 1) + 1),
            cache=c["cache"],
        )
        row = dict(cache=c["cache"], block=c["block"], batch=c["batch"])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            run(warm, c["batch"])
            ms = [run(wl, c["batch"]) for _ in range(2)]
            row.update(meas_ms=sum(ms) / 2, runs_ms=ms, peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        except torch.OutOfMemoryError:
            row.update(oom=True)
        rows.append(row)
        print(json.dumps(row), flush=True)
json.dump(
    dict(
        family=family,
        repo=repo,
        preset=preset,
        load_s=load_s,
        sweep_s=time.perf_counter() - t_all,
        rows=rows,
    ),
    open(out_path, "w"),
    indent=1,
)
print("wrote", out_path, flush=True)
