# ruff: noqa: B023 - each closure is timed in the iteration that builds it
"""Measure refinement steps of one dLLM on one GPU, with and without a KV cache.

usage: python wsteps.py <family> <hf_repo> <preset> <out.json>

A step is a forward over q positions attending to kv (kv - q of them cached;
q == kv is a full refresh) and the confidence sampler over those q logits.
family: dream | llada | sdar | lladamoe | llada2. LLaDA, LLaDA-MoE and LLaDA2.0
ship no KV cache, so only their full steps are measured; LLaDA2.0's carry the
block-causal mask its generate passes.
"""

import json
import os
import sys

import torch
from transformers import AutoModel, AutoModelForCausalLM

family, repo, preset, out = sys.argv[1:5]
SMOKE = bool(os.getenv("SMOKE"))
dev = torch.device("cpu" if SMOKE else "cuda")
FULL_S = [int(x) for x in os.getenv("FULL_S", "256,512,1024,2048,4096").split(",")]
FULL_B = [int(x) for x in os.getenv("FULL_B", "1,2,4,8,16,32").split(",")]
CACHE_Q = [int(x) for x in os.getenv("CACHE_Q", "32,128").split(",")]
CACHE_KV = [int(x) for x in os.getenv("CACHE_KV", "512,1024,2048,4096").split(",")]
CACHE_B = [int(x) for x in os.getenv("CACHE_B", "1,8,32").split(",")]
MAX_TOK = int(os.getenv("MAX_TOK", "65536"))

cls = AutoModel if family == "dream" else AutoModelForCausalLM
model = cls.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev).eval()
vocab = model.config.vocab_size
weights_gb = 0.0 if SMOKE else torch.cuda.memory_allocated() / 1e9
gpu = "cpu" if SMOKE else torch.cuda.get_device_properties(0).uuid
print(
    f"{repo}: {weights_gb:.2f} GB on {gpu}, attn {getattr(model.config, '_attn_implementation', None)}",
    flush=True,
)


def new_cache():
    from transformers.cache_utils import DynamicCache

    return DynamicCache()


def forward(ids, cache=None, store=False):
    """Logits for ids; with a cache, ids attend to its contents."""
    if family == "dream":
        o = model.model(ids, past_key_values=cache, use_cache=cache is not None)
        return model.lm_head(o.last_hidden_state)
    if family == "sdar":
        kv = ids.shape[1] + (cache.get_seq_length() if cache is not None else 0)
        mask = torch.ones(1, 1, ids.shape[1], kv, dtype=torch.bool, device=dev)
        return model(
            ids,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=cache is not None,
            store_kv=store,
        ).logits
    if family == "llada2":
        return model(ids, attention_mask=block_causal(ids.shape[0], ids.shape[1])).logits
    if cache is None:
        return model(ids).logits
    return model(ids, past_key_values=cache, use_cache=True).logits


_masks = {}


def block_causal(b, s, block=32):
    """The additive bf16 block-causal mask LLaDA2.0's generate passes every step."""
    if s not in _masks:
        n = -(-s // block)
        m = (
            torch.tril(torch.ones(n, n, device=dev))
            .repeat_interleave(block, 0)
            .repeat_interleave(block, 1)
        )
        _masks[s] = m[:s, :s].log().to(torch.bfloat16)[None, None]
    return _masks[s].expand(b, 1, s, s)


def sample(logits):
    conf, pred = torch.softmax(logits.float(), dim=-1).max(dim=-1)
    return conf.topk(max(1, conf.shape[-1] // 8), dim=-1).indices, pred


def timed(fn, warmup=2, iters=5):
    if SMOKE:
        fn()
        return 0.0
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def cached_step_fn(b, q, kv):
    """A step of q positions against kv - q cached ones, the cache restored after."""
    prefix = torch.randint(0, vocab - 1000, (b, kv - q), device=dev)
    ids = torch.randint(0, vocab - 1000, (b, q), device=dev)
    cache = new_cache()
    forward(prefix, cache, store=True)
    if family == "sdar":
        return lambda: sample(forward(ids, cache, store=False))

    def step():
        r = sample(forward(ids, cache))
        cache.crop(kv - q)
        return r

    return step


FULL_ONLY = {"llada", "lladamoe", "llada2"}  # their reference code keeps no KV cache
shapes = [(b, s, s) for s in FULL_S for b in FULL_B if b * s <= MAX_TOK]
if family not in FULL_ONLY:
    shapes += [
        (b, q, kv)
        for q in CACHE_Q
        for kv in CACHE_KV
        for b in CACHE_B
        if q < kv and b * kv <= 4 * MAX_TOK
    ]
if SMOKE:
    shapes = [(1, 16, 16), (2, 4, 32)] if family not in FULL_ONLY else [(1, 16, 16), (2, 40, 40)]
rows = []
with torch.inference_mode():
    for b, q, kv in shapes:
        try:
            if not SMOKE:
                torch.cuda.empty_cache()
            if q == kv:
                ids = torch.randint(0, vocab - 1000, (b, q), device=dev)
                fn = lambda: sample(forward(ids))
            else:
                fn = cached_step_fn(b, q, kv)
            ms = timed(fn)
        except torch.OutOfMemoryError:
            print(f"b{b} q{q} kv{kv}: OOM", flush=True)
            continue
        rows.append(dict(batch=b, q=q, kv=kv, logits=q, step_ms=ms))
        print(f"b{b:>3} q{q:>5} kv{kv:>5}: {ms:9.2f} ms", flush=True)
        del fn
json.dump(
    dict(
        family=family,
        repo=repo,
        preset=preset,
        device="cpu" if SMOKE else torch.cuda.get_device_name(0),
        weights_gb=weights_gb,
        rows=rows,
    ),
    open(out, "w"),
    indent=1,
)
print("wrote", out, flush=True)
