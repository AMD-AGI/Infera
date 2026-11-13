# ruff: noqa: B023, F821 - each closure is timed in the iteration that builds it
"""Measure dLLM refinement steps (no cache: q == kv == canvas) on one GPU.

usage: python steps.py <hf_repo> <preset> <out.json> [batches] [seqs]
"""

import json
import os
import sys

import torch
from transformers import AutoModel

from infera.projection.core.projection.inference_projection.dllm.anchor import (
    AnchorPoint,
    DLLMAnchor,
    save_anchor,
    target_regime,
)
from infera.projection.core.projection.inference_projection.dllm.model import load_model

repo, preset, out = sys.argv[1], sys.argv[2], sys.argv[3]
batches = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,2,4,8,16").split(",")]
seqs = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "256,512,1024,2048").split(",")]
dev = torch.device("cuda")
spec = load_model(preset)

model = (
    AutoModel.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16)
    .to(dev)
    .eval()
)
torch.cuda.synchronize()
weights_gb = torch.cuda.memory_allocated() / 1e9
impl = getattr(model.config, "_attn_implementation", None)
print(f"{repo}: weights {weights_gb:.2f} GB, attn impl {impl}", flush=True)


def logits_of(out_):
    return out_.logits if hasattr(out_, "logits") else out_[0]


def fwd(ids):
    return logits_of(model(ids))


def sample(logits):
    conf, pred = torch.softmax(logits.float(), dim=-1).max(dim=-1)
    return conf.topk(max(1, conf.shape[-1] // 8), dim=-1).indices, pred


def timed(fn, warmup=2, iters=5):
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


rows, points = [], []
with torch.inference_mode():
    for s in seqs:
        for b in batches:
            if b * s > 32768:
                continue
            ids = torch.randint(0, spec.vocab_size - 1000, (b, s), device=dev)
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                lg = fwd(ids)
                torch.cuda.synchronize()
                peak_fwd = torch.cuda.max_memory_allocated() / 1e9 - weights_gb
                torch.cuda.reset_peak_memory_stats()
                sample(lg)
                torch.cuda.synchronize()
                peak_step = torch.cuda.max_memory_allocated() / 1e9 - weights_gb
                t_fwd = timed(lambda: fwd(ids))
                t_smp = timed(lambda: sample(lg))
                t_step = timed(lambda: sample(fwd(ids)))
                del lg
            except torch.OutOfMemoryError:
                print(f"b{b} s{s}: OOM", flush=True)
                torch.cuda.empty_cache()
                continue
            r = dict(
                batch=b,
                seq=s,
                fwd_ms=t_fwd,
                sampler_ms=t_smp,
                step_ms=t_step,
                peak_fwd_gb=peak_fwd,
                peak_step_gb=peak_step,
            )
            rows.append(r)
            points.append(AnchorPoint(b, s, s, s, t_step))
            print(
                f"b{b:>3} s{s:>5}: step {t_step:8.2f} ms  fwd {t_fwd:8.2f}  sampler {t_smp:7.2f}  "
                f"peak act {peak_step:6.2f} GB  tok/s {b * s / t_step * 1e3:9.0f}",
                flush=True,
            )

# One profiled step for the kernel-category breakdown.
prof_b, prof_s = int(os.getenv("PROF_B", "8")), int(os.getenv("PROF_S", "1024"))
cats = {}
with torch.inference_mode():
    ids = torch.randint(0, spec.vocab_size - 1000, (prof_b, prof_s), device=dev)
    for _ in range(2):
        sample(fwd(ids))
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        sample(fwd(ids))
        torch.cuda.synchronize()
    kern = []
    for ev in prof.key_averages():
        dt = getattr(ev, "device_time_total", None)
        if dt is None:
            dt = getattr(ev, "cuda_time_total", 0)
        if dt:
            kern.append((ev.key, dt / 1e3, ev.count))
    kern.sort(key=lambda x: -x[1])

    def cat(name):
        n = name.lower()
        if any(
            s in n
            for s in (
                "gemm",
                "cijk",
                "matmul",
                "_mm",
                "addmm",
                "gemv",
                "hipblaslt",
                "rocblas",
                "wvsplit",
            )
        ):
            return "gemm"
        if any(s in n for s in ("attn", "attention", "flash", "fmha", "sdpa")):
            return "attention"
        return "other"

    for k, t, c in kern:
        cats[cat(k)] = cats.get(cat(k), 0.0) + t
    top = [{"kernel": k[:120], "ms": t, "count": c, "cat": cat(k)} for k, t, c in kern[:40]]
    print(
        f"profile b{prof_b} s{prof_s}: " + ", ".join(f"{k} {v:.2f} ms" for k, v in cats.items()),
        flush=True,
    )
    for x in top[:15]:
        print(f"   {x['ms']:8.3f} ms x{x['count']:<5} [{x['cat']}] {x['kernel']}", flush=True)

os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
with open(out, "w") as f:
    json.dump(
        dict(
            repo=repo,
            preset=preset,
            device=torch.cuda.get_device_name(0),
            torch=torch.__version__,
            attn_impl=impl,
            weights_gb=weights_gb,
            rows=rows,
            profile=dict(batch=prof_b, seq=prof_s, by_cat_ms=cats, top=top),
        ),
        f,
        indent=1,
    )
regime = target_regime(spec, attention_backend="sdpa", graph_mode="eager")
save_anchor(
    DLLMAnchor(regime=regime, points=points, gpu_arch=os.getenv("ARCH", "mi325x"), backend="hf"),
    out.replace(".json", ".anchor.json"),
)
print("wrote", out, flush=True)
