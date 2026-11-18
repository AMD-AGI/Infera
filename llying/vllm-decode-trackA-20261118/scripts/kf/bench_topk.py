"""GLM-5.2 DSA indexer decode top-k (k = 2048) at vLLM's flattened MTP3 decode shapes: 4 rows per request with
lengths L-3..L, L uniform in [CTX_MIN, CTX_MAX], logits rows max_model_len wide (vLLM v0.34 resolves GLM-5.2 to
1,048,576, and the FULL decode graph uses that bound).

Candidates:
  aiter     aiter.ops.topk.top_k_per_row_decode (what vLLM v0.34 runs: rocm_aiter_ops.indexer_top_k_decode)
  vllm_c    torch.ops._C.top_k_per_row_decode (vLLM's in-tree fallback)
  triton    sglang PR #47872 single-pass est/collect/select, vLLM port (vllm_triton_topk_decode.py)

Exactness check per row (tie-robust): min(L, k) distinct in-range indices, every selected logit >= the k-th largest
logit of the row (torch.topk reference), -1 padding after. Timing: CUDA events per call; FLUSH=1 writes a 512 MB
buffer before each call so the logits are not served from the 256 MB MALL (default 0: in the engine the paged-logits
kernel has just written them).
Usage: python bench_topk.py [rows ...]   env CTX_MIN (40000), CTX_MAX (200000), ITERS (50), DIST (randn|relu|ties),
       MAX_MODEL_LEN (1048576), FLUSH (0), CANDS (aiter vllm_c triton)
"""

import os
import sys

import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

TOPK, NEXT_N = 2048, 4
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", 1048576))
CTX_MIN = int(os.environ.get("CTX_MIN", 40000))
CTX_MAX = int(os.environ.get("CTX_MAX", 200000))
ITERS = int(os.environ.get("ITERS", 50))
DIST = os.environ.get("DIST", "randn")
FLUSH = os.environ.get("FLUSH", "0") == "1"
dev = torch.device("cuda")


def make_inputs(rows, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    reqs = (rows + NEXT_N - 1) // NEXT_N
    ctx = torch.randint(CTX_MIN, CTX_MAX + 1, (reqs,), generator=g, dtype=torch.int32)
    lens = (ctx[:, None] - (NEXT_N - 1) + torch.arange(NEXT_N, dtype=torch.int32)[None]).reshape(-1)[:rows]
    lens = lens.contiguous().to(dev)
    logits = torch.full((rows, MAX_MODEL_LEN), float("-inf"), device=dev)
    gd = torch.Generator(device=dev).manual_seed(seed)
    for r in range(rows):
        n = int(lens[r])
        x = torch.randn(n, device=dev, generator=gd)
        if DIST == "relu":
            x = torch.relu(x) * (1 + torch.rand(n, device=dev, generator=gd)) + 0.01 * x
        elif DIST == "ties":
            x = torch.round(x * 4) / 4
        logits[r, :n] = x
    return logits, lens


def check(logits, lens, out):
    bad = 0
    for r in range(logits.shape[0]):
        n = int(lens[r])
        k = min(n, TOPK)
        idx = out[r, :TOPK]
        sel, pad = idx[:k], idx[k:]
        if (pad != -1).any() or (sel < 0).any() or (sel >= n).any() or sel.unique().numel() != k:
            bad += 1
            continue
        kth = torch.topk(logits[r, :n], k).values[-1]
        if (logits[r, sel.long()] < kth).any():
            bad += 1
    return bad


def main():
    rows_list = [int(a) for a in sys.argv[1:]] or [16, 32, 64, 104, 128]
    cands = {}
    want = os.environ.get("CANDS", "aiter vllm_c triton").split()
    if "aiter" in want:
        from aiter.ops.topk import top_k_per_row_decode as aiter_topk

        cands["aiter"] = lambda lg, ln, o: aiter_topk(lg, 1, ln, o, lg.shape[0], lg.stride(0), lg.stride(1), k=TOPK)
    if "vllm_c" in want:
        import vllm._custom_ops  # noqa: F401

        cands["vllm_c"] = lambda lg, ln, o: torch.ops._C.top_k_per_row_decode(
            lg, 1, ln, o, lg.shape[0], lg.stride(0), lg.stride(1), TOPK)
    if "triton" in want:
        import vllm_triton_topk_decode as tk

        cands["triton"] = lambda lg, ln, o: tk.triton_top_k_per_row_decode(lg, ln, o, TOPK)
    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name} {props.gcnArchName}; ctx {CTX_MIN}-{CTX_MAX}, row width {MAX_MODEL_LEN}, dist {DIST}, "
          f"flush {int(FLUSH)}, iters {ITERS}", flush=True)
    print("rows | " + " | ".join(f"{n} us (bad rows)" for n in cands) + " | triton vs aiter", flush=True)
    flush_buf = torch.empty(512 << 20, dtype=torch.uint8, device=dev) if FLUSH else None
    for rows in rows_list:
        logits, lens = make_inputs(rows, 7 + rows)
        out = torch.empty((rows, TOPK), dtype=torch.int32, device=dev)
        cells, times = [], {}
        for name, fn in cands.items():
            try:
                out.fill_(-7)
                fn(logits, lens, out)
                torch.cuda.synchronize()
                bad = check(logits, lens, out)
                for _ in range(3):
                    fn(logits, lens, out)
                ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(ITERS)]
                for s, e in ev:
                    if flush_buf is not None:
                        flush_buf.fill_(1)
                    s.record()
                    fn(logits, lens, out)
                    e.record()
                torch.cuda.synchronize()
                ts = sorted(s.elapsed_time(e) * 1000 for s, e in ev)
                us = ts[len(ts) // 2]
                times[name] = us
                cells.append(f"{us:.1f} ({bad})")
            except Exception as e:  # report and keep going
                cells.append(f"error: {type(e).__name__}: {str(e)[:120]}")
        rel = f"{times['aiter'] / times['triton']:.2f}x" if "aiter" in times and "triton" in times else "-"
        print(f"{rows} | " + " | ".join(cells) + f" | {rel}", flush=True)


if __name__ == "__main__":
    main()
