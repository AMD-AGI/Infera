"""GLM-5.2 DSA indexer decode logits microbenchmark at vLLM's DP-attention decode shapes (32 heads x 128, fp8,
64-token preshuffled pages, MTP3 so 4 query rows per request).

Candidates:
  aiter     aiter deepgemm_fp8_paged_mqa_logits on [B, next_n=4] with per-request context lengths, ChunkK 256,
            WavePerEU 2
  aiter_flat  what vLLM v0.34 actually runs on ROCm with MTP3: decode is flattened (indexer._use_flattening), so
            aiter gets one query per row with per-row lengths and block-table rows
  gluon     sglang PR #47897 gluon_fp8_paged_mqa_logits as sglang calls it: one query per row, per-row context
            lengths and per-row (expanded) block tables; tunables as in the PR (6-row LPT groups)
  gluon_g4  the same with 4-row LPT groups / NG2 32, matching MTP3 verify rows (the team's K5 setting)

gluon's per-row inputs (lengths, expanded block table) are built once per step in vLLM and shared by every indexer
layer, so they are not timed; the cost of building them is printed separately. Correctness: every candidate's
logits on each row's visible columns against aiter's.
Usage: python bench_paged_logits.py [batch ...]   env CTX_MIN (40000), CTX_MAX (200000), NPOOL (4), ITERS (40)
"""

import os
import sys

import torch

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)

HEADS, DIM, PAGE, NEXT_N = 32, 128, 64, 4
MAX_MODEL_LEN = 202752
CTX_MIN = int(os.environ.get("CTX_MIN", 40000))
CTX_MAX = int(os.environ.get("CTX_MAX", 200000))
NPOOL = int(os.environ.get("NPOOL", 4))
ITERS = int(os.environ.get("ITERS", 40))
FP8 = torch.float8_e4m3fn
dev = torch.device("cuda")


def make_inputs(batch, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ctx = torch.randint(CTX_MIN, CTX_MAX + 1, (batch,), generator=g, dtype=torch.int32)
    blocks = (ctx + PAGE - 1) // PAGE
    total = int(blocks.sum())
    max_blocks = (MAX_MODEL_LEN + PAGE - 1) // PAGE
    perm = torch.randperm(total, generator=g).to(torch.int32)
    table = torch.zeros(batch, max_blocks, dtype=torch.int32)
    off = 0
    for b in range(batch):
        n = int(blocks[b])
        table[b, :n] = perm[off : off + n]
        off += n
    kv = torch.empty(total, PAGE * (DIM + 4), dtype=torch.uint8, device=dev)
    kv[:, : PAGE * DIM] = (torch.randn(total, PAGE * DIM, device=dev) * 0.5).to(FP8).view(torch.uint8)
    kv[:, PAGE * DIM :] = torch.ones(total, PAGE, dtype=torch.float32, device=dev).view(torch.uint8)
    q = (torch.randn(batch, NEXT_N, HEADS, DIM, device=dev) * 0.5).to(FP8)
    w = torch.rand(batch * NEXT_N, HEADS, device=dev, dtype=torch.float32)
    return dict(ctx=ctx.to(dev), table=table.to(dev), kv=kv.view(total, PAGE, 1, DIM + 4), q=q, w=w)


def per_row(inp):
    """sglang's per-row form: row j of a request sees ctx - (NEXT_N - 1 - j) tokens."""
    ctx = inp["ctx"]
    lens = (ctx[:, None] - (NEXT_N - 1) + torch.arange(NEXT_N, device=dev, dtype=torch.int32)[None]).reshape(-1)
    return lens.contiguous(), inp["table"].repeat_interleave(NEXT_N, dim=0).contiguous()


def run_aiter(fn, inp, out):
    fn(inp["q"], inp["kv"], inp["w"], out, inp["ctx"], inp["table"], MAX_MODEL_LEN,
       ChunkK=256, Preshuffle=True, KVBlockSize=PAGE, WavePerEU=2)


def run_aiter_flat(fn, inp, out):
    b = inp["q"].shape[0]
    fn(inp["q"].view(b * NEXT_N, 1, HEADS, DIM), inp["kv"], inp["w"], out, inp["row_lens"], inp["row_table"],
       MAX_MODEL_LEN, ChunkK=256, Preshuffle=True, KVBlockSize=PAGE, WavePerEU=2)


def make_gluon(group_rows, ng2):
    import sgl47897_gluon_paged_mqa_logits as gm

    def run(inp, out):
        gm._LPT_GROUP_ROWS, gm._LPT_NG2 = group_rows, ng2
        b = inp["q"].shape[0]
        gm.gluon_fp8_paged_mqa_logits(inp["q"].view(b * NEXT_N, 1, HEADS, DIM), inp["kv"], inp["w"], out,
                                      inp["row_lens"], inp["row_table"], MAX_MODEL_LEN,
                                      Preshuffle=True, KVBlockSize=PAGE, ChunkK=256)

    return run


def timed(fn, inps, out):
    for inp in inps:
        fn(inp, out)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for it in range(ITERS):
        fn(inps[it % len(inps)], out)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / ITERS


def visible_diff(ref, out, row_lens):
    cols = torch.arange(ref.shape[1], device=dev)[None]
    mask = cols < row_lens[:, None]
    a, b = ref[mask], out[mask]
    return float((a - b).abs().max()), float(a.abs().max())


def main():
    try:
        from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits as aiter_fn
    except ImportError:
        from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits as aiter_fn
    batches = [int(a) for a in sys.argv[1:]] or [4, 8, 16, 26, 32]
    props = torch.cuda.get_device_properties(0)
    print(f"device {props.name} {props.gcnArchName}; ctx {CTX_MIN}-{CTX_MAX}, next_n {NEXT_N}, pools {NPOOL}, "
          f"iters {ITERS}", flush=True)
    cands = {"aiter": lambda inp, out: run_aiter(aiter_fn, inp, out),
             "aiter_flat": lambda inp, out: run_aiter_flat(aiter_fn, inp, out),
             "gluon": make_gluon(6, 16), "gluon_g4": make_gluon(4, 32)}
    print("batch (rows) | " + " | ".join(f"{n} us (max |diff| / max |ref|)" for n in cands)
          + " | per-row prep us | best gluon vs aiter_flat", flush=True)
    for batch in batches:
        inps = [make_inputs(batch, 1000 * batch + i) for i in range(NPOOL)]
        for inp in inps:
            inp["row_lens"], inp["row_table"] = per_row(inp)
        out = torch.full((batch * NEXT_N, MAX_MODEL_LEN), float("nan"), device=dev)
        run_aiter(aiter_fn, inps[0], out)
        ref = out.clone()
        cells, times = [], {}
        for name, fn in cands.items():
            try:
                out.fill_(float("nan"))
                fn(inps[0], out)
                torch.cuda.synchronize()
                d, m = visible_diff(ref, out, inps[0]["row_lens"])
                us = timed(fn, inps, out)
                times[name] = us
                cells.append(f"{us:.1f} ({d:.3g} / {m:.3g})")
            except Exception as e:  # report and keep going
                cells.append(f"error: {type(e).__name__}: {str(e)[:100]}")
        prep = timed(lambda inp, out: per_row(inp), inps, out)
        best = min(((t, n) for n, t in times.items() if n.startswith("gluon")), default=None)
        rel = f"{best[1]} {times['aiter_flat'] / best[0]:.2f}x" if best and "aiter_flat" in times else "-"
        print(f"{batch} ({batch * NEXT_N}) | " + " | ".join(cells) + f" | {prep:.1f} | {rel}", flush=True)


if __name__ == "__main__":
    main()
