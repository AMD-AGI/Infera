#!/usr/bin/env python3
"""CPU simulation of the Triton decode top-k's sampled threshold (scripts/vpd_triton_topk_decode.py, _topk_est):
per-row probability that the full-row candidate count lands outside [k, CAND] (the select's slow whole-row path).

One call has ~88 rows and takes as long as its slowest row, so a per-row failure rate of 1-2% already makes most calls
slow (a14: select p50 4 ms). Synthetic rows with position correlation, which the bench distributions lacked:
  ar      AR(1) along the row, rho 0.995 (smooth relevance drift)
  seg     segments (mean 2,000 tokens) with a N(0,1) level + N(0,0.3) noise, negatives set to 0 (ReLU-like)
  hot     iid N(0,1) background, last 4,000 tokens + 8 random 1,000-token regions shifted up by 3
  iid     N(0,1) reference
Rules: (probes, probe length, OVER, CAND) for the sample of probes x length logits at evenly spaced spans.
  sim_topk_est.py [TRIALS=200] [L=700000]
"""
import sys

import numpy as np

K = 2048


def row(kind, L, rng):
    if kind == "iid":
        return rng.standard_normal(L).astype(np.float32)
    if kind == "ar":
        e = rng.standard_normal(L) * np.sqrt(1 - 0.995 ** 2)
        x = np.empty(L)
        x[0] = rng.standard_normal()
        # AR(1) by blocks (exact recursion is slow in Python): lfilter-free cumulative form
        a = 0.995
        idx = np.arange(L)
        w = a ** (-idx.astype(np.float64) % 50000)
        x = np.zeros(L)
        for s in range(0, L, 50000):
            seg = e[s:s + 50000]
            ww = a ** np.arange(len(seg))
            x[s:s + 50000] = ww * (np.cumsum(seg / ww) + (x[s - 1] if s else rng.standard_normal()))
        return x.astype(np.float32)
    if kind == "seg":
        x = np.empty(L, dtype=np.float32)
        p = 0
        while p < L:
            n = int(rng.exponential(2000)) + 1
            x[p:p + n] = rng.standard_normal() + 0.3 * rng.standard_normal(min(n, L - p))
            p += n
        return np.maximum(x, 0)
    if kind == "hot":
        x = rng.standard_normal(L).astype(np.float32)
        x[-4000:] += 3
        for s in rng.integers(0, L - 1000, 8):
            x[s:s + 1000] += 3
        return x
    raise ValueError(kind)


def est_count(x, probes, plen, over, rng):
    L = len(x)
    span = max(plen, L // probes)
    off = (np.arange(probes) * span)[:, None] + np.arange(plen)[None, :]
    smp = x[off[off < L]]
    ns = smp.size
    t = max(1, min(over * K * ns // L, ns))
    thr = np.partition(smp, ns - t)[ns - t]       # t-th largest sample (the kernel rounds it down a little)
    return int((x >= thr).sum())


def main():
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    L = int(sys.argv[2]) if len(sys.argv) > 2 else 700_000
    rules = [(64, 128, 2, 8192), (512, 16, 2, 8192), (512, 16, 3, 16384), (1024, 8, 3, 16384), (2048, 4, 3, 16384)]
    if len(sys.argv) > 3:  # e.g. "1024,8,3,16384;1024,8,2,12288"
        rules = [tuple(int(v) for v in r.split(",")) for r in sys.argv[3].split(";")]
    rng = np.random.default_rng(0)
    print(f"L {L}, k {K}, {trials} rows per kind; failure = full-row count < k or > CAND")
    for kind in ("iid", "ar", "seg", "hot"):
        rows = [row(kind, L, rng) for _ in range(trials)]
        cells = []
        for probes, plen, over, cand in rules:
            c = np.array([est_count(x, probes, plen, over, rng) for x in rows])
            under, overflow = (c < K).mean(), (c > cand).mean()
            q = np.quantile(c, [0.0, 0.001, 0.5, 0.999, 1.0]).astype(int)
            cells.append(f"{probes}x{plen} O{over} C{cand}: under {under:.4f} over {overflow:.4f} (min/p0.1/p50/p99.9/max {q.tolist()})")
        print(f"{kind:>4}: " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
