"""analyze.py : summarize results/k{K}_{set}_{mode}.jsonl into a table (results/summary.md)."""
import glob, json, os, statistics as st
D = "/shared_nfs/yihou_glm52/llrepro_20261023/acc.yihou/results"
def implied_p(al, K):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        m = (lo + hi) / 2
        (lo, hi) = (m, hi) if sum(m ** i for i in range(K + 1)) < al else (lo, m)
    return lo
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]
out = ["| K | set | mode | reqs | tokens | verify steps | accept len (tok/step) | hist mean (correct+1) | per-req p10/p50/p90 | implied const p | empirical step-wise accept P(d_i ok given d_{i-1} ok) |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(D + "/k*_*_*.jsonl")):
    K, s, m = os.path.basename(f)[:-6].split("_")
    K = int(K[1:])
    rs = [json.loads(l) for l in open(f)]
    rs = [r for r in rs if "error" not in r and r["verify"] > 0]
    tn = sum(r["completion_tokens"] for r in rs); tv = sum(r["verify"] for r in rs)
    H = [0] * (K + 1)
    for r in rs:
        for i, c in enumerate(r["hist"] or []):
            H[i] += c
    hs = sum(H); hm = sum((i + 1) * c for i, c in enumerate(H)) / max(hs, 1)
    ge = [sum(H[i:]) for i in range(K + 1)]
    cond = [ge[i] / ge[i - 1] if ge[i - 1] else 0 for i in range(1, K + 1)]
    acc = [r["acc"] for r in rs]
    al = tn / tv
    out.append(f"| {K} | {s} | {m} | {len(rs)} | {tn} | {tv} | {al:.3f} | {hm:.3f} | {pct(acc,.1):.2f}/{pct(acc,.5):.2f}/{pct(acc,.9):.2f} | "
               f"{implied_p(al, K):.3f} | {' '.join(f'{c:.2f}' for c in cond)} |")
open(D + "/summary.md", "w").write("\n".join(out) + "\n")
print("\n".join(out))
