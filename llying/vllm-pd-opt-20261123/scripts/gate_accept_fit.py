#!/usr/bin/env python3
"""Real acceptance of gates without the runaway completions.
  gate_accept_fit.py GROUP=TAG[,TAG...] [GROUP=...]      (TAG = gate tag, $R/results/gate-p4d4-<TAG>/)
The gate's accept_len is gen_tokens / drafts over everything it generated. A completion that runs to max_tokens
(16,383 on the probe set, a repetitive loop) is drafted almost perfectly and is ~10% of all tokens, and which probes run
away differs from run to run (decoding is not bitwise reproducible across runs even at temperature 0), so one runaway
more or less moves accept_len by ~1%. Fit drafts = other_tokens / a_group + capped_tokens / a_cap over all gates
(least squares; one a per group, one common a_cap) and print a per group and per gate.
"""
import json
import os
import re
import sys

R = "/home/liyingli/bench_agentx/vllm-pd-20261117"


def solve(A, b):
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][-1] / M[i][i] for i in range(n)]


def main():
    groups = [(a.split("=", 1)[0], a.split("=", 1)[1].split(",")) for a in sys.argv[1:]]
    summ = {}
    for line in open(f"{R}/logs/summary.out", errors="replace"):
        m = re.search(r" gate p4d4-(\S+): .*?metrics (\{.*?\});", line)
        if m:
            summ[m.group(1)] = json.loads(m.group(2))
    rows = []
    for g, tags in groups:
        for t in tags:
            cap = non = 0
            for f in ("probe", "longctx", "gsm8k"):
                p = f"{R}/results/gate-p4d4-{t}/{f}.jsonl"
                if os.path.exists(p):
                    for line in open(p):
                        r = json.loads(line)
                        c = r.get("completion_tokens") or 0
                        cap, non = (cap + c, non) if r.get("finish_reason") == "length" else (cap, non + c)
            d = summ[t]
            rows.append((g, t, cap, d["gen_tokens"] - cap, d["drafts"], d["accept_len"]))
    names = [g for g, _ in groups] + ["capped"]
    n = len(names)
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for g, _, cap, non, dr, _ in rows:
        v = [0.0] * n
        v[names.index(g)], v[-1] = non, cap
        for i in range(n):
            b[i] += v[i] * dr
            for j in range(n):
                A[i][j] += v[i] * v[j]
    x = solve(A, b)
    for nm, xi in zip(names, x):
        print(f"{nm:12s} acceptance {1 / xi:.3f}")
    for g, t, cap, non, dr, acc in rows:
        fit = x[names.index(g)] * non + x[-1] * cap
        print(f"  {g:12s} {t[:44]:44s} capped {cap:6d} other {non:6d} drafts {dr:6d} (fit {fit:6.0f}) "
              f"accept_len {acc:.3f}, without capped {non / (dr - x[-1] * cap):.3f}")


if __name__ == "__main__":
    main()
