#!/usr/bin/env python3
"""Split arm C's extra decode step time into a traffic-independent part and a
part that grows with HiCache traffic, using arm B as the equal-load baseline.

Input: per-second, per-DP-rank CSVs from server_metrics_extract.py.
step_ms = num_running_reqs * spec_accept_length / gen_throughput * 1000.
Residual = step_ms minus B's median step_ms in the same (running, context) bin.
Node traffic for a second = sum over ranks within +-1 s, because DP ranks step in
lockstep and a copy on one rank can slow the others.

Usage: hicache_step_overhead.py B_CSV C_CSV
"""
import collections
import csv
import statistics as st
import sys


def load(path):
    rows = {}
    for r in csv.DictReader(open(path)):
        f = lambda k: float(r[k]) if r[k] else 0.0
        rows[(int(r["dp"]), int(r["sec"]))] = dict(
            run=f("num_running_reqs"), ctx=f("decode_sum_seq_lens"), gen=f("gen_throughput"),
            acc=f("spec_accept_length"), backup=f("hicache_backup_tokens"),
            load=f("load_back_tokens"), evicted=f("evicted_tokens"))
    return rows


def key(r):
    return min(int((r["run"] - 1) // 2), 5), min(int(r["ctx"] // 0.5e6), 3)


def step(r):
    return 1000 * r["run"] * r["acc"] / r["gen"]


def valid(r):
    return r["run"] > 0 and r["gen"] > 0 and r["acc"] > 0


def baseline(rows):
    d = collections.defaultdict(list)
    for r in rows.values():
        if valid(r):
            d[key(r)].append(step(r))
    return {k: st.median(v) for k, v in d.items() if len(v) >= 40}


def residuals(rows, base, fields):
    node = collections.defaultdict(float)
    for (_, sec), r in rows.items():
        for k in (-1, 0, 1):
            node[sec + k] += sum(r[f] for f in fields)
    return [(step(r) - base[key(r)], node[sec]) for (_, sec), r in rows.items()
            if valid(r) and key(r) in base]


def quartiles(name, res):
    res.sort(key=lambda x: x[1])
    k = len(res) // 4
    print(f"{name}  (n={len(res)})")
    for i in range(4):
        q = res[i * k:(i + 1) * k] if i < 3 else res[3 * k:]
        print(f"  Q{i + 1}: traffic {q[0][1] / 1e3:7.0f}k-{q[-1][1] / 1e3:7.0f}k tokens  "
              f"residual median {st.median(x[0] for x in q):+.2f} ms  mean {st.mean(x[0] for x in q):+.2f} ms")


def main():
    b, c = load(sys.argv[1]), load(sys.argv[2])
    tot = lambda rows, f: sum(r[f] for r in rows.values()) / 1e6
    print(f"profiling-window totals: C backup {tot(c, 'backup'):.1f}M, load-back {tot(c, 'load'):.1f}M, "
          f"evicted {tot(c, 'evicted'):.1f}M tokens; B evicted {tot(b, 'evicted'):.1f}M tokens")
    quiet = {sec for (_, sec) in c} - {sec + k for (_, sec), r in c.items()
                                        if r["backup"] or r["load"] or r["evicted"] for k in range(-2, 3)}
    print(f"C seconds with no backup, load-back or eviction on any rank within +-2 s: "
          f"{len(quiet)} of {len({s for _, s in c})}")
    base = baseline(b)
    quartiles("B vs its own baseline, by node evicted tokens (request churn proxy)",
              residuals(b, base, ["evicted"]))
    quartiles("C vs B baseline, by node evicted tokens", residuals(c, base, ["evicted"]))
    quartiles("C vs B baseline, by node backup + load-back tokens", residuals(c, base, ["backup", "load"]))


if __name__ == "__main__":
    main()
