#!/usr/bin/env python3
"""Collate the 8 bench_serving JSONs into one table, plus the per-point MTP
acceptance and kvd counters that the JSON itself cannot carry.

WHY THE ACCEPTANCE COMES FROM A SIDE-CHANNEL. bench_serving reads
`avg_spec_accept_length` from `<base_url>/server_info`
(sglang/benchmark/serving.py:1525). Our `--base-url` is the infera ROUTER, which
has no `/server_info`, so `accept_length` is `None` in every JSON. That is a
property of routing through the router, not a missing feature -- so the sweep
snapshots the DECODE leg's own `/server_info` after each point instead, and this
script joins the two.

Usage: extract_sweep.py <bench_dir> [out.md]
"""
import glob
import json
import os
import re
import sys

D = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None


def num(x, nd=1):
    return "-" if x is None else f"{x:,.{nd}f}"


rows = []
for f in sorted(glob.glob(os.path.join(D, "*_isl*_osl*_conc*.json"))):
    m = re.search(r"_(p\d+)_isl(\d+)_osl(\d+)_conc(\d+)\.json$", os.path.basename(f))
    if not m:
        continue
    pt, isl, osl, conc = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    with open(f) as fh:
        d = json.load(fh)

    acc = None
    si = os.path.join(D, f"serverinfo_{pt}_conc{conc}.json")
    if os.path.isfile(si):
        try:
            with open(si) as fh:
                sd = json.load(fh)
            if "decode" in sd:
                sd = sd["decode"][0]
            st = (sd.get("internal_states") or [{}])[0]
            acc = st.get("avg_spec_accept_length")
        except Exception:
            pass

    kvd = {}
    kf = os.path.join(D, f"kvd_{pt}_conc{conc}.json")
    if os.path.isfile(kf):
        try:
            with open(kf) as fh:
                kvd = json.load(fh)
        except Exception:
            pass

    rows.append(dict(
        point=pt, isl=isl, osl=osl, conc=conc,
        completed=d.get("completed"),
        dur=d.get("duration"),
        req_tps=d.get("request_throughput"),
        in_tok=d.get("total_input_tokens"),
        out_tok=d.get("total_output_tokens"),
        out_tps=d.get("output_throughput"),
        # sglang 0.5.15 names these `total_throughput` / `input_throughput`;
        # `total_token_throughput` (the printed label) is NOT a JSON key.
        tot_tps=d.get("total_throughput"),
        in_tps=d.get("input_throughput"),
        real_conc=d.get("concurrency"),
        ttft_p50=d.get("median_ttft_ms"), ttft_p90=d.get("p90_ttft_ms"),
        ttft_p99=d.get("p99_ttft_ms"),
        tpot_p50=d.get("median_tpot_ms"), tpot_p90=d.get("p90_tpot_ms"),
        tpot_p99=d.get("p99_tpot_ms"),
        itl_p50=d.get("median_itl_ms"), itl_p99=d.get("p99_itl_ms"),
        e2e_p50=d.get("median_e2e_latency_ms"), e2e_p90=d.get("p90_e2e_latency_ms"),
        e2e_p99=d.get("p99_e2e_latency_ms"), e2e_mean=d.get("mean_e2e_latency_ms"),
        acc=acc,
        kvd_gets=kvd.get("gets_total"), kvd_hits=kvd.get("hits_total"),
        kvd_sets=kvd.get("sets_total"), kvd_ev=kvd.get("evictions_total"),
    ))

rows.sort(key=lambda r: (r["point"], r["conc"]))
L = []
A = L.append
A("## bench_serving sweep — 8 points, one server\n")
A("| point | ISL | OSL | conc | done | dur(s) | req/s | in tok/s | out tok/s | tot tok/s | real conc |")
A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    A(f"| {r['point']} | {r['isl']:,} | {r['osl']:,} | {r['conc']} | {r['completed']} | "
      f"{num(r['dur'])} | {num(r['req_tps'],3)} | {num(r['in_tps'])} | {num(r['out_tps'])} | "
      f"{num(r['tot_tps'])} | {num(r['real_conc'],2)} |")

A("\n### Latency (ms)\n")
A("| point | conc | TTFT p50 | p90 | p99 | TPOT p50 | p90 | p99 | E2E p50 | p90 | p99 |")
A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    A(f"| {r['point']} | {r['conc']} | {num(r['ttft_p50'])} | {num(r['ttft_p90'])} | "
      f"{num(r['ttft_p99'])} | {num(r['tpot_p50'],2)} | {num(r['tpot_p90'],2)} | "
      f"{num(r['tpot_p99'],2)} | {num(r['e2e_p50'])} | {num(r['e2e_p90'])} | {num(r['e2e_p99'])} |")

A("\n### MTP acceptance and kvd, per point\n")
A("(acceptance from the DECODE leg's /server_info — the router has none, so the "
  "bench JSON's own `accept_length` is null by construction)\n")
A("| point | conc | avg_spec_accept_length | kvd gets | hits | sets | evictions |")
A("|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    A(f"| {r['point']} | {r['conc']} | {num(r['acc'],3)} | {num(r['kvd_gets'],0)} | "
      f"{num(r['kvd_hits'],0)} | {num(r['kvd_sets'],0)} | {num(r['kvd_ev'],0)} |")

txt = "\n".join(L)
print(txt)
if OUT:
    with open(OUT, "w") as fh:
        fh.write(txt + "\n")
    print(f"\n[written] {OUT}", file=sys.stderr)
