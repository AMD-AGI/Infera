#!/usr/bin/env python3
"""Locate the bottleneck phase of an SGLang PD deployment from scheduler
queue-depth counters, and measure the client's ARRIVAL RATE alongside it.

Method (queueing theory, qualitative Little's Law): in a pipeline a queue grows
immediately upstream of the slowest stage and stays near-empty downstream of it.
So read WHICH queue is backed up, rather than measuring rates.

  state 1  prefill  #queue-req      requests waiting for prefill compute
  state 2  prefill  #inflight-req   KV computed, waiting for transfer to finish
  state 3  decode   #prealloc-req   arrived at decode, waiting for a KV seat
  state 4  decode   #transfer-req   seat reserved, KV arriving over the wire
  state 5  decode   #running-req    actively generating
           decode   #retracted-req  memory pressure

Verdict:
  prefill-bound  state 1 deep, state 2 shallow, decode running well under cap
  transfer-bound state 2 deep  <- the ONLY config where transfer is the limiter
  decode-bound   state 3 deep AND decode running at cap

TWO TRAPS this script exists to avoid:

1. PER-RANK vs MIXED. Under DP-attention each rank runs its own scheduler and
   prints its own counters. Averaging all ranks together yields a number that is
   not any queue's depth. This script splits by the `DPn` tag and reports each
   rank; it also reports the per-second cross-rank SUM, which is the only
   meaningful aggregate (a per-rank running of 1.0 across 8 ranks is 8 requests
   in flight, not an idle engine).

2. ARRIVAL RATE IS NOT A CONSTANT. Under a closed-loop driver (the AgentX
   agentic_replay strategy issues turn N+1 from turn N's return callback),
   slowing the server LOWERS the arrival rate, which drains the prefill queue
   and improves TTFT. Reading the queue table without the arrival rate will
   attribute that to the server getting faster. --jsonl computes it.

Usage:
  pd_bottleneck.py --prefill P.log --decode D.log [--label arm1]
                   [--jsonl profile_export.jsonl] [--decode-cap 256]
Logs may be plain or .gz. Engine logs contain binary bytes: read as latin-1.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics as st
from collections import defaultdict

FIELDS_P = ["queue-req", "inflight-req"]
FIELDS_D = ["prealloc-req", "transfer-req", "running-req", "retracted-req"]
STATE_NAME = {
    "queue-req": "1 prefill input",
    "inflight-req": "2 prefill outbound",
    "prealloc-req": "3 decode admission",
    "transfer-req": "4 decode transfer-in",
    "running-req": "5 decode running",
    "retracted-req": "- decode retracted",
}


def read(path: str) -> str:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as f:
        return f.read().decode("latin-1", "replace")


def q(vals, frac):
    s = sorted(vals)
    return s[min(int(len(s) * frac), len(s) - 1)]


def summarize(vals):
    if not vals:
        return None
    return dict(n=len(vals), mean=st.mean(vals), p95=q(vals, 0.95), mx=max(vals))


def scan(text: str, kind: str):
    """Return (per_rank, per_second_sum). kind is 'Prefill' or 'Decode'."""
    fields = FIELDS_P if kind == "Prefill" else FIELDS_D
    per_rank = defaultdict(lambda: defaultdict(list))
    per_sec = defaultdict(lambda: defaultdict(int))
    sec_ranks = defaultdict(set)
    for line in text.splitlines():
        if f"{kind} batch" not in line:
            continue
        # DP tag is absent when DP-attention is off -> single global scheduler.
        m = re.search(r" (DP\d+) TP", line)
        rank = m.group(1) if m else "(single)"
        ts = re.match(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", line)
        sec = ts.group(1) if ts else None
        if sec:
            sec_ranks[sec].add(rank)
        for f in fields:
            t = re.search(rf"#{f}: (\d+)", line)
            if not t:
                continue
            v = int(t.group(1))
            per_rank[rank][f].append(v)
            if sec:
                per_sec[sec][f] += v
        tu = re.search(r"token usage: ([\d.]+)", line)
        if tu:
            per_rank[rank]["tokusg"].append(float(tu.group(1)))
    return per_rank, per_sec, sec_ranks


def arrival(jsonl: str):
    ts = []
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            md = r.get("metadata", {})
            if md.get("benchmark_phase") != "profiling":
                continue
            s = md.get("request_start_ns")
            if s:
                ts.append(s)
    if len(ts) < 2:
        return None
    ts.sort()
    span = (ts[-1] - ts[0]) / 1e9
    gaps = [(b - a) / 1e9 for a, b in zip(ts, ts[1:])]
    return dict(n=len(ts), span=span, rate=len(ts) / span,
                gap_p50=q(gaps, 0.5), gap_mean=st.mean(gaps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", required=True)
    ap.add_argument("--decode", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--jsonl", help="aiperf profile_export.jsonl for arrival rate")
    ap.add_argument("--decode-cap", type=int, default=None,
                    help="per-rank max_running_requests (the DP-ADJUSTED one, "
                         "i.e. the 2nd value printed in server_args)")
    a = ap.parse_args()

    ptext, dtext = read(a.prefill), read(a.decode)
    prank, _, _ = scan(ptext, "Prefill")
    drank, dsec, dsec_ranks = scan(dtext, "Decode")

    print(f"===== {a.label} =====\n")
    print("--- prefill (states 1-2) ---")
    print(f"{'rank':<10}{'lines':>7}{'state1 mean':>13}{'p95':>6}{'max':>6}"
          f"{'state2 mean':>13}{'p95':>6}{'max':>6}")
    for r in sorted(prank):
        d = prank[r]
        s1, s2 = summarize(d.get("queue-req", [])), summarize(d.get("inflight-req", []))
        if not s1:
            continue
        print(f"{r:<10}{s1['n']:>7}{s1['mean']:>13.2f}{s1['p95']:>6}{s1['mx']:>6}"
              f"{s2['mean']:>13.2f}{s2['p95']:>6}{s2['mx']:>6}")

    print("\n--- decode (states 3-5), PER RANK ---")
    print(f"{'rank':<10}{'lines':>7}{'run mean':>10}{'run max':>9}"
          f"{'prealloc max':>14}{'xfer mean':>11}{'xfer max':>9}"
          f"{'retract max':>13}{'tokusg p95':>12}")
    for r in sorted(drank, key=lambda x: (x != "(single)", x)):
        d = drank[r]
        run = d.get("running-req", [])
        if not run:
            continue
        tu = d.get("tokusg", [0.0])
        print(f"{r:<10}{len(run):>7}{st.mean(run):>10.2f}{max(run):>9}"
              f"{max(d.get('prealloc-req', [0])):>14}"
              f"{st.mean(d.get('transfer-req', [0])):>11.3f}"
              f"{max(d.get('transfer-req', [0])):>9}"
              f"{max(d.get('retracted-req', [0])):>13}{q(tu, 0.95):>12.3f}")

    if dsec:
        tot = [v.get("running-req", 0) for v in dsec.values()]
        nr = [len(s) for s in dsec_ranks.values()]
        print(f"\ncross-rank SUM of running-req per second: "
              f"p50={q(tot,.5)} p90={q(tot,.9)} max={max(tot)}   "
              f"| distinct ranks active/sec: p50={q(nr,.5)} max={max(nr)}")
        if a.decode_cap:
            nranks = len([r for r in drank if r != "(single)"]) or 1
            occ = 100 * st.mean(tot) / (a.decode_cap * nranks)
            print(f"decode occupancy vs cap ({a.decode_cap}/rank x {nranks} ranks): {occ:.2f}%")

    if a.jsonl:
        ar = arrival(a.jsonl)
        if ar:
            print(f"\n--- client ARRIVAL (closed-loop: this is an OUTPUT, not a knob) ---")
            print(f"n={ar['n']} span={ar['span']:.1f}s  arrival_rate={ar['rate']:.3f} req/s  "
                  f"gap p50={ar['gap_p50']:.2f}s mean={ar['gap_mean']:.2f}s")
            print("Compare this ACROSS arms before attributing any TTFT change to the server.")


if __name__ == "__main__":
    main()
