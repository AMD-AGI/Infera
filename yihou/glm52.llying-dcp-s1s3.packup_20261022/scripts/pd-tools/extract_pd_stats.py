#!/usr/bin/env python3
"""Extract SGLang PD-disaggregation scheduler queue depths for bottleneck analysis.

Parses prefill/decode server logs, maps SGLang's per-rank scheduler counters onto
the five pipeline states, filters to steady state, and reports mean/median/p95/max
per state. Also aggregates mooncake transfer-engine throughput (MC_TE_METRIC=1) and
scans for transfer-layer error signals.

The SGLang server_args line contains the same counter substrings and would pollute
a naive grep; this parser only accepts real "Prefill batch,"/"Decode batch," lines.

Usage:
  python3 extract_pd_stats.py \
      --prefill p.log [p2.log ...] \
      --decode  d.log [d2.log ...] \
      [--rank DP0] [--label 1P1D] [--run-min 15] [--json out.json]

Counters -> states (see sglang/SKILL.md):
  prefill: #queue-req(1 input) #bootstrap-req(handshake) #inflight-req(2 outbound/KV-send)
  decode:  #prealloc-req(3 admission) #transfer-req(4 KV-in) #running-req(5 running)
           #retracted-req(mem pressure) #queue-req(decode input)
"""
import argparse
import re
import statistics as st
import json
import sys

PREFILL_KEYS = ["queue-req", "bootstrap-req", "inflight-req"]
DECODE_KEYS = ["running-req", "transfer-req", "prealloc-req", "queue-req", "retracted-req"]

STATE_LABEL = {
    ("prefill", "queue-req"): "1 prefill-input",
    ("prefill", "bootstrap-req"): "  handshake",
    ("prefill", "inflight-req"): "2 prefill-outbound(KV-send)",
    ("decode", "prealloc-req"): "3 decode-admission",
    ("decode", "transfer-req"): "4 decode-transfer-in",
    ("decode", "running-req"): "5 decode-running",
    ("decode", "retracted-req"): "  mem-retracted",
    ("decode", "queue-req"): "  decode-input",
}


def parse_side(paths, side, rank, run_min):
    keys = PREFILL_KEYS if side == "prefill" else DECODE_KEYS
    marker = "Prefill batch," if side == "prefill" else "Decode batch,"
    stats = {k: [] for k in keys}
    lines_seen = 0
    for path in paths:
        try:
            data = open(path, errors="ignore").read().splitlines()
        except FileNotFoundError:
            print(f"WARN: {path} not found", file=sys.stderr)
            continue
        for line in data:
            if marker not in line:
                continue
            if rank and f"{rank} " not in line:
                continue
            row = {}
            for k in keys:
                m = re.search(rf"#{k}: (\d+)", line)
                if m:
                    row[k] = int(m.group(1))
            if not row:
                continue
            # steady-state filter: drop warmup/drain
            if side == "decode" and row.get("running-req", 0) < run_min:
                continue
            if side == "prefill" and (row.get("queue-req", 0) + row.get("inflight-req", 0)) == 0:
                continue
            lines_seen += 1
            for k, v in row.items():
                stats[k].append(v)
    return stats, lines_seen


def pctl(v, p):
    if not v:
        return 0
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p))]


def summarize(stats, side):
    out = {}
    for k, v in stats.items():
        if not v:
            continue
        out[k] = {
            "state": STATE_LABEL.get((side, k), k),
            "mean": round(st.mean(v), 1),
            "median": st.median(v),
            "p95": pctl(v, 0.95),
            "max": max(v),
            "n": len(v),
        }
    return out


def parse_max_running(paths):
    """Return the DP-adjusted effective max_running_requests (2nd occurrence)."""
    vals = []
    for path in paths:
        try:
            data = open(path, errors="ignore").read()
        except FileNotFoundError:
            continue
        vals += [int(x) for x in re.findall(r"max_running_requests=(\d+)", data)]
    # server_args prints passed value then adjusted; the smaller later one is per-rank
    return vals[-1] if vals else None


def parse_mooncake_bw(paths):
    """Aggregate mooncake transfer throughput: sum per-thread MB/s within each
    timestamp-second window, return peak/median aggregate in Gb/s."""
    windows = {}
    disabled = False
    for path in paths:
        try:
            data = open(path, errors="ignore").read().splitlines()
        except FileNotFoundError:
            continue
        for line in data:
            if "Metrics reporting is disabled" in line:
                disabled = True
            m = re.search(r"Transfer Engine Throughput: ([\d.]+) MB/s", line)
            ts = re.search(r"(\d\d:\d\d:\d\d)", line)
            if m and ts:
                windows.setdefault(ts.group(1), []).append(float(m.group(1)))
    if not windows:
        return {"available": False, "disabled_note": disabled}
    agg = [sum(v) for v in windows.values()]  # MB/s aggregate per second
    return {
        "available": True,
        "peak_gbps": round(max(agg) * 8 / 1000, 2),
        "median_gbps": round(st.median(agg) * 8 / 1000, 2),
        "mean_gbps": round(st.mean(agg) * 8 / 1000, 2),
        "windows": len(agg),
    }


ERROR_SIGNALS = [
    "Outstanding work requests",
    "Failed to poll completion",
    "Slice timeout detected",
    "ASIO exception",
    "Process failed for slice",
    "invalid_slice_size",
]


def scan_errors(paths):
    counts = {s: 0 for s in ERROR_SIGNALS}
    for path in paths:
        try:
            data = open(path, errors="ignore").read()
        except FileNotFoundError:
            continue
        for s in ERROR_SIGNALS:
            counts[s] += data.count(s)
    return {k: v for k, v in counts.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", nargs="+", default=[], help="prefill server log(s)")
    ap.add_argument("--decode", nargs="+", default=[], help="decode server log(s)")
    ap.add_argument("--rank", default="DP0", help="which DP rank to read (default DP0)")
    ap.add_argument("--label", default="config", help="config name for the report")
    ap.add_argument("--run-min", type=int, default=15, help="min #running-req for decode steady state")
    ap.add_argument("--json", help="also write raw stats to this JSON path")
    args = ap.parse_args()

    result = {"label": args.label, "rank": args.rank}

    if args.prefill:
        ps, pn = parse_side(args.prefill, "prefill", args.rank, args.run_min)
        result["prefill"] = summarize(ps, "prefill")
        result["prefill_steady_lines"] = pn
    if args.decode:
        ds, dn = parse_side(args.decode, "decode", args.rank, args.run_min)
        result["decode"] = summarize(ds, "decode")
        result["decode_steady_lines"] = dn
        cap = parse_max_running(args.decode)
        if cap:
            result["decode_running_cap_per_rank"] = cap
            rr = result["decode"].get("running-req")
            if rr:
                result["decode_running_occupancy_pct"] = round(100 * rr["mean"] / cap, 1)

    all_logs = args.prefill + args.decode
    result["rdma"] = parse_mooncake_bw(all_logs)
    errs = scan_errors(all_logs)
    result["transfer_errors"] = errs if errs else "none"

    # ---- pretty print ----
    print(f"\n===== PD bottleneck stats: {args.label} (rank {args.rank}) =====")
    for side in ("prefill", "decode"):
        if side not in result:
            continue
        print(f"\n[{side}]  (steady-state lines: {result.get(side+'_steady_lines',0)})")
        print(f"  {'state':<30}{'mean':>8}{'median':>8}{'p95':>6}{'max':>6}")
        order = PREFILL_KEYS if side == "prefill" else DECODE_KEYS
        for k in order:
            if k in result[side]:
                s = result[side][k]
                print(f"  {s['state']:<30}{s['mean']:>8}{s['median']:>8}{s['p95']:>6}{s['max']:>6}")
    if result.get("decode_running_cap_per_rank"):
        print(f"\n  decode running cap/rank = {result['decode_running_cap_per_rank']}"
              f"  ->  occupancy = {result.get('decode_running_occupancy_pct','?')}%")
    r = result["rdma"]
    if r.get("available"):
        print(f"\n[rdma] mooncake aggregate: peak={r['peak_gbps']} Gb/s"
              f"  median={r['median_gbps']} Gb/s  ({r['windows']} windows)")
    else:
        print(f"\n[rdma] no mooncake metrics"
              + ("  (MC_TE_METRIC was disabled — rerun with MC_TE_METRIC=1 for BW)"
                 if r.get("disabled_note") else ""))
    print(f"[transfer errors] {result['transfer_errors']}")

    # ---- hint the verdict inputs ----
    print("\n--- verdict inputs (apply parent skill's rule) ---")
    p1 = result.get("prefill", {}).get("queue-req", {}).get("max", "?")
    p2 = result.get("prefill", {}).get("inflight-req", {}).get("max", "?")
    d3 = result.get("decode", {}).get("prealloc-req", {}).get("max", "?")
    occ = result.get("decode_running_occupancy_pct", "?")
    print(f"  state1 prefill-input max={p1} | state2 prefill-outbound max={p2} "
          f"| state3 decode-admission max={d3} | decode-running occupancy={occ}%")
    print("  rule: input deep + outbound shallow + decode headroom => PREFILL-bound")
    print("        outbound(state2) deep                          => TRANSFER-bound")
    print("        admission(state3) deep + decode near cap        => DECODE-bound")

    if args.json:
        json.dump(result, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
