#!/usr/bin/env python3
"""Recompute every solo-run ladder from the raw per-request arrays.

Reads metrics.jsonl(.gz) and emits the tables the analysis is written from.
Nothing here reads summary.json -- the point is that every number is
reproducible from the raw samples.

  python3 solo_analyze.py <metrics.jsonl[.gz]> [--phase sustain]
"""
import sys, json, gzip, math, argparse

PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def P(a, p):
    """Linear-interpolated percentile. Matches the Case A analysis method."""
    if not a:
        return float("nan")
    a = sorted(a)
    k = (len(a) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (k - lo)


def ladder(name, vals, scale=1.0, unit="", fmt="{:,.0f}"):
    if not vals:
        print(f"\n## {name}: NO SAMPLES")
        return
    v = [x * scale for x in vals]
    print(f"\n## {name}  (n={len(v)}{unit})")
    print("| stat | value |")
    print("|---|---|")
    print(f"| min | {fmt.format(min(v))} |")
    for p in PCTS:
        star = "**" if p in (50, 90, 99) else ""
        print(f"| {star}p{p}{star} | {star}{fmt.format(P(v, p))}{star} |")
    print(f"| max | {fmt.format(max(v))} |")
    print(f"| mean | {fmt.format(sum(v) / len(v))} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--phase", default="sustain",
                    help="sustain | ramp | drain | all")
    a = ap.parse_args()

    op = gzip.open if a.path.endswith(".gz") else open
    F = {k: [] for k in ("ttft", "e2e", "tpot", "inp", "gen", "acc",
                         "chr", "ichr", "ia")}
    inflight, nsess, rows = [], [], 0
    for line in op(a.path, "rt"):
        r = json.loads(line)
        if a.phase != "all" and r.get("phase") != a.phase:
            continue
        rows += 1
        F["ttft"] += r.get("new_ttfts") or []
        F["e2e"] += r.get("new_e2es") or []
        F["tpot"] += r.get("new_tpots") or []
        F["inp"] += r.get("new_prompt_lengths") or []
        F["gen"] += r.get("new_generation_lengths") or []
        F["acc"] += r.get("new_acceptance_lengths") or []
        F["chr"] += r.get("new_cache_hit_rates") or []
        F["ichr"] += r.get("new_ideal_cache_hit_rates") or []
        F["ia"] += r.get("new_inter_arrival_times") or []
        if r.get("in_flight") is not None:
            inflight.append(r["in_flight"])
        if r.get("num_sessions_active") is not None:
            nsess.append(r["num_sessions_active"])

    n = len(F["ttft"])
    print(f"# SOLO ladders — phase={a.phase}, {rows} rows, {n} requests")
    if not n:
        sys.exit("no samples in this phase")

    # Alignment is the invariant the SOLO_M1 patch exists to guarantee.
    bad = {k: len(F[k]) for k in ("e2e", "tpot", "inp", "gen") if len(F[k]) != n}
    print(f"\nalignment ttft={n} e2e={len(F['e2e'])} tpot={len(F['tpot'])} "
          f"input={len(F['inp'])} gen={len(F['gen'])}"
          + (f"  ** MISALIGNED {bad} **" if bad else "  OK"))

    print(f"\nin_flight distinct={sorted(set(inflight))} max={max(inflight) if inflight else '-'}"
          f"   active_sessions distinct={sorted(set(nsess))}")

    ladder("TTFT (ms)", F["ttft"], 1000)
    ladder("E2E (ms)", F["e2e"], 1000)
    tp = [x for x in F["tpot"] if x > 0]
    ladder("TPOT (ms)", tp, 1000, f", {len(F['tpot']) - len(tp)} filtered", "{:,.2f}")
    ladder("input tokens", F["inp"])
    ladder("output tokens", F["gen"])
    ladder("MTP acceptance length", F["acc"], 1.0, "", "{:.3f}")
    ladder("cache hit rate", F["chr"], 100.0, " %", "{:.2f}")
    ladder("inter-arrival (s)", F["ia"], 1.0, "", "{:.2f}")

    # Consistency: does TTFT + (gen-1) x TPOT reproduce the measured E2E?
    # In Case A this had to BE the E2E; here it is a cross-check on a measured one.
    print("\n## E2E composition cross-check")
    resid = []
    for i in range(n):
        if F["tpot"][i] > 0 and F["gen"][i] > 1:
            pred = F["ttft"][i] + (F["gen"][i] - 1) * F["tpot"][i]
            resid.append((pred - F["e2e"][i]) * 1000)
    if resid:
        print(f"predicted - measured (ms): n={len(resid)} "
              f"mean={sum(resid)/len(resid):+.1f} p50={P(resid,50):+.1f} "
              f"p90={P(resid,90):+.1f} max={max(resid):+.1f}")

    # Duty cycle: how much of the window was actually spent serving.
    if F["e2e"]:
        busy = sum(F["e2e"])
        span = rows  # 1 Hz rows == seconds
        print(f"\n## Duty cycle\nbusy {busy:.0f}s / window {span}s = "
              f"{100*busy/span:.1f}%   (idle = the <=1s respawn tick)")


if __name__ == "__main__":
    main()
