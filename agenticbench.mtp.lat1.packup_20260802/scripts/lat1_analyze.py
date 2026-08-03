#!/usr/bin/env python3
"""Analyze a lat1 (concurrency-1) run.

Two things this prints that a percentile ladder alone cannot give:

  1. Ladders WITH their sample count stated, so a p99 computed from ~190
     samples is never mistaken for a p99 computed from 2,811.
  2. TTFT-vs-input and TPOT-vs-output BINNED CURVES. At concurrency 1 there is
     no queueing, so TTFT is a function of prompt length and TPOT is a function
     of batch-of-one decode -- the mixed-distribution percentiles blur exactly
     the structure the experiment exists to expose.

TPOT caveat, load-bearing: the driver persists only p50/p90/p99/mean of TPOT in
summary.json and NO per-request array in metrics.jsonl (verified: the only
generation-side keys are `generation_tps` and `new_generation_lengths`). So the
per-request TPOT here is RECONSTRUCTED as a lower bound from what is persisted:
  gen_tokens / generation_tps  -> generation_time -> /(gen_tokens-1) -> tpot
and `generation_tps` is a WINDOW AVERAGE, not per-request. The reconstruction is
therefore only used for the CURVE SHAPE; every headline TPOT number comes from
summary.json's own tpot_ms block.

Usage: lat1_analyze.py <run_dir> [--json out.json]
"""
import sys, json, math, gzip, os


def P(a, p):
    if not a:
        return float("nan")
    a = sorted(a)
    k = (len(a) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (k - lo)


def opener(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load(run_dir):
    mj = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(mj):
        mj += ".gz"
    recs = [json.loads(l) for l in opener(mj)]
    summ = json.load(open(os.path.join(run_dir, "summary.json")))
    return recs, summ


LADDER = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def ladder(name, vals, unit="", scale=1.0):
    if not vals:
        print(f"\n{name}: NO SAMPLES")
        return
    print(f"\n{name}  (n={len(vals)})")
    row = "  " + "".join(f"{'p'+str(p):>9}" for p in LADDER) + f"{'mean':>9}{'max':>9}"
    print(row)
    print("  " + "".join(f"{P(vals,p)*scale:>9,.1f}" for p in LADDER)
          + f"{sum(vals)/len(vals)*scale:>9,.1f}{max(vals)*scale:>9,.1f}" + f"  {unit}")


def curve(name, xs, ys, edges, xunit, yunit, yscale=1.0):
    """Bin ys by xs and print mean/median per bin with counts."""
    print(f"\n{name}")
    print(f"  {'bin ('+xunit+')':>22}{'n':>5}{'mean '+yunit:>14}{'p50':>12}{'min':>12}{'max':>12}")
    print("  " + "-" * 77)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [y for x, y in zip(xs, ys) if lo <= x < hi]
        if not sel:
            continue
        lbl = f"{lo:,.0f}-{hi:,.0f}" if hi < float("inf") else f"{lo:,.0f}+"
        print(f"  {lbl:>22}{len(sel):>5}{sum(sel)/len(sel)*yscale:>14,.1f}"
              f"{P(sel,50)*yscale:>12,.1f}{min(sel)*yscale:>12,.1f}{max(sel)*yscale:>12,.1f}")


def main():
    run_dir = sys.argv[1]
    recs, summ = load(run_dir)

    ramp = summ.get("context", {}).get("ramp_duration_secs")
    if ramp is None:
        ramp = 180.0

    # Per-request arrays, in completion order (all four advance together).
    ttft, plen, glen, acc, chr_ = [], [], [], [], []
    inflight, sessions, gtps_t = [], [], []
    ttft_s, plen_s, glen_s, acc_s = [], [], [], []   # sustain only
    for r in recs:
        t = r.get("elapsed_seconds", 0.0)
        a = r.get("new_ttfts") or []
        b = r.get("new_prompt_lengths") or []
        c = r.get("new_generation_lengths") or []
        d = r.get("new_acceptance_lengths") or []
        e = r.get("new_cache_hit_rates") or []
        ttft += a; plen += b; glen += c; acc += d; chr_ += e
        if t >= ramp:
            ttft_s += a; plen_s += b; glen_s += c; acc_s += d
        inflight.append(r.get("in_flight", 0))
        sessions.append(r.get("num_sessions_active", 0))
        if r.get("generation_tps"):
            gtps_t.append((t, r["generation_tps"]))

    print("=" * 80)
    print(f"lat1 analysis — {run_dir}")
    print("=" * 80)
    print(f"duration        {summ.get('duration_s',0):,.1f} s   (ramp {ramp:.0f} s excluded from SUSTAIN)")
    print(f"requests        {summ.get('requests_sent')} sent / {summ.get('requests_completed')} completed"
          f" / {summ.get('errors')} errors   success={summ.get('success_rate',0):.4f}")
    print(f"qps             {summ.get('actual_average_qps',0):.4f}  (closed loop, one at a time)")
    print(f"\nCONCURRENCY GUARANTEE   in_flight max={max(inflight)}   sessions_active max={max(sessions)}")
    if max(inflight) > 1 or max(sessions) > 1:
        print("  *** VIOLATED — this is not a concurrency-1 measurement ***")
    else:
        print("  holds: never more than one request in flight, never more than one live session")

    print("\n" + "-" * 80)
    print("HEADLINE (from summary.json, the driver's own computation)")
    print("-" * 80)
    for k in ("ttft_ms", "tpot_ms"):
        v = summ.get(k, {})
        print(f"  {k:10} mean={v.get('mean',0):>9,.1f}  p50={v.get('p50',0):>9,.1f}"
              f"  p90={v.get('p90',0):>9,.1f}  p99={v.get('p99',0):>9,.1f}")
    ca = summ.get("cache", {})
    print(f"  cache      actual={ca.get('actual_hit_rate',0):.4f}  ideal={ca.get('ideal_hit_rate',0):.4f}"
          f"  eff={ca.get('efficiency',0):.4f}  evict={ca.get('eviction_rate',0):.4f}")

    print("\n" + "-" * 80)
    print("LADDERS — ALL requests")
    print("-" * 80)
    ladder("TTFT", ttft, "ms", 1000.0)
    ladder("prompt length", plen, "tok")
    ladder("generation length", glen, "tok")
    ladder("acceptance (driver-side)", acc, "")
    ladder("cache hit", chr_, "%", 100.0)

    print("\n" + "-" * 80)
    print(f"LADDERS — SUSTAIN only (elapsed >= {ramp:.0f} s)")
    print("-" * 80)
    ladder("TTFT", ttft_s, "ms", 1000.0)
    ladder("prompt length", plen_s, "tok")
    ladder("generation length", glen_s, "tok")
    ladder("acceptance (driver-side)", acc_s, "")

    print("\n" + "-" * 80)
    print("THE CURVE THAT ANSWERS THE QUESTION")
    print("-" * 80)
    n = min(len(ttft), len(plen))
    curve("TTFT vs input length  (concurrency 1 -> pure prefill service time)",
          plen[:n], ttft[:n],
          [0, 40000, 60000, 80000, 100000, 130000, 160000, 200000, 240000, float("inf")],
          "input tok", "TTFT ms", 1000.0)
    if n >= 2:
        # Per-1K-token slope by least squares — the prefill rate implied.
        xs = plen[:n]; ys = [t * 1000 for t in ttft[:n]]
        mx = sum(xs) / n; my = sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            sl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
            ic = my - sl * mx
            print(f"\n  least-squares fit:  TTFT_ms = {ic:,.0f} + {sl*1000:,.2f} x (input_ktok)")
            print(f"  -> fixed overhead {ic:,.0f} ms;  marginal prefill rate "
                  f"{1000.0/sl if sl>0 else float('nan'):,.0f} tok/s")

    m = min(len(glen), len(ttft))
    print("\n  (TPOT-vs-output is not reconstructible per request from these artifacts —")
    print("   the driver persists no per-request TPOT array. summary.json's tpot_ms")
    print("   block above is the authoritative TPOT for this run.)")

    if len(sys.argv) > 3 and sys.argv[2] == "--json":
        out = {
            "run_dir": run_dir, "ramp_s": ramp,
            "n_all": len(ttft), "n_sustain": len(ttft_s),
            "in_flight_max": max(inflight), "sessions_max": max(sessions),
            "summary": summ,
            "ttft_ladder_all": {f"p{p}": P(ttft, p) * 1000 for p in LADDER},
            "ttft_ladder_sustain": {f"p{p}": P(ttft_s, p) * 1000 for p in LADDER},
            "prompt_ladder_sustain": {f"p{p}": P(plen_s, p) for p in LADDER},
            "gen_ladder_sustain": {f"p{p}": P(glen_s, p) for p in LADDER},
            "acc_ladder_sustain": {f"p{p}": P(acc_s, p) for p in LADDER},
        }
        json.dump(out, open(sys.argv[3], "w"), indent=2)
        print(f"\n[json] wrote {sys.argv[3]}")


if __name__ == "__main__":
    main()
