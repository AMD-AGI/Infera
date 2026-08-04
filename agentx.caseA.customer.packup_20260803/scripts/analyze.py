#!/usr/bin/env python3
"""Recompute every ladder from aiperf's raw per-request records.

Same discipline as the par8 kit: nothing is copied from a summary line. Reads
profile_export.jsonl (one JSON object per completed request) and reports the
PROFILING phase only — warmup is excluded, exactly as the ramp phase is in par8.

Usage: analyze.py <art_dir> [<art_dir> ...]
"""
import json, sys, os, statistics as st


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def ladder(vals, scale=1.0):
    if not vals:
        return None
    return {
        "n": len(vals),
        "p50": pct(vals, .50) * scale,
        "p90": pct(vals, .90) * scale,
        "p99": pct(vals, .99) * scale,
        "max": max(vals) * scale,
        "mean": st.mean(vals) * scale,
    }


def fmt(d, unit="", dec=0):
    if d is None:
        return "n/a"
    f = f"{{:,.{dec}f}}"
    return (f"p50 {f.format(d['p50'])}{unit} | p90 {f.format(d['p90'])}{unit} | "
            f"p99 {f.format(d['p99'])}{unit} | max {f.format(d['max'])}{unit} | "
            f"mean {f.format(d['mean'])}{unit}")


def load(art):
    p = os.path.join(art, "profile_export.jsonl")
    recs = [json.loads(l) for l in open(p) if l.strip()]
    return recs


def analyze(art):
    recs = load(art)
    prof = [r for r in recs
            if r["metadata"].get("benchmark_phase") == "profiling"
            and not r["metadata"].get("was_cancelled")]
    warm = [r for r in recs if r["metadata"].get("benchmark_phase") == "warmup"]

    def g(rs, k):
        return [r["metrics"][k]["value"] for r in rs if k in r.get("metrics", {})]

    ttft = g(prof, "time_to_first_token")
    e2e = g(prof, "request_latency")
    itl = g(prof, "inter_token_latency")
    isl = g(prof, "input_sequence_length")
    osl = g(prof, "output_sequence_length")
    cache_read = g(prof, "usage_prompt_cache_read_tokens")
    prompt_tok = g(prof, "usage_prompt_tokens")

    # measurement window: first credit issue -> last request end, profiling only
    starts = [r["metadata"]["request_start_ns"] for r in prof
              if r["metadata"].get("request_start_ns")]
    ends = [r["metadata"]["request_end_ns"] for r in prof
            if r["metadata"].get("request_end_ns")]
    dur = (max(ends) - min(starts)) / 1e9 if starts and ends else None

    out = {
        "art": art,
        "n_profiling": len(prof),
        "n_warmup": len(warm),
        "n_cancelled": sum(1 for r in recs if r["metadata"].get("was_cancelled")),
        "n_ctx_overflow_skip": sum(1 for r in recs
                                   if r["metadata"].get("context_overflow_skip")),
        "window_s": dur,
        "req_per_s": len(prof) / dur if dur else None,
        "out_tok_per_s": sum(osl) / dur if dur and osl else None,
        "ttft_ms": ladder(ttft),
        "e2e_ms": ladder(e2e),
        "itl_ms": ladder(itl),
        "isl_tok": ladder(isl),
        "osl_tok": ladder(osl),
    }
    if cache_read and prompt_tok:
        out["server_cache_pct"] = 100.0 * sum(cache_read) / sum(prompt_tok)
        out["n_cache_reported"] = len(cache_read)
    return out


def ttft_by_isl(art):
    recs = load(art)
    prof = [r for r in recs
            if r["metadata"].get("benchmark_phase") == "profiling"
            and not r["metadata"].get("was_cancelled")]
    buckets = [(0, 50_000), (50_000, 100_000), (100_000, 160_000),
               (160_000, 220_000), (220_000, 300_000)]
    rows = []
    for lo, hi in buckets:
        v = [r["metrics"]["time_to_first_token"]["value"] for r in prof
             if "time_to_first_token" in r["metrics"]
             and lo <= r["metrics"].get("input_sequence_length", {}).get("value", -1) < hi]
        rows.append((f"{lo//1000}-{hi//1000}K", len(v),
                     pct(v, .5) if v else None, pct(v, .9) if v else None))
    return rows


if __name__ == "__main__":
    for art in sys.argv[1:]:
        r = analyze(art)
        print(f"\n{'='*66}\n{r['art']}\n{'='*66}")
        print(f"profiling requests : {r['n_profiling']}  "
              f"(warmup {r['n_warmup']}, cancelled {r['n_cancelled']}, "
              f"ctx-overflow-skip {r['n_ctx_overflow_skip']})")
        if r["window_s"]:
            print(f"window             : {r['window_s']:.1f} s   "
                  f"{r['req_per_s']:.3f} req/s   "
                  f"{r['out_tok_per_s']:.1f} out tok/s")
        print(f"TTFT  (ms)         : {fmt(r['ttft_ms'])}")
        print(f"E2E   (ms)         : {fmt(r['e2e_ms'])}")
        print(f"ITL   (ms)         : {fmt(r['itl_ms'], dec=2)}")
        print(f"ISL   (tok)        : {fmt(r['isl_tok'])}")
        print(f"OSL   (tok)        : {fmt(r['osl_tok'])}")
        if "server_cache_pct" in r:
            print(f"server cache hit   : {r['server_cache_pct']:.2f} %  "
                  f"(from usage_prompt_cache_read_tokens, n={r['n_cache_reported']})")
        print("\nTTFT by input size:")
        print(f"  {'bucket':<12}{'n':>6}{'p50 ms':>10}{'p90 ms':>10}")
        for name, n, p50, p90 in ttft_by_isl(art):
            print(f"  {name:<12}{n:>6}"
                  f"{('%.0f' % p50) if p50 else '-':>10}"
                  f"{('%.0f' % p90) if p90 else '-':>10}")
