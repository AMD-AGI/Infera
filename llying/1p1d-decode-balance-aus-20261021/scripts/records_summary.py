#!/usr/bin/env python3
"""Client metrics of an AgentX run from AIPerf's per-request records (aiperf_artifacts/profile_export.jsonl), for runs
cut before AIPerf wrote its summary (the records are appended as requests finish).

Profiling-phase, non-cancelled requests only. Interactivity follows InferenceX: 1 / percentile of the per-request
full-response inter-token latency. Throughput counts input + output tokens of the requests that finished inside the
sending window [first request start, first start + DURATION], per GPU.
Usage: records_summary.py PROFILE_EXPORT_JSONL [DURATION_S=1800] [GPUS=12]
"""

import json
import statistics
import sys


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main() -> int:
    path = sys.argv[1]
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 1800.0
    gpus = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    recs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            md, m = d["metadata"], d["metrics"]
            if md.get("benchmark_phase") != "profiling" or md.get("was_cancelled"):
                continue
            recs.append((md["request_start_ns"], md["request_end_ns"], m))
    if not recs:
        print("no profiling records")
        return 1
    t0 = min(r[0] for r in recs)
    t_end = t0 + duration * 1e9
    done = [r for r in recs if r[1] <= t_end]
    val = lambda m, k: (m.get(k) or {}).get("value")  # noqa: E731
    itl = [v for _, _, m in done if (v := val(m, "full_response_inter_token_latency"))]
    ttft = [v / 1000 for _, _, m in done if (v := val(m, "time_to_first_token")) is not None]
    tokens = sum((val(m, "input_sequence_length") or 0) + (val(m, "output_sequence_length") or 0) for _, _, m in done)
    out_tokens = sum(val(m, "output_sequence_length") or 0 for _, _, m in done)
    covered = (max(r[1] for r in done) - t0) / 1e9
    # A run cut inside the window (preemption) is measured over the part it covered.
    span = min(duration, covered)
    print(f"requests {len(done)} in {covered:.0f} s of a {duration:.0f} s window ({len(recs)} profiling records)"
          f"{'  PARTIAL' if covered < duration - 60 else ''}")
    print(f"throughput {tokens / span / gpus:,.0f} tok/s/GPU  output {out_tokens / span / gpus:,.1f}  "
          f"rps {len(done) / span:.3f}")
    print(f"interactivity p50 / p90 {1000 / pct(itl, 0.5):.1f} / {1000 / pct(itl, 0.9):.1f}  "
          f"ITL p50 / p90 {pct(itl, 0.5):.1f} / {pct(itl, 0.9):.1f} ms")
    print(f"TTFT p50 / p90 / mean {pct(ttft, 0.5):.2f} / {pct(ttft, 0.9):.2f} / {statistics.mean(ttft):.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
