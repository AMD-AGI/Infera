#!/usr/bin/env python3
"""The baseline protocol, re-runnable: **5 rounds, each of >= 10 timed iterations**.

The series task book asks for "5 次加权平均，每次运行 loop 10 次以上取平均" — five
rounds, each averaging over at least ten iterations. This is that, with two
deliberate departures, both argued rather than assumed:

**Each round is a fresh process.** A loop inside one process shares a warm
Triton JIT cache, a warm allocator and one set of clocks, so it measures
run-to-run jitter and not round-to-round reproducibility. Round-to-round is the
number that matters, because that is the comparison a speedup claim is made
across. `subprocess` per round is the only way to get it.

**The reported statistic is the median, not the mean.** Measured on this
machine 2026-08-31: the ATen baseline is tight (0.3% spread across five rounds)
while an optimized replacement was loose (~8%). A mean over five rounds of the
loose side is dragged by its outliers, and a single such outlier already
produced one wrong conclusion here — 21.67 us was read as a real regression when
the median was 18.9 us. The mean is printed too, so a reader can see the two
disagree when they do.

Usage
-----
    python3 measure_baseline.py                 # 5 rounds x 30 iters, all cases
    python3 measure_baseline.py --rounds 3      # cheaper, for a smoke test
    python3 measure_baseline.py --json out.json # machine-readable, same numbers

Nothing here imports the kernel. It shells out to `driver.py --bench-mode`, so
it measures exactly what the forge loop measures and cannot drift from it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DRIVER = _HERE / "driver.py"


def _round(warmup: int, iters: int) -> dict[str, float]:
    """One round in a fresh process. Returns {case_id: median_ms} as the driver reports it."""
    proc = subprocess.run(
        [sys.executable, str(_DRIVER), "--bench-mode", "--warmup", str(warmup), "--iters", str(iters)],
        capture_output=True,
        text=True,
        cwd=_HERE,
    )
    if proc.returncode != 0:
        raise SystemExit(f"driver.py exited {proc.returncode}\n--- stderr ---\n{proc.stderr}")
    out: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        # The driver's contract: `case_ms: <case_id> <median_ms>`. Parsed rather
        # than recomputed, so this script and the forge loop read one number.
        if line.startswith("case_ms:"):
            _, case_id, value = line.split()
            out[case_id] = float(value)
    if not out:
        raise SystemExit(f"driver.py printed no `case_ms:` line\n--- stdout ---\n{proc.stdout}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="the 5x10 baseline protocol")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30, help="timed iterations per round; the floor is 10")
    ap.add_argument("--json", metavar="PATH", help="also write the numbers here")
    args = ap.parse_args()

    if args.iters < 10:
        # The floor is the protocol's, not this script's, so it refuses rather
        # than silently reporting a number the protocol would not accept.
        print(f"error: --iters {args.iters} is below the protocol floor of 10", file=sys.stderr)
        return 2

    rounds = [_round(args.warmup, args.iters) for _ in range(args.rounds)]
    cases = sorted(rounds[0])

    report: dict[str, dict[str, float | list[float]]] = {}
    print(f"# {args.rounds} rounds x {args.iters} iters, fresh process each round\n")
    print(f"{'case':<16} {'median_ms':>10} {'mean_ms':>10} {'min_ms':>10} {'max_ms':>10} {'spread':>8}")
    for case in cases:
        samples = [r[case] for r in rounds if case in r]
        median, mean = statistics.median(samples), statistics.fmean(samples)
        lo, hi = min(samples), max(samples)
        # Spread relative to the median: how far apart the rounds are, which is
        # the number that says whether a speedup claim is safe from one sample.
        spread = (hi - lo) / median if median else 0.0
        report[case] = {
            "samples_ms": samples,
            "median_ms": median,
            "mean_ms": mean,
            "min_ms": lo,
            "max_ms": hi,
            "spread": spread,
        }
        print(f"{case:<16} {median:>10.6f} {mean:>10.6f} {lo:>10.6f} {hi:>10.6f} {spread:>7.1%}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"rounds": args.rounds, "iters": args.iters, "cases": report}, indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
