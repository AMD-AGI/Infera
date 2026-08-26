#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
"""Turn eval result directories into numbers that can actually be compared.

``sglang.test.run_eval`` writes a score and nothing that bounds it. A bare 0.945 invites
exactly one question -- "is that better or worse than last time" -- and cannot answer it,
because the honest answer depends on how many questions produced it. At 200 questions a
two-point move is noise; at 1300 it is not. So every score here carries its 95% interval,
and every comparison carries the interval of the DIFFERENCE.

    summarise_eval.py <dir>              one run: scores, intervals, prefix-reuse delta
    summarise_eval.py <dir-a> <dir-b>    two runs: per-eval delta with its own interval
    summarise_eval.py --gate <a> <b>     the same, exiting 1 if some eval dropped

Comparing is the mode that answers the question that comes up. Nobody knows what
GLM-5.2-MXFP4 with an fp8 KV cache "should" score, so an absolute number has no external
baseline -- but a config change, a leg restart or a new image can always be measured
against the same deployment's own earlier run.

WHY A DIFFERENCE GETS ITS OWN INTERVAL RATHER THAN A COMPARISON OF TWO INTERVALS.
"Do the two intervals overlap" is the intuitive test and it is too conservative: the
uncertainty on a difference is sqrt(se_a^2 + se_b^2), not se_a + se_b, so two runs can
overlap visibly and still differ significantly. Reading overlap hides real regressions.
The difference is estimated directly instead, and the question becomes whether its
interval contains zero.

Input is the ``.index`` that engine/lm_eval.sh writes beside the results
(``<eval>\\t<scored>\\t<json filename>``); the question count lives there because
run_eval's own json does not record it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import NamedTuple

# Named because it appears in three places, and a stray 1.65 in one of them would silently
# turn a 95% statement into a 90% one.
Z95 = 1.96

RED, GRN, YEL, NC = "\033[0;31m", "\033[0;32m", "\033[0;33m", "\033[0m"


class Result(NamedTuple):
    name: str
    score: float
    # Questions scored, counted by lm_eval.sh out of the html report. run_eval divides by
    # this, so the interval must too. Smaller than the dataset: GSM8K ships 1319 rows and
    # `gsm8k` scores 1314, the few-shot examples being sliced off the evaluation set.
    n: int | None
    latency: float | None
    repeats: list | None

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson(self.score, self.n) if self.n else None


def wilson(p: float, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Not the textbook Wald interval (p +/- z*sqrt(p(1-p)/n)): that one collapses to zero
    width at p=0 and p=1, so a 16-question run that happens to score 1.000 reports
    [1.000, 1.000] -- an assertion of certainty drawn from sixteen questions. Wilson stays
    finite there (the same run reports [0.806, 1.000]) and is better behaved at the high
    scores this deployment produces. The two agree to a few thousandths by 200 questions.
    """
    d = 1.0 + Z95 * Z95 / n
    centre = (p + Z95 * Z95 / (2 * n)) / d
    half = Z95 / d * math.sqrt(max(p * (1.0 - p), 0.0) / n + Z95 * Z95 / (4 * n * n))
    return max(centre - half, 0.0), min(centre + half, 1.0)


def load(run_dir: str) -> dict[str, Result]:
    index = os.path.join(run_dir, ".index")
    if not os.path.isdir(run_dir):
        raise SystemExit(f"not a directory: {run_dir}")
    if not os.path.exists(index):
        raise SystemExit(f"no .index in {run_dir} — was this written by engine/lm_eval.sh?")

    out: dict[str, Result] = {}
    for line in open(index):
        # Filename taken from the last column so a stray extra field cannot shift it.
        cols = [c for c in line.rstrip("\n").split("\t") if c]
        if len(cols) < 3:
            continue
        name, n_raw, path = cols[0], cols[1], os.path.join(run_dir, cols[-1])
        if not os.path.exists(path):
            continue
        try:
            d = json.load(open(path))
        except Exception:  # noqa: BLE001
            continue
        score = d.get("score", d.get("mean_score"))
        if score is None:
            continue
        out[name] = Result(
            name,
            float(score),
            int(n_raw) if n_raw.isdigit() else None,
            d.get("latency"),
            d.get("scores"),
        )
    return out


def diff_interval(a: Result, b: Result) -> tuple[float, float, float] | None:
    """(delta, lo, hi) for b.score - a.score, or None when either n is unknown.

    Newcombe's method, which composes the two Wilson intervals rather than adding variances,
    so it inherits their behaviour at the extremes instead of degenerating alongside Wald.
    """
    if not a.n or not b.n:
        return None
    lo_a, hi_a = wilson(a.score, a.n)
    lo_b, hi_b = wilson(b.score, b.n)
    delta = b.score - a.score
    return (
        delta,
        delta - math.sqrt((b.score - lo_b) ** 2 + (hi_a - a.score) ** 2),
        delta + math.sqrt((hi_b - b.score) ** 2 + (a.score - lo_a) ** 2),
    )


def report_one(results: dict[str, Result]) -> int:
    print(f"  {'eval':<22}{'scored':>7}{'score':>8}{'95% CI':>20}{'latency':>10}")
    for r in results.values():
        iv = r.interval
        ci = f"[{iv[0]:.3f}, {iv[1]:.3f}]" if iv else ""
        lat = f"{r.latency:.0f}s" if isinstance(r.latency, (int, float)) else ""
        print(f"  {r.name:<22}{r.n or '?':>7}{r.score:>8.3f}{ci:>20}{lat:>10}")
        if r.repeats:
            print(f"  {'':<22}repeats: {r.repeats}")

    # The one comparison this kit exists to make. mixed_prefix_gsm8k asks the SAME questions
    # as gsm8k behind partially-shared few-shot prefixes, so it is the only load here that
    # puts the radix cache, kvd and kv-aware routing on the correctness path. A gap is prefix
    # reuse changing answers, which no throughput number would ever show.
    base, mixed = results.get("gsm8k"), results.get("mixed_prefix_gsm8k")
    if not (base and mixed):
        return 0
    print()
    d = diff_interval(base, mixed)
    if d is None:
        print(
            f"  prefix-reuse delta (mixed_prefix_gsm8k - gsm8k): "
            f"{mixed.score - base.score:+.3f}  (no interval: question count unknown)"
        )
        return 0
    delta, lo, hi = d
    print(
        f"  prefix-reuse delta (mixed_prefix_gsm8k - gsm8k): {delta:+.3f}"
        f"   95% CI [{lo:+.3f}, {hi:+.3f}]"
    )
    if lo <= 0.0 <= hi:
        print(
            f"  {GRN}the interval contains zero{NC} — no evidence that prefix reuse changes answers"
        )
    else:
        print(f"  {RED}the interval excludes zero{NC} — shared prefixes ARE changing answers.")
        print("  suspect the radix cache, kvd L2/L3 or kv-aware routing, not the model:")
        print("    re-run with ROUTER_POLICY=round-robin, then with PREFILL_KVD=0,")
        print("    and see which one closes the gap.")
    return 0


def report_compare(
    a: dict[str, Result], b: dict[str, Result], dir_a: str, dir_b: str, gate: bool
) -> int:
    print(f"  A  {dir_a}\n  B  {dir_b}\n")
    print(f"  {'eval':<22}{'A':>8}{'B':>9}{'delta':>9}{'95% CI of delta':>22}   verdict")

    shared = [k for k in a if k in b]
    regressed = False
    for name in shared:
        ra, rb = a[name], b[name]
        d = diff_interval(ra, rb)
        if d is None:
            print(
                f"  {name:<22}{ra.score:>8.3f}{rb.score:>9.3f}"
                f"{rb.score - ra.score:>+9.3f}{'(n unknown)':>22}   ?"
            )
            continue
        delta, lo, hi = d
        if lo <= 0.0 <= hi:
            verdict = f"{GRN}same{NC}"
        elif delta < 0:
            verdict, regressed = f"{RED}REGRESSED{NC}", True
        else:
            verdict = f"{YEL}improved{NC}"
        print(
            f"  {name:<22}{ra.score:>8.3f}{rb.score:>9.3f}{delta:>+9.3f}"
            f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}   {verdict}"
        )
        # Different sizes are legal and the interval accounts for them, but the two sides are
        # then scoring different question sets and a reader would not otherwise notice.
        if ra.n != rb.n:
            print(
                f"  {'':<22}{YEL}scored: A={ra.n} B={rb.n}{NC} — different question sets,"
                f" so this delta is not quite like-for-like"
            )

    for label, only in (("A", [k for k in a if k not in b]), ("B", [k for k in b if k not in a])):
        if only:
            print(f"\n  only in {label}: {', '.join(only)}")
    if not shared:
        print("\n  the two runs share no eval — there is nothing to compare")
        return 0

    print("\n  'same' means the delta's interval contains zero, i.e. this pair of runs cannot")
    print("  tell the two configurations apart. It is not proof that they are identical —")
    print("  with 200 questions per side the interval is roughly +/-5 points wide.")
    if gate and regressed:
        print(f"\n{RED}REGRESSION{NC} — at least one eval dropped by more than measurement noise")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "dirs",
        nargs="+",
        metavar="DIR",
        help="one result directory to summarise, or two to compare (A then B)",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="in compare mode, exit 1 if any eval dropped significantly",
    )
    args = ap.parse_args()

    if len(args.dirs) > 2:
        raise SystemExit("at most two directories: summarise one, or compare two")
    loaded = [load(d) for d in args.dirs]
    if not any(loaded):
        print("  (no results)")
        return 0
    if len(loaded) == 1:
        return report_one(loaded[0])
    return report_compare(loaded[0], loaded[1], args.dirs[0], args.dirs[1], args.gate)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
