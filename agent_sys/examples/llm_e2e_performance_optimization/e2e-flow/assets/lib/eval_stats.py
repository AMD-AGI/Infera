#!/usr/bin/env python3
"""Intervals for eval scores, and for the difference between two of them.

Ported from `examples/sglang_1p1d_glm5.2/engine/tools/summarise_eval.py`, which
is the same arithmetic serving the same purpose one stage earlier. It lives in
`lib/` because `compare` computes the verdict with it and `check_no_regression`
recomputes the verdict with it, and those two agreeing is the whole point of the
validator -- which they cannot do if each carries its own copy.

Why intervals at all: 200 questions is a coin with 200 flips. Two runs of the
same deployment differ by several points routinely, so "the score went down" is
not evidence. What is evidence is a difference whose interval excludes zero.
"""

from __future__ import annotations

import math

Z95 = 1.959963984540054


def wilson(successes: float, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson and not Wald. The Wald interval has zero width at p=0 and p=1, which
    is exactly where an eval that is working perfectly or failing completely
    lands, and a zero-width interval says "certain" about a sample of 200.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def newcombe(succ_a: float, n_a: int, succ_b: float, n_b: int, z: float = Z95) -> tuple[float, float]:
    """95% interval for (p_b - p_a), by Newcombe's composition of two Wilson intervals.

    Not a normal approximation to the difference: the two proportions are
    measured on samples that are small and often near 1.0, where the normal
    approximation understates the interval on the side that matters.
    """
    lo_a, hi_a = wilson(succ_a, n_a, z)
    lo_b, hi_b = wilson(succ_b, n_b, z)
    p_a = succ_a / n_a if n_a else 0.0
    p_b = succ_b / n_b if n_b else 0.0
    delta = p_b - p_a
    lower = delta - math.sqrt((p_b - lo_b) ** 2 + (hi_a - p_a) ** 2)
    upper = delta + math.sqrt((hi_b - p_b) ** 2 + (p_a - lo_a) ** 2)
    return (lower, upper)


def compare_scores(score_a: float, n_a: int, score_b: float, n_b: int) -> dict:
    """Arm A against arm B on one eval. A is the baseline, B is the candidate.

    `verdict` is `same` whenever the interval contains zero, which is the honest
    answer at these sample sizes far more often than not.
    """
    succ_a, succ_b = score_a * n_a, score_b * n_b
    lo, hi = newcombe(succ_a, n_a, succ_b, n_b)
    delta = score_b - score_a
    if lo <= 0.0 <= hi:
        verdict = "same"
    elif delta < 0:
        verdict = "REGRESSED"
    else:
        verdict = "improved"
    return {
        "scored_a": n_a,
        "score_a": round(score_a, 6),
        "ci95_a": [round(x, 6) for x in wilson(succ_a, n_a)],
        "scored_b": n_b,
        "score_b": round(score_b, 6),
        "ci95_b": [round(x, 6) for x in wilson(succ_b, n_b)],
        "delta": round(delta, 6),
        "ci95_delta": [round(lo, 6), round(hi, 6)],
        "verdict": verdict,
    }


def relative_change(a: float, b: float) -> float | None:
    """(b - a) / a, or None when a is zero or missing.

    None rather than 0.0 or inf: a baseline of zero means the measurement did
    not happen, and reporting a 0% change for it is the kind of quiet lie this
    package spends its validators trying to avoid.
    """
    if a is None or b is None or a == 0:
        return None
    return (b - a) / a


def perf_verdict(a: float | None, b: float | None, *, max_regression: float,
                 higher_is_better: bool) -> dict:
    """One metric, one round, both arms.

    `max_regression` is a fraction: 0.05 means the candidate may be 5% worse
    before this says so.
    """
    rel = relative_change(a, b)
    if rel is None:
        return {"stock": a, "patched": b, "rel_delta": None, "verdict": "unmeasured"}
    worse = -rel if higher_is_better else rel
    verdict = "REGRESSED" if worse > max_regression else "same"
    if verdict == "same" and worse < -max_regression:
        verdict = "improved"
    return {
        "stock": a,
        "patched": b,
        "rel_delta": round(rel, 6),
        "verdict": verdict,
    }
