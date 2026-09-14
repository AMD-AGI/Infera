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


def reduce_rounds(values: list) -> tuple[float | None, dict]:
    """One number from a metric's per-round values, plus what was reduced.

    **Why this exists: the gate used to judge every round separately, and that
    made it worse the more evidence it was given.** `compare` emitted one row
    per (round, metric, column) and `check_no_regression` refused if any row
    breached, so R rounds were 7R independent tests at the bar. Measured
    family-wise false-refusal at a 1% per-row rate: 6.8% at R=1, 29.7% at R=5,
    50.5% at R=10. An instrument that gets less trustworthy as evidence
    accumulates is the wrong shape, and no choice of bar fixes it.

    **Median, and the choice is honest rather than measured.** At R=1 it is the
    identity, so every artefact produced so far is judged exactly as before and
    this change cannot move an existing verdict. At R>=3 it costs one bad round
    rather than being dragged by it, which is what m4's heavy-tailed kernel
    distribution argues for. But **the round-to-round distribution of these
    metrics has never been measured** — every sealed bench artefact in the
    package carries exactly one round — so median over mean is reasoning, not
    evidence. `todo.md` T25 owes the measurement; revisit with it in hand.

    Returns `(reduced, detail)`. `detail` carries `n`, the values reduced, and
    the mean and spread, so a reader can see what the median stood for and a
    later reduction can be argued against the same record.
    """
    seen = [v for v in values if isinstance(v, (int, float))]
    if not seen:
        return None, {"n": 0, "values": [], "mean": None, "spread": None}
    ordered = sorted(seen)
    mid = len(ordered) // 2
    reduced = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    mean = sum(seen) / len(seen)
    # Peak-to-peak rather than a standard deviation: at the R this gate will
    # ever see (1 to about 5) an sd is not informative and a range is.
    spread = (max(seen) - min(seen)) / mean if mean else None
    return reduced, {
        "n": len(seen),
        "values": [round(v, 6) for v in seen],
        "mean": round(mean, 6),
        "spread": None if spread is None else round(spread, 6),
    }


def pooled_mean(values: list, counts: list) -> tuple[float | None, int]:
    """The mean over every request in every round, and how many that was.

    **No per-request data has to travel for this.** A pooled mean is the round
    averages weighted by their request counts — `sum(v_i * n_i) / sum(n_i)` — and
    AIPerf's summary already carries both per round. So the statistic that makes
    `noise_floor` exact costs nothing extra to compute and nothing extra to
    carry, and `check_no_regression` can recompute it from the same rows.

    Returns `(mean, n)`. `n` is the pooled request count, which is what the floor
    is a function of: pooling R rounds narrows it by about sqrt(R), which is the
    whole reason this statistic is worth having.
    """
    pairs = [(v, c) for v, c in zip(values, counts)
             if isinstance(v, (int, float)) and isinstance(c, (int, float)) and c > 0]
    if not pairs:
        return None, 0
    total = sum(c for _, c in pairs)
    return sum(v * c for v, c in pairs) / total, int(total)


def noise_floor(rsd: float | None, n: int | None) -> float | None:
    """The smallest relative difference this measurement can resolve.

    The 95th percentile of |rel_delta| between two arms that are **the same
    deployment**: with each arm's statistic a mean of `n` requests whose
    per-request relative spread is `rsd`, the difference of two such means is
    approximately normal with sd `sqrt(2) * rsd / sqrt(n)`, so

        floor = 1.96 * sqrt(2) * rsd / sqrt(n)

    **Closed form and not a bootstrap, so a validator can recompute it.** Checked
    against a 20 000-draw bootstrap over the sealed arms' own per-request
    records, six cases: agreement within 0.9 points everywhere and exact to
    three decimals on three of them (stock 12.8 vs 13.1, 17.9 vs 18.5, 20.3 vs
    21.2; patched 5.7 vs 5.7, 3.7 vs 3.8, 10.1 vs 10.1). `avg`, `std` and
    `request_count` are all already in AIPerf's summary, so nothing extra is
    measured and nothing per-request has to travel in the report.

    **Only valid for a mean.** A quantile's sampling distribution needs the
    density at the quantile, which no summary carries — measured on the same
    arms, a p90 floor runs from 5.3% to 20.0% with no stable relation to the
    mean's. So `p90` columns are not gated on this; their fix is more samples
    (pooling took ttft p90 from 20.0% to 5.5% at R=5), not a floor.

    **Valid for the statistic actually compared, and that is now the pooled
    mean.** An earlier round of this gated on `R=1`, because the judged
    statistic was a *median of R round averages* whose noise is dominated by
    round-to-round drift that no within-round dispersion can see — claiming a
    floor there was optimistic, the direction that lets a run assert it can
    resolve a difference it cannot.

    `pooled_mean` removed that restriction rather than working around it: the
    judged statistic **is** a mean over `n` requests again, at any R, with `n`
    the pooled count. So the formula is exact at every R and pooling narrows the
    floor by about sqrt(R), which is the point.

    The median is still computed and still carried — see `compare`'s
    `reduction_disagrees` — but it is the cross-check, not the thing the floor
    describes.
    """
    if rsd is None or not n or n <= 0:
        return None
    return Z95 * math.sqrt(2.0) * float(rsd) / math.sqrt(float(n))


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
