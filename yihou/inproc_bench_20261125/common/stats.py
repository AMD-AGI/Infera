"""Small, dependency-free statistics shared by both kits (also used by the CPU tests)."""

import math


def percentile(xs, q):
    """Linear-interpolated percentile, q in [0, 100]. Matches numpy's default ('linear')."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def summarize(xs):
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": sum(xs) / len(xs),
        "median": percentile(xs, 50),
        "p10": percentile(xs, 10),
        "p90": percentile(xs, 90),
        "min": min(xs),
        "max": max(xs),
    }
