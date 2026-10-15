"""Golden acceptance lengths InferenceX pins its speculative runs to.

InferenceX forces every speculative submission to one committed acceptance
length (AL) per model, thinking mode and draft depth, measured with real
drafting on SPEED-Bench's coding category and injected through each engine's
synthetic-acceptance option (``inferencex-e2e/infx/golden_al_distribution``).
A frontier projected with the same values is comparable to theirs.

Models InferenceX has not measured take the mean curve of their method, and
say so in the source they report.
"""

from __future__ import annotations

# curve -> (method, {thinking: {draft depth: AL}})
CURVES: dict[str, tuple[str, dict[str, dict[int, float]]]] = {
    "glm5.2_mtp": (
        "mtp",
        {"on": {1: 1.84, 2: 2.50, 3: 2.99, 4: 3.33, 5: 3.61, 6: 3.78, 7: 3.91, 8: 4.06}},
    ),
    "dsv4_mtp": (
        "mtp",
        {
            "on": {1: 1.79, 2: 2.27, 3: 2.49, 4: 2.54, 5: 2.55, 6: 2.55, 7: 2.54, 8: 2.58},
            "off": {1: 1.92, 2: 2.61, 3: 2.97, 4: 3.00, 5: 3.10, 6: 3.16, 7: 3.00, 8: 3.11},
        },
    ),
    "qwen3.5_mtp": (
        "mtp",
        {
            "on": {1: 1.93, 2: 2.73, 3: 3.39, 4: 3.97, 5: 4.41, 6: 4.80, 7: 5.04, 8: 5.21},
            "off": {1: 1.92, 2: 2.70, 3: 3.33, 4: 3.83, 5: 4.20, 6: 4.48, 7: 4.73, 8: 4.71},
        },
    ),
    "minimaxm3_eagle3": (
        "eagle3",
        {
            "on": {1: 1.78, 2: 2.38, 3: 2.83, 4: 3.11, 5: 3.35, 6: 3.45, 7: 3.53, 8: 3.74},
            "off": {1: 1.84, 2: 2.52, 3: 2.93, 4: 3.41, 5: 3.59, 6: 3.57, 7: 3.85, 8: 4.06},
        },
    ),
}

# Model preset -> its own curve.
MODEL_CURVES: dict[str, str] = {
    "glm5": "glm5.2_mtp",
    "deepseek_v4_pro": "dsv4_mtp",
    "qwen3_5_397B_A17B": "qwen3.5_mtp",
    "minimax_m3": "minimaxm3_eagle3",
}

# Engine method names -> the curve family. SGLang serves a checkpoint's own
# MTP head as ``EAGLE``/``NEXTN``, so those read as MTP.
_FAMILY = {"mtp": "mtp", "nextn": "mtp", "eagle": "mtp", "eagle3": "eagle3"}


def _depths(curve: str, thinking: str) -> dict[int, float]:
    modes = CURVES[curve][1]
    return modes.get(thinking) or modes["on"]


def golden_al(model: str, method: str, k: int, thinking: str = "on") -> tuple[float, str] | None:
    """``(AL, source)`` for ``model`` drafting ``k`` tokens with ``method``.

    The model's own curve if InferenceX measured it with that method, else the
    mean of the method's curves at ``k``; ``None`` when neither has ``k``.
    """
    fam = _FAMILY.get(method)
    if fam is None or k < 1:
        return None
    own = MODEL_CURVES.get(model)
    if own and CURVES[own][0] == fam:
        al = _depths(own, thinking).get(k)
        if al is not None:
            return al, own
    vals = [
        _depths(c, thinking)[k]
        for c, (m, _) in sorted(CURVES.items())
        if m == fam and k in _depths(c, thinking)
    ]
    if not vals:
        return None
    return sum(vals) / len(vals), f"{fam} mean"


def max_depth(method: str) -> int:
    fam = _FAMILY.get(method)
    return max((max(d) for m, t in CURVES.values() if m == fam for d in t.values()), default=0)


def rate_for_al(al: float, k: int) -> float:
    """The constant per-draft rate ``a`` with ``sum(a**i, i=0..k) == al``."""
    if k < 1 or al <= 1.0:
        return 0.0
    if al >= k + 1:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sum(mid**i for i in range(k + 1)) < al:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
