###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The speculative drafter, for a projection that has no measurement of it.

Two things decide what speculation buys, and the simulate path knew neither.
What the draft costs was a free knob that defaults to nothing, so every
speculating stack drafted for free. How many drafts the target keeps was a
single per-token rate handed in from outside, so the projection could not say
how that changes with the draft depth or with what is being generated.

The cost is the drafter's own forward, and a drafter is a small piece of the
target: an MTP head is one more layer of the target's own shape, an EAGLE3
head is one dense layer over a pruned vocabulary, and DeepSeek's DSpark is a
three-layer MoE backbone run once per block with a low-rank serial head on top.
``Drafter`` records that shape; the projector prices it from the same layer,
head and sampling costs the target step is priced from.

The acceptance is a chain: draft ``i`` is only reached if every draft before it
was accepted, so a step commits ``AL = 1 + sum_i prod_{j<=i} AR_j`` tokens,
the conditional rate ``AR_j`` falling with position. Following SPEED-Bench
(arXiv 2604.09557) the rate is a property of the drafter *and* of the domain's
entropy: code completion and sorting accept long drafts, creative writing and
roleplay short ones, STEM and QA sit between. Their throughput split names the
three classes ``low_entropy``, ``mixed`` and ``high_entropy``.

Everything below is fitted, at import, from published measurements:

* the per-position decay from the golden acceptance lengths InferenceX pins
  its throughput runs to, which exist as a sweep over draft depth for GLM-5.2's
  MTP head and Kimi-K3's DSpark drafter;
* each method's level and its entropy sensitivity from SPEED-Bench Table 1,
  which reports acceptance length per domain at a draft length of 3.

A projection that knows its own drafter's acceptance at one depth passes it as
the reference, and that reference is reproduced exactly at that depth; the
fitted shape only carries it to other depths and other entropy classes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

# A drafter never keeps every draft; capping the rate keeps the chain finite
# when a low-entropy multiplier would push it past one.
AR_MAX = 0.99

_ALIASES = {
    "mtp": "mtp",
    "deepseek_mtp": "mtp",
    "nextn": "mtp",
    "qwen3_next_mtp": "mtp",
    "glm4_moe_mtp": "mtp",
    "eagle": "mtp",
    "eagle3": "eagle3",
    "eagle-3": "eagle3",
    "dspark": "dspark",
    "ngram": "ngram",
    "n-gram": "ngram",
    "prompt_lookup": "ngram",
    "draft_model": "draft_model",
    "vanilla": "draft_model",
    "standalone": "draft_model",
}


def canonical_method(name: str | None) -> str | None:
    """The drafter family a method name belongs to, or ``None`` if unknown.

    SGLang's ``EAGLE`` algorithm is how it serves a checkpoint's own MTP head
    (Qwen3.5, GLM), so it is read as MTP; ``EAGLE3`` is the separate head.
    """
    if not name:
        return None
    return _ALIASES.get(str(name).strip().lower())


@dataclass(frozen=True)
class Drafter:
    """The shape of one drafter, in units of the target's own pieces.

    ``layers`` draft layers run per pass; ``moe`` says they carry the target's
    expert FFN rather than a dense one. A ``parallel`` drafter runs one pass
    over the whole block, an autoregressive one a pass per draft token.
    ``window`` caps the context a draft layer attends over (0 = all of it).
    ``vocab_fraction`` sizes the draft LM head against the target's, and
    ``serial_head_rank`` is the rank of a block drafter's left-to-right head,
    one ``rank x vocab`` projection and one sample per draft position.
    """

    method: str
    layers: int
    moe: bool
    parallel: bool
    window: int = 0
    vocab_fraction: float = 1.0
    serial_head_rank: int = 0


# EAGLE3 heads prune the vocabulary they draft over, usually to 32k tokens
# (SPEED-Bench Sec. 8.3).
EAGLE3_DRAFT_VOCAB = 32000

# DSpark on DeepSeek-V4: "three MoE layers with mHC and a sliding window
# attention of 128", a rank-256 Markov head (arXiv 2607.05147, Secs. 3.1, 5.1).
DSPARK_LAYERS = 3
DSPARK_WINDOW = 128
DSPARK_MARKOV_RANK = 256


def drafter_for(
    method: str,
    *,
    target_vocab: int,
    mtp_layers: int = 0,
    mtp_window: int = 0,
    layers: int | None = None,
    vocab: int | None = None,
    window: int | None = None,
) -> Drafter:
    """The drafter a method runs, with the target's geometry filled in.

    The explicit ``layers`` / ``vocab`` / ``window`` override the method's
    defaults, for a checkpoint whose drafter differs from the published one.
    """
    m = canonical_method(method)
    if m is None:
        raise ValueError(f"unknown speculative method {method!r}")
    tv = max(1, int(target_vocab or 1))

    def frac(v: int | None, default: float) -> float:
        return min(1.0, max(1, int(v)) / tv) if v else default

    if m == "mtp":
        d = Drafter(
            m,
            layers=int(layers or mtp_layers or 1),
            moe=True,
            parallel=False,
            window=int(mtp_window if window is None else window),
            vocab_fraction=frac(vocab, 1.0),
        )
    elif m == "eagle3":
        d = Drafter(
            m,
            layers=int(layers or 1),
            moe=False,
            parallel=False,
            window=int(window or 0),
            vocab_fraction=frac(vocab or EAGLE3_DRAFT_VOCAB, 1.0),
        )
    elif m == "dspark":
        d = Drafter(
            m,
            layers=int(layers or DSPARK_LAYERS),
            moe=True,
            parallel=True,
            window=int(DSPARK_WINDOW if window is None else window),
            vocab_fraction=frac(vocab, 1.0),
            serial_head_rank=DSPARK_MARKOV_RANK,
        )
    elif m == "ngram":
        d = Drafter(m, layers=0, moe=False, parallel=False)
    else:
        # A separate draft checkpoint: priced as ``layers`` dense layers of the
        # target's width, so it needs the depth stated.
        if not layers:
            raise ValueError("a draft_model drafter needs speculative_draft_layers")
        d = Drafter(
            m,
            layers=int(layers),
            moe=False,
            parallel=False,
            window=int(window or 0),
            vocab_fraction=frac(vocab, 1.0),
        )
    return d


# ---- acceptance ---------------------------------------------------------------

# SPEED-Bench Table 1: mean acceptance length at a draft length of 3, BS=32,
# temperature 0, on the Qualitative Split, one entry per target the method was
# run on. N-Gram: Llama 3.3 70B, GPT-OSS 120B. Vanilla: Llama 3.3 70B,
# Qwen3 235B. EAGLE3: Llama 3.3 70B, GPT-OSS 120B, Qwen3 235B, Qwen3-Next.
# MTP: DeepSeek R1, Qwen3-Next.
_SPEED_BENCH_DL = 3
_SPEED_BENCH_AL = {
    "ngram": {
        "Coding": (1.54, 1.31),
        "Math": (1.43, 1.30),
        "STEM": (1.38, 1.30),
        "QA": (1.21, 1.27),
        "Writing": (1.33, 1.20),
        "Roleplay": (1.15, 1.25),
    },
    "draft_model": {
        "Coding": (2.72, 2.62),
        "Math": (2.43, 2.69),
        "STEM": (2.45, 2.44),
        "QA": (2.35, 2.28),
        "Writing": (2.45, 2.19),
        "Roleplay": (2.14, 1.92),
    },
    "eagle3": {
        "Coding": (3.00, 2.46, 2.26, 3.17),
        "Math": (2.45, 2.46, 2.37, 2.90),
        "STEM": (2.39, 2.28, 2.18, 2.47),
        "QA": (2.35, 2.25, 2.23, 2.13),
        "Writing": (2.63, 1.98, 2.02, 2.09),
        "Roleplay": (2.04, 1.87, 1.87, 1.80),
    },
    "mtp": {
        "Coding": (2.76, 3.34),
        "Math": (2.77, 3.13),
        "STEM": (2.62, 2.85),
        "QA": (2.52, 2.71),
        "Writing": (2.33, 2.46),
        "Roleplay": (2.14, 2.09),
    },
}

# The three entropy classes of SPEED-Bench's throughput split, by the
# Qualitative-Split domains the paper files under each: low (coding, math),
# mixed (STEM and QA, the category its step latencies were measured on), high
# (creative writing, roleplay).
ENTROPY_CLASSES = {
    "low": ("Coding", "Math"),
    "mixed": ("STEM", "QA"),
    "high": ("Writing", "Roleplay"),
}
_ENTROPY_POINT = {"low": -1.0, "mixed": 0.0, "high": 1.0}
_ENTROPY_NAMES = {
    "low": "low",
    "low_entropy": "low",
    "mixed": "mixed",
    "mixed_entropy": "mixed",
    "medium": "mixed",
    "high": "high",
    "high_entropy": "high",
}

# Golden acceptance length by draft depth, as InferenceX pins its throughput
# runs: GLM-5.2's MTP head (glm5.2_mtp.yaml) and Kimi-K3's DSpark drafter
# (kimik3_dspark golden_al_distribution). The only published sweeps over depth.
_DEPTH_CURVES = {
    "mtp": {3: 2.99, 4: 3.33, 5: 3.61},
    "dspark": {3: 3.00, 4: 3.36, 5: 3.62, 7: 3.84},
}
# Methods without a depth sweep of their own take the MTP decay. SPEED-Bench
# reports EAGLE3 losing acceptance faster with depth than MTP or a vanilla
# draft model, but publishes no chain numbers past a draft length of 3.
_DECAY_SOURCE = {"mtp": "mtp", "dspark": "dspark"}
# DSpark has no SPEED-Bench column; it drafts off the target's fused hidden
# states like an MTP head, so it takes MTP's entropy sensitivity.
_ENTROPY_SOURCE = {"dspark": "mtp"}


def chain_rates(a: float, decay: float, k: int) -> list[float]:
    """Conditional acceptance ``AR_1..AR_k`` of a chain starting at ``a``."""
    return [min(AR_MAX, max(0.0, a * decay**j)) for j in range(max(0, int(k)))]


def acceptance_length(rates) -> float:
    """Tokens one verify step commits: the bonus token plus the kept drafts."""
    al, keep = 1.0, 1.0
    for r in rates:
        keep *= r
        al += keep
    return al


def _invert_rate(al: float, k: int, decay: float) -> float:
    """The first-position rate whose chain commits ``al`` tokens at depth ``k``."""
    if k <= 0 or al <= 1.0:
        return 0.0
    lo, hi = 0.0, AR_MAX
    if acceptance_length(chain_rates(hi, decay, k)) <= al:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if acceptance_length(chain_rates(mid, decay, k)) < al:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@cache
def _fit_curve(source: str) -> tuple[float, float]:
    """Least-squares ``(a, decay)`` through one depth sweep."""
    curve = _DEPTH_CURVES[source]
    best = (float("inf"), 0.0, 1.0)
    for i in range(501):
        d = 0.5 + i * 0.001
        lo, hi = 0.0, AR_MAX
        # SSE is unimodal in ``a`` at fixed decay: golden-section on it.
        g = (math.sqrt(5.0) - 1.0) / 2.0
        x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)

        def sse(a: float, d: float = d) -> float:
            return sum(
                (acceptance_length(chain_rates(a, d, k)) - al) ** 2 for k, al in curve.items()
            )

        f1, f2 = sse(x1), sse(x2)
        for _ in range(50):
            if f1 < f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - g * (hi - lo)
                f1 = sse(x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + g * (hi - lo)
                f2 = sse(x2)
        a = 0.5 * (lo + hi)
        err = sse(a)
        if err < best[0]:
            best = (err, a, d)
    return best[1], best[2]


def decay_for(method: str) -> float:
    m = canonical_method(method) or "mtp"
    return _fit_curve(_DECAY_SOURCE.get(m, "mtp"))[1]


@cache
def _class_rates(method: str) -> dict[str, float]:
    """First-position rate per entropy class, from SPEED-Bench Table 1."""
    m = _ENTROPY_SOURCE.get(method, method)
    table = _SPEED_BENCH_AL[m]
    d = decay_for(m)
    out = {}
    for cls, domains in ENTROPY_CLASSES.items():
        rates = [_invert_rate(al, _SPEED_BENCH_DL, d) for dom in domains for al in table[dom]]
        out[cls] = sum(rates) / len(rates)
    return out


def resolve_entropy(value) -> float:
    """An entropy setting as a point on the low (-1) / mixed (0) / high (+1) axis.

    Takes a class name (``low`` / ``mixed`` / ``high``, or SPEED-Bench's
    ``low_entropy`` / ``high_entropy``) or a number on that axis; numbers are
    clamped to it, since nothing measured lies beyond its ends.
    """
    if value is None or value == "":
        return 0.0
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _ENTROPY_NAMES:
            return _ENTROPY_POINT[_ENTROPY_NAMES[key]]
        value = float(key)
    return max(-1.0, min(1.0, float(value)))


def entropy_multiplier(method: str, entropy) -> float:
    """How the first-position rate moves from the mixed class to ``entropy``.

    Read off SPEED-Bench as the ratio of each class's rate to the mixed
    class's for the same method; between classes the log-ratio is linear.
    """
    m = canonical_method(method) or "mtp"
    rates = _class_rates(m)
    x = resolve_entropy(entropy)
    end = rates["low"] if x < 0 else rates["high"]
    base = rates["mixed"]
    if base <= 0.0 or end <= 0.0:
        return 1.0
    return math.exp(abs(x) * math.log(end / base))


def default_rate(method: str) -> float:
    """A method's mixed-class first-position rate, with no reference given."""
    m = canonical_method(method) or "mtp"
    if m == "dspark":
        return _fit_curve("dspark")[0]
    return _class_rates(m)["mixed"]


def acceptance_profile(
    method: str,
    k: int,
    *,
    entropy=None,
    reference: tuple[int, float] | None = None,
) -> list[float]:
    """Conditional acceptance ``AR_1..AR_k`` of ``method`` at draft depth ``k``.

    ``reference`` is a measured ``(depth, acceptance length)`` for this
    target's own drafter, taken as the mixed class; without it the method's
    SPEED-Bench level stands in. ``entropy`` then moves the whole chain.
    """
    m = canonical_method(method) or "mtp"
    d = decay_for(m)
    if reference and reference[0] > 0 and reference[1] > 1.0:
        a = _invert_rate(float(reference[1]), int(reference[0]), d)
    else:
        a = default_rate(m)
    a *= entropy_multiplier(m, entropy)
    return chain_rates(min(AR_MAX, a), d, k)


def reference_from_rate(k: int, rate: float) -> tuple[int, float] | None:
    """The acceptance length a single per-token ``rate`` implies at depth ``k``.

    The projector's legacy input is that flat rate (``1 + a + ... + a^k``
    tokens a step); reading it back as a length lets a caller that only has
    the rate still anchor the chain.
    """
    if k <= 0 or rate <= 0.0:
        return None
    if rate >= 1.0:
        return (k, float(k + 1))
    return (k, (1.0 - rate ** (k + 1)) / (1.0 - rate))
