###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The speculative drafter, for a projection that has no measurement of it.

Two things decide what speculation buys: what the draft costs, and how many
drafts the target keeps.

The cost is the drafter's own forward, and a drafter is a small piece of the
target: an MTP head is one more layer of the target's own shape, an EAGLE3
head is one dense layer over a pruned vocabulary, and DeepSeek's DSpark is a
three-layer MoE backbone run once per block with a low-rank serial head on top.
``Drafter`` records that shape; the projector prices it from the same layer,
head and sampling costs the target step is priced from.

The acceptance is a chain: draft ``i`` is only reached if every draft before it
was accepted, so a step commits ``AL = 1 + sum_i prod_{j<=i} AR_j`` tokens,
the conditional rate ``AR_j`` falling with position. The level comes only from
measured acceptance-length files (``golden_al``): the chain is solved to
reproduce the file's AL at the depth being run, so the files fix how many
tokens a step commits and the chain only spreads them over draft positions.

An entropy class can be asked for on top. The files are SPEED-Bench's coding
domain; following SPEED-Bench (arXiv 2604.09557) code and math accept long
drafts, creative writing and roleplay short ones, and STEM and QA sit between.
Asking for a class moves the chain by the ratio of that class's acceptance to
coding's in SPEED-Bench Table 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

# A drafter never keeps every draft; capping the rate keeps the chain finite
# when an entropy multiplier would push it past one.
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


# What the flat draft-cost fallback charges per draft token, as a fraction of
# one target decode token, when the drafter cannot be priced from its own
# layers (a draft model with no stated depth, or a simulator that cannot price
# the target's layers on this host). Not measured: an order-of-magnitude guess
# kept so a fallback does not silently draft for free.
FLAT_DRAFT_COST = {
    "mtp": 0.05,
    "eagle3": 0.08,
    "dspark": 0.05,
    "draft_model": 0.15,
    "ngram": 0.0,
}


# ---- acceptance ---------------------------------------------------------------


def chain_rates(a: float, decay: float, k: int) -> list[float]:
    """Conditional acceptance ``AR_1..AR_k`` of a chain starting at ``a``."""
    return [min(1.0, max(0.0, a * decay**j)) for j in range(max(0, int(k)))]


def acceptance_length(rates) -> float:
    """Tokens one verify step commits: the bonus token plus the kept drafts."""
    al, keep = 1.0, 1.0
    for r in rates:
        keep *= r
        al += keep
    return al


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
# DSpark has no SPEED-Bench column; it drafts off the target's fused hidden
# states like an MTP head, so it takes MTP's entropy sensitivity.
_ENTROPY_SOURCE = {"dspark": "mtp"}


def resolve_entropy(value) -> float | None:
    """An entropy setting as a point on the low (-1) / mixed (0) / high (+1) axis.

    ``None`` (or ``coding``) leaves the curve as measured. Otherwise a class
    name (``low`` / ``mixed`` / ``high``, or SPEED-Bench's ``low_entropy`` /
    ``high_entropy``) or a number on that axis, clamped to it since nothing
    measured lies beyond its ends.
    """
    if value is None or value == "":
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("coding", "none", "golden"):
            return None
        if key in _ENTROPY_NAMES:
            return _ENTROPY_POINT[_ENTROPY_NAMES[key]]
        value = float(key)
    return max(-1.0, min(1.0, float(value)))


@cache
def _class_rates(method: str) -> dict[str, float]:
    """First-position rate per entropy class and for coding, from SPEED-Bench Table 1."""
    from . import golden_al

    m = _ENTROPY_SOURCE.get(method, method)
    table = golden_al._speed_bench()
    dl = int(table["draft_length"])
    decay = golden_al.fit_chain(golden_al.default_curve(m).points("on")[0])[1]

    def mean_rate(domains) -> float:
        rates = [
            golden_al.invert_rate(al, dl, decay) for al in golden_al.speed_bench_al(m, domains)
        ]
        return sum(rates) / len(rates)

    out = {cls: mean_rate(doms) for cls, doms in table["entropy_classes"].items()}
    out["coding"] = mean_rate(["Coding"])
    return out


def entropy_multiplier(method: str, entropy) -> float:
    """How the first-position rate moves from coding, where the curves were measured.

    The ratio of the class's SPEED-Bench rate to the coding rate for the same
    method; between classes the log of the rate is linear on the axis.
    """
    x = resolve_entropy(entropy)
    if x is None:
        return 1.0
    m = canonical_method(method) or "mtp"
    if m not in ("mtp", "eagle3", "dspark", "draft_model", "ngram"):
        return 1.0
    rates = _class_rates(m)
    end = rates["low"] if x < 0 else rates["high"]
    mid = rates["mixed"]
    if min(end, mid, rates["coding"]) <= 0.0:
        return 1.0
    target = math.exp(math.log(mid) + abs(x) * math.log(end / mid))
    return target / rates["coding"]


def acceptance_profile(
    acceptance,
    k: int,
    *,
    method: str = "mtp",
    entropy=None,
    reference: tuple[int, float] | None = None,
) -> list[float]:
    """Conditional acceptance ``AR_1..AR_k`` at draft depth ``k``.

    ``acceptance`` is the resolved curve (``golden_al.Acceptance``): its AL at
    ``k`` is what the chain commits. A ``reference`` -- an acceptance length
    stated for this deployment at one depth -- rescales the curve so it passes
    through that point, keeping the curve's shape between depths. ``entropy``
    then moves the whole chain from coding to the asked-for class.
    """
    from .golden_al import invert_rate

    k = int(k)
    if k <= 0:
        return []
    al = acceptance.at(k)
    if reference and reference[0] > 0 and reference[1] > 1.0:
        ref_k, ref_al = int(reference[0]), float(reference[1])
        base = acceptance.at(ref_k) - 1.0
        al = 1.0 + (al - 1.0) * (ref_al - 1.0) / base if base > 0.0 else ref_al
        al = min(float(k + 1), max(1.0, al))
    # Every draft kept: no decay can reach that, so the chain is flat.
    decay = 1.0 if al >= k + 1 - 1e-9 else acceptance.decay
    a = invert_rate(al, k, decay)
    mult = entropy_multiplier(method, entropy)
    if mult != 1.0:
        a = min(AR_MAX, a * mult)
    return chain_rates(a, decay, k)


def reference_from_rate(k: int, rate: float) -> tuple[int, float] | None:
    """The acceptance length a single per-token ``rate`` implies at depth ``k``.

    A flat rate commits ``1 + a + ... + a^k`` tokens a step; reading it back as
    a length lets a caller that states acceptance that way anchor the curve.
    """
    if k <= 0 or rate <= 0.0:
        return None
    if rate >= 1.0:
        return (k, float(k + 1))
    return (k, (1.0 - rate ** (k + 1)) / (1.0 - rate))
