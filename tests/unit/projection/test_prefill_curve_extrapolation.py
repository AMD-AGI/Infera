###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""How far past its probed lengths a measured prefill curve may be read.

An anchor harvested at four or more prompt lengths gets a quadratic fit, and
the projection bills a prefill step at ``(a + b*n)`` per token. ``b`` carries
GEMM efficiency improving as the step widens, so it is routinely *negative*:
DeepSeek-V4-Pro's TP4 MI355X anchor fits ``b = -1.05e-6`` over 1024..8192
tokens at R2=0.98, which is a good fit and a fine local approximation.

Continued as a quadratic it crosses zero at about 53.5k tokens. At ISL 130000
-- an ordinary agentic prompt -- that curve priced prefill at -80.7 us/token:
negative per-token compute. It read out as a 480-second disaggregated TTFT, and
it ranked that config 31st on a metric where trace replay of the same candidate
puts it first. The projection warned that it was extrapolating and then used
the number anyway.

So the curvature is held at the longest length probed rather than continued
past it, and the resulting rate is floored at something physical. The fixed
cost and the linear term stay exactly as fitted; what stops is crediting
efficiency nobody measured.
"""

from __future__ import annotations

import json

import pytest

# The shape of the defect, verbatim: a concave fit over a short ladder, read at
# a prompt length two orders of magnitude past its longest point.
CONCAVE_FIT = {
    "fixed_ms": 92.50508071374996,
    "ms_per_token": 0.05640328862982155,
    "ms_per_token_sq": -1.0546633852394908e-06,
    "r2": 0.9805863332491604,
}
PROBED = [1024, 2816, 4608, 6400, 8192]
ZERO_CROSSING = CONCAVE_FIT["ms_per_token"] / -CONCAVE_FIT["ms_per_token_sq"]


def _anchor(tmp_path, *, tp=8, fit=None, probed=None):
    """A whole-model vLLM anchor carrying a length-probe curve fit."""
    probed = PROBED if probed is None else probed
    art = {
        "backend": "vllm",
        "measured": {"model": {"prefill_ms": 8944.23, "decode_ms": 30.5}},
        "sweep": [
            {"batch": 1, "prefill_ms": 139.75, "decode_ms": 15.28},
            {"batch": 64, "prefill_ms": 8944.23, "decode_ms": 30.5},
        ],
        "meta": {
            "batch": 64,
            "input_len": 8192,
            "tp": tp,
            "pp": 1,
            "ep": 1,
            "prefix_caching": False,
            "prefill_anchor": {
                "method": "ttft difference across prompt lengths",
                "points": [{"input_len": n, "mean_ttft_ms": 100.0 + n * 0.05} for n in probed],
                "curve_fit": dict(CONCAVE_FIT if fit is None else fit),
            },
        },
    }
    path = tmp_path / f"anchor_tp{tp}.json"
    path.write_text(json.dumps(art))
    return path


def _project(tmp_path, *, input_len, **over):
    from .conftest import project_spec

    return project_spec(
        tp=8,
        ep=1,
        input_len=input_len,
        output_len=128,
        concurrency=64,
        load_benchmark=_anchor(tmp_path, **over),
    )


def test_the_fit_this_guards_against_really_does_go_negative():
    """Not a hypothetical: the coefficients are off a real MI355X anchor."""
    a, b = CONCAVE_FIT["ms_per_token"], CONCAVE_FIT["ms_per_token_sq"]
    assert a + b * max(PROBED) > 0, "the fit is sound over the range it was measured"
    assert ZERO_CROSSING == pytest.approx(53480, rel=0.01)
    assert a + b * 130000 < 0, "and unguarded it prices prefill negative at 130k"


def test_a_much_longer_prompt_costs_much_more_time_to_first_token(tmp_path):
    """TTFT is the metric the sign flip actually destroyed.

    Note what it does *not* look like. ``prefill_throughput_tps`` is a
    reciprocal, so it launders the sign: the unguarded curve reports a
    perfectly ordinary-looking 12396 tok/s at ISL 130000, which is a good part
    of why this survived review. TTFT is where it shows, because the negative
    per-token term cancels the prompt's real work -- unguarded, a 16x longer
    prompt reaches its first token 1.6% later (2425 ms against 8192's 2387).
    """
    short = _project(tmp_path, input_len=8192)["ttft_ms"]
    long = _project(tmp_path, input_len=130000)["ttft_ms"]
    assert long > 4.0 * short, (
        f"a 130000-token prompt is 16x the prefill work of an 8192-token one "
        f"but reaches its first token in {long:.0f} ms against {short:.0f} ms "
        f"({long / short:.2f}x), so per-token prefill cost went negative"
    )


def test_a_longer_prompt_is_never_billed_less_per_token_than_a_shorter_one(tmp_path):
    """Monotonicity is the property the zero crossing broke.

    Prefill throughput is quoted per token, so holding the curvature means a
    prompt past the ladder is billed at the ladder's last rate -- never at a
    better one, which is what a concave extrapolation would invent.
    """
    at_probe = _project(tmp_path, input_len=8192)["prefill_throughput_tps"]
    beyond = _project(tmp_path, input_len=130000)["prefill_throughput_tps"]
    assert beyond <= at_probe * 1.01, (
        f"a 130000-token prompt prefills at {beyond:.0f} tok/s against "
        f"{at_probe:.0f} at the anchor's own 8192, so the curve was read past "
        f"its ladder and credited efficiency it never measured"
    )


def test_the_held_rate_is_the_one_the_anchor_measured_at_its_longest_probe(tmp_path):
    """Holding, not discarding: the fitted curve still sets the rate.

    Refusing the fit outright would fall back to the sweep's average rate and
    throw away a measurement with R2=0.98. The rate past the ladder is the
    fit's own value at its longest probed length.
    """
    a, b = CONCAVE_FIT["ms_per_token"], CONCAVE_FIT["ms_per_token_sq"]
    want_tok_s = 1000.0 / (a + b * max(PROBED))
    got = _project(tmp_path, input_len=130000)["prefill_throughput_tps"]
    assert got == pytest.approx(want_tok_s, rel=0.05), (
        f"expected the fit's rate at {max(PROBED)} tokens (~{want_tok_s:.0f} tok/s), got {got:.0f}"
    )


def test_a_curve_read_inside_its_ladder_is_left_alone(tmp_path):
    """The guard must not move the configs it was not written for.

    Inside the probed span the quadratic is doing the job it was fitted for,
    and a prompt at the anchor's own length must be priced exactly as measured.
    """
    a, b = CONCAVE_FIT["ms_per_token"], CONCAVE_FIT["ms_per_token_sq"]
    got = _project(tmp_path, input_len=4608)["prefill_throughput_tps"]
    assert got == pytest.approx(1000.0 / (a + b * 4608), rel=0.05)


def test_a_convex_fit_still_grows_with_context(tmp_path):
    """The clamp is a ceiling on curvature, not a cap on cost.

    A positive ``b`` -- attention growing with context, which is what the
    quadratic is nominally there to capture -- would be *silenced* by holding
    the rate if the hold applied in both directions. It must not: cost is
    allowed to keep rising past the ladder, only the discount stops.
    """
    convex = dict(CONCAVE_FIT, ms_per_token_sq=+1.0546633852394908e-06)
    at_probe = _project(tmp_path, input_len=8192, fit=convex)["prefill_throughput_tps"]
    beyond = _project(tmp_path, input_len=130000, fit=convex)["prefill_throughput_tps"]
    assert beyond <= at_probe, (
        "a convex curve says a longer context costs more per token; the guard "
        "must not turn that into a flat rate"
    )
