###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""An anchor is a measurement of one attention layout, and cannot stand in for
another.

Data-parallel attention is not a tuning knob layered over the same machine. A
rank stops holding the whole batch at ``tp`` heads and starts holding ``1/dp``
of the requests at ``tp/dp`` heads, and one of the two per-layer all-reduces
disappears with it. Both changes are structural, and both are already in the
analytical model -- so the restore's ``sim(target)/sim(bench)`` ratio carries
them for the same reason it carries a change of TP.

What it could not do was notice. ``attn_dp`` was absent from the transport axes,
absent from the artifact, and absent from the restore trigger, so an anchor
harvested at ``attn_dp=1`` and projected at ``attn_dp=8`` took the
already-returned-early path: no transport, no warning, the raw measured step
reported as a calibrated one. That is the case these tests pin, because it is
the default way MLA models are served and therefore the case that mattered.
"""

from __future__ import annotations

import pytest

from infera.projection.core.projection.inference_projection.search import regime


def _read_layout(server_args, *, tp):
    """Imported per-call so the rest of this module still collects against a
    build that does not have the reader yet."""
    from infera.projection.core.projection.inference_projection.performance import (
        _attention_dp_from_server_args,
    )

    return _attention_dp_from_server_args(server_args, tp=tp)


# --------------------------------------------------------------------------
# Reading the layout off the run that produced the anchor
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "server_args,expected",
    [
        # SGLang / ATOM: gated behind the flag, sized by --dp-size, and sized to
        # the tensor-parallel group when left implicit -- which is how the MLA
        # recipes are actually served.
        ("--enable-dp-attention --dp-size 8", 8),
        ("--enable-dp-attention --dp-size=4", 4),
        ("--enable-dp-attention", 8),
        # Underscores are the same flag; an engine that spells it either way
        # must not read as "no DP".
        ("--enable_dp_attention", 8),
        # vLLM drives the same layout from --data-parallel-size.
        ("--data-parallel-size 4 --enable-expert-parallel", 4),
        # Flags present and not asking for it: the feature is opt-in, so this
        # is a statement that the run was not data-parallel.
        ("--tp 8 --attention-backend TRITON_MLA", 1),
        ("--disable-dp-attention", 1),
    ],
)
def test_the_layout_is_read_from_the_flags_the_anchor_recorded(server_args, expected):
    assert _read_layout(server_args, tp=8) == expected


def test_no_flag_string_at_all_is_unknown_rather_than_one():
    """The one case that may not be inferred.

    Silence *within* a flag string means the opt-in feature was not asked for.
    Having no flag string is different: the artifact predates this tracking and
    may have run either way. Collapsing the two would turn every pre-tracking
    anchor into a confident claim of ``attn_dp=1``, which is the same silent
    reuse by another route.
    """
    assert _read_layout("", tp=8) is None
    assert _read_layout(None, tp=8) is None


# --------------------------------------------------------------------------
# The axis exists, and is transportable rather than regime-defining
# --------------------------------------------------------------------------


def test_the_layout_is_a_transport_axis():
    """Transportable, not regime-defining, because the model describes it.

    Speculation is regime-defining because acceptance cannot be derived from a
    non-speculative anchor. Attention-DP is the opposite case: the per-rank
    batch and the collective count both follow from the degree, so the ratio
    reconstructs it and a separate harvest is not required.
    """
    assert "attn_dp" in regime.TRANSPORT_AXES
    assert "attn_dp" not in regime.REGIME_AXES


def test_every_recipe_source_reports_the_layout():
    """An axis only constrains matching if all three adapters populate it.

    The store compares an artifact's recipe against the target config's. If
    either side leaves the axis out, it compares equal by absence and the
    mismatch is invisible again.
    """
    from_meta = regime.recipe_from_meta({"attention_data_parallel_size": 8}, model="m")
    assert from_meta["attn_dp"] == 8

    cfg_recipe = regime.recipe_from_inference_config(
        _inference_config(tp=8, attn_dp=8, disaggregate=False)
    )
    assert cfg_recipe["attn_dp"] == 8


def test_an_artifact_predating_the_tracking_stays_unknown():
    """Unknown is skipped by ``regime_distance``; a guessed 1 would not be.

    An anchor that never recorded its layout must not claim one. Reading the
    absent key as 1 would have it match a non-DP target exactly and mismatch a
    DP target -- both asserted from nothing.
    """
    assert regime.recipe_from_meta({}, model="m")["attn_dp"] is None


def test_the_harvester_records_the_layout_it_ran():
    """The transport is only reachable if the artifact says what it measured, so
    the harvester's key and the projector's reader have to stay spelled the same."""
    from infera.projection.core.projection.inference_projection import (
        benchmark_serving,
        benchmark_vllm,
    )

    for mod in (benchmark_serving, benchmark_vllm):
        with open(mod.__file__) as fh:
            assert '"attention_data_parallel_size"' in fh.read(), (
                f"{mod.__name__} must record the attention layout in anchor meta"
            )


# --------------------------------------------------------------------------
# A layout change triggers the restore that transports it
# --------------------------------------------------------------------------


def _inference_config(*, tp: int, attn_dp: int, disaggregate: bool):
    from infera.projection.core.projection.training_config import (
        DisaggregationConfig,
        InferenceConfig,
        InferenceRequestConfig,
        ModelConfig,
        ModelParallelConfig,
    )

    return InferenceConfig(
        model_config=ModelConfig(
            num_layers=4,
            hidden_size=512,
            num_attention_heads=8,
        ),
        request_config=InferenceRequestConfig(batch_size=8, input_seq_len=128, output_seq_len=32),
        model_parallel_config=ModelParallelConfig(
            tensor_model_parallel_size=tp,
            attention_data_parallel_size=attn_dp,
        ),
        disaggregation_config=DisaggregationConfig(enabled=disaggregate),
    )


class _Restorable:
    """``_setup_restoration``'s decision, over just the state it reads.

    The decision is what these tests are about: whether a layout change is
    recognised as something to transport. Building a real projector would pull
    in a GEMM backend and a simulator tree to answer a question that is settled
    before either is touched.
    """

    def __init__(self, *, bench_tp, bench_attn_dp, tgt_tp, tgt_attn_dp):
        from infera.projection.core.projection.inference_projection.performance import (
            InferencePerformanceProjector,
        )

        self.cfg = _inference_config(tp=tgt_tp, attn_dp=tgt_attn_dp, disaggregate=False)
        self._bench_tp, self._bench_ep, self._bench_pp = bench_tp, 1, 1
        self._bench_attn_dp = bench_attn_dp
        self._restore = False
        self._restore_layout_moved = False
        self._bench_attn_dp_eff = 1
        self._scaling_mode = "off"  # skip the origami build; only the trigger matters
        self._cc = self.cfg.collective_config
        self._moe_imbalance = 1.0
        self._view = None
        self._setup = InferencePerformanceProjector._setup_restoration.__get__(self)

    def _moe_imbalance_for_ep(self, ep):
        return 1.0

    def decide(self):
        """The recorded decision, with the refusal tolerated.

        These stubs do not build the simulator trees, so a transportable
        mismatch reaches the refusal below -- which is correct behaviour and is
        asserted on its own further down. The decision is recorded before it is
        raised, so it survives to be inspected here.
        """
        import contextlib

        with contextlib.suppress(ValueError):
            self._setup()
        return self


def test_a_layout_change_alone_triggers_the_restore():
    """The defect, stated as the branch it took.

    With TP, EP and PP all matching, the trigger returned False and
    ``_setup_restoration`` returned early -- so the measured step was used
    verbatim at a layout it was never measured at.
    """
    p = _Restorable(bench_tp=8, bench_attn_dp=1, tgt_tp=8, tgt_attn_dp=8).decide()

    assert p._restore, "attn_dp 1 -> 8 must be transported, not reused"
    assert p._restore_layout_moved


def test_a_matching_layout_still_costs_nothing():
    """The fix must not turn every colocated projection into a restore."""
    p = _Restorable(bench_tp=8, bench_attn_dp=8, tgt_tp=8, tgt_attn_dp=8).decide()
    assert not p._restore
    assert not p._restore_layout_moved


def test_the_bench_side_model_is_built_at_the_anchors_own_layout():
    """The ratio is only meaningful if its denominator is the anchor's machine.

    ``sim(target)/sim(bench)`` transports the change by evaluating the model
    twice. Leaving the bench side at the *target* layout -- which is what
    ``replace(mp, ...)`` did before the layout was added to it -- makes the two
    sides identical in attention and the whole ratio cancels to 1, which is the
    silent reuse wearing a restore's clothes.

    Asserted on the collective models because that is where the layout has a
    consequence: attention-DP drops one of the two per-layer all-reduces, so the
    two sides have to disagree about the comm they charge.
    """
    p = _Restorable(bench_tp=8, bench_attn_dp=1, tgt_tp=8, tgt_attn_dp=8).decide()

    assert p._bench_attn_dp_eff == 1
    assert p._comm_bench.attn_dp == 1, "bench side must model the anchor's layout"
    assert p._comm_tgt.attn_dp == 8, "target side must model the deployment's layout"


def test_a_degree_wider_than_the_group_it_splits_is_clamped():
    """Attention-DP subdivides the tensor-parallel group, so it cannot exceed it.

    A recorded degree wider than the bench TP is a malformed artifact; it is
    clamped rather than passed into a view whose attention would then shard to
    ``tp // dp == 0`` heads.
    """
    p = _Restorable(bench_tp=4, bench_attn_dp=16, tgt_tp=8, tgt_attn_dp=8).decide()
    assert p._bench_attn_dp_eff <= 4


def test_an_unrecorded_layout_against_a_dp_target_is_reported(capsys):
    """Neither available guess is free, so it is said out loud instead.

    Calling the anchor non-DP applies a ratio the measurement may already
    contain; calling it a match is the silent reuse being fixed. So the number
    is left as measured and the user is told the anchor cannot be checked --
    which is the one thing the old behaviour never did.
    """
    p = _Restorable(bench_tp=8, bench_attn_dp=None, tgt_tp=8, tgt_attn_dp=8).decide()
    assert not p._restore, "an unrecorded layout is reported, not transported from a guess"

    out = capsys.readouterr().out.lower()
    assert "attention" in out and "harvest" in out, (
        f"an uncheckable anchor must not be used in silence; got {out!r}"
    )


def test_a_layout_change_is_refused_when_the_ratio_is_unavailable():
    """The refusal that keeps the fallback laws honest.

    Every other restore law here is a function of the GPU count -- the measured
    TP fit, the ideal ``TP^-1`` sharding -- and attention-DP does not change the
    GPU count. So with the simulator ratio gone they all leave the layout change
    unpriced and report the result as transported regardless. Refusing is the
    only answer that is not a wrong number.
    """
    p = _Restorable(bench_tp=8, bench_attn_dp=1, tgt_tp=8, tgt_attn_dp=8)

    with pytest.raises(ValueError, match="attention-DP"):
        p._setup()


def test_the_refusal_does_not_care_whether_tp_also_moved():
    """A TP law cannot absorb a layout change on the way past.

    When TP moves as well, the fallback has something to say and returns a
    plausible non-unit ratio -- but it is a ratio about GPU count, and the
    layout change rides along unpriced inside a number that now looks restored.
    That is harder to notice than the layout-only case, not easier, so it is
    refused on the same terms.
    """
    p = _Restorable(bench_tp=4, bench_attn_dp=1, tgt_tp=8, tgt_attn_dp=8)

    with pytest.raises(ValueError, match="attention-DP"):
        p._setup()


def test_the_refusal_can_be_overridden_deliberately(monkeypatch, capsys):
    """An escape hatch, on the same terms as ``INFERASIM_ALLOW_FOREIGN_ANCHOR``.

    Refusing is right by default, but it must not be the only option: someone
    comparing against the old behaviour, or who wants the uncorrected number
    knowingly, should not have to patch the projector. The override still says
    what it is giving back.
    """
    monkeypatch.setenv("INFERASIM_ALLOW_LAYOUT_MISMATCH", "1")
    p = _Restorable(bench_tp=8, bench_attn_dp=1, tgt_tp=8, tgt_attn_dp=8)
    p._setup()

    assert "not being transported" in capsys.readouterr().out.lower()
