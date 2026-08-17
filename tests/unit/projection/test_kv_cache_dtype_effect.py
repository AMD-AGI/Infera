###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Quantising the KV cache has to change the decode step.

Decode attention is a stream out of the KV cache, so storing that cache in one
byte per element instead of two halves the traffic, and at long context that
traffic is most of the step. The projection accepted ``--kv-cache-dtype`` and
routed it to the request config, while the attention roofline read the same
field off the *model* config, where nothing ever set it. The two never met, so
every projection priced the cache at two bytes regardless of what was asked for:
an fp8 candidate scored as bf16, and a bf16 one scored as if it were fp8.

That is invisible in aggregate error against a bf16 measurement set -- which is
all of ours -- so it is asserted here on the physics instead: the saving must
exist, and it must grow with context, because the KV term grows with context
while everything else in the step does not.
"""

from __future__ import annotations

import pytest

from .conftest import project_spec


def _tpot(kv_dtype: str, ctx: int) -> float:
    return project_spec(
        model="gpt_oss_120B",
        concurrency=64,
        input_len=ctx,
        output_len=1024,
        weight_dtype="mxfp4",
        kv_cache_dtype=kv_dtype,
        tp=8,
    )["tpot_ms"]


def test_fp8_kv_is_cheaper_than_bf16():
    assert _tpot("fp8", 16384) < _tpot("bf16", 16384)


def test_the_saving_grows_with_context():
    """The KV stream scales with context; the rest of the step does not.

    So the *fraction* of the step that fp8 removes has to rise with context. A
    model that applied a flat discount would pass the test above and fail this
    one.
    """
    def saving(ctx: int) -> float:
        bf16 = _tpot("bf16", ctx)
        return (bf16 - _tpot("fp8", ctx)) / bf16 * 100.0

    short, long = saving(1024), saving(32768)
    assert long > short * 3.0, (
        f"fp8 saves {short:.2f}% at 1k context and {long:.2f}% at 32k; the KV "
        "term dominates at long context so the gap should widen sharply"
    )


def test_the_saving_is_bounded_by_halving_the_kv_stream():
    """fp8 removes half the KV bytes, never more than that, and never the step.

    Guards the other direction: a propagation bug that applied the dtype twice,
    or applied it to weights as well, would show up as a saving larger than the
    KV stream can account for.
    """
    bf16 = _tpot("bf16", 32768)
    fp8 = _tpot("fp8", 32768)
    assert 0.0 < (bf16 - fp8) / bf16 < 0.5


@pytest.mark.parametrize("ctx", [1024, 8192, 32768])
def test_bf16_is_the_default_when_unset(ctx: int):
    """Omitting the flag must price the cache as bf16, not as whatever ran last.

    The propagation writes onto the model config, which is the kind of thing
    that leaks between runs if it is ever made shared; this pins the default.
    """
    explicit = _tpot("bf16", ctx)
    implicit = project_spec(
        model="gpt_oss_120B", concurrency=64, input_len=ctx, output_len=1024,
        weight_dtype="mxfp4", tp=8,
    )["tpot_ms"]
    assert implicit == pytest.approx(explicit, rel=1e-9)
