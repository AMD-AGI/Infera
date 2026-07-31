###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""infera must not append the decode radix cache under speculative decoding.

With kv-events on, ``parse_sglang_args`` appends
``--disaggregation-decode-enable-radix-cache`` to a mooncake decode leg so the
router can steer repeats to the rank holding the prefix. SGLang rejects that flag
outright when a speculative algorithm is set::

    sglang/srt/arg_groups/pd_disaggregation_hook.py
        if server_args.speculative_algorithm is not None:
            raise ValueError(
                "--disaggregation-decode-enable-radix-cache is incompatible "
                "with speculative decoding (--speculative-algorithm EAGLE)")

so the decode leg dies during argument parsing, before it loads a weight. The
gate already excludes non-mooncake backends for exactly this "SGLang rejects it"
reason; speculative decoding is the same class of exclusion.

Guarded by ``importorskip("sglang")`` because ``infera.engine.sglang.args``
imports ``sglang.srt.server_args`` at module load: runs in the SGLang container,
skips on bare dev boxes. Same pattern as ``test_disagg_allow_tcp_args.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sglang")

from infera.engine.sglang.args import parse_sglang_args  # noqa: E402

_BASE = [
    "--model-path",
    "Qwen/Qwen3-0.6B",
    "--etcd-endpoint",
    "127.0.0.1:2379",
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "mooncake",
]

_EAGLE = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]

_FLAG = "--disaggregation-decode-enable-radix-cache"


def test_appended_without_speculative_decoding():
    """The pre-existing behaviour must survive the new condition."""
    assert _FLAG in parse_sglang_args(_BASE).sglang_argv


def test_not_appended_under_eagle():
    """Fails on the pre-fix code, where the flag is appended unconditionally and
    SGLang then raises."""
    assert _FLAG not in parse_sglang_args([*_BASE, *_EAGLE]).sglang_argv


def test_operator_supplied_flag_still_raises_under_eagle():
    """We only decline to ADD it; we do not silently strip an explicit one.

    ``parse_sglang_args`` builds a ``ServerArgs``, whose ``__post_init__`` runs
    SGLang's own validation — so an operator who passes the flag deliberately
    gets SGLang's error here, at parse time, rather than a config that quietly
    differs from what they asked for.
    """
    with pytest.raises(ValueError, match="incompatible with speculative decoding"):
        parse_sglang_args([*_BASE, *_EAGLE, _FLAG])
