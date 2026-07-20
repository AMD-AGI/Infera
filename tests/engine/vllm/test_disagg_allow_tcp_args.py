###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Phase 1: --disaggregation-allow-tcp plumbing for the vLLM launcher.

Guarded by importorskip("vllm") because infera.engine.vllm.args
imports vLLM's CLI parser at parse time; runs in the vLLM container,
skips on bare dev boxes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm")

from infera.common.disagg_preflight import (  # noqa: E402
    DisaggPreflightError,
    validate_vllm_transport,
)
from infera.engine.vllm.args import parse_vllm_args  # noqa: E402

_BASE = ["--model", "Qwen/Qwen3-0.6B", "--etcd-endpoint", "127.0.0.1:2379"]


def test_allow_tcp_defaults_false():
    args = parse_vllm_args(_BASE)
    assert args.disaggregation_allow_tcp is False


def test_allow_tcp_flag_sets_true():
    args = parse_vllm_args([*_BASE, "--disaggregation-allow-tcp"])
    assert args.disaggregation_allow_tcp is True


def test_disagg_producer_without_known_connector_rejected():
    # kv_producer role but no recognized connector → empty disagg_meta →
    # preflight rejects (unless allow_tcp).
    args = parse_vllm_args(
        [
            *_BASE,
            "--host",
            "10.0.0.5",
            "--kv-transfer-config",
            '{"kv_role":"kv_producer"}',
        ]
    )
    is_disagg = args.disagg_mode.value != "mixed"
    if not is_disagg or args.disagg_meta.get("protocol"):
        pytest.skip("connector mapped to a protocol; nothing to reject")
    with pytest.raises(DisaggPreflightError):
        validate_vllm_transport(
            args.disagg_meta, is_disagg=is_disagg, allow_tcp=args.disaggregation_allow_tcp
        )
