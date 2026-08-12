###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Every engine launcher must apply the ROCm RDMA env defaults itself.

The helpers in ``rocm_rdma_env`` are set-if-unset, so a launcher that never calls
them raises nothing and starts normally — the cost lands at run time as a silent
TCP fallback, or KV sent to the public NIC. The ATOM launcher shipped that way:
``apply_kv_host_ip_default`` names ``ATOM_HOST_IP`` and ATOM's ``get_ip()`` in its
own docstring, yet only the sglang and vLLM launchers ever called it, so on ATOM
the whole set only arrived if a launch script happened to pass every var by hand.

Read as source rather than imported: these modules pull in their engine at import
time, which is not installed in the unit environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_REQUIRED = {"apply_rocm_rdma_env_defaults", "apply_kv_host_ip_default"}


def _called_names(engine: str) -> set[str]:
    src = (_REPO / "infera" / "engine" / engine / "__main__.py").read_text()
    return {
        node.func.id
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


@pytest.mark.parametrize("engine", ["atom", "sglang", "vllm"])
def test_launcher_applies_rdma_env_defaults(engine):
    missing = _REQUIRED - _called_names(engine)
    assert not missing, (
        f"infera.engine.{engine}.__main__ never calls {sorted(missing)}; "
        "its PD KV transfer would fall back to TCP or bind the wrong interface"
    )
