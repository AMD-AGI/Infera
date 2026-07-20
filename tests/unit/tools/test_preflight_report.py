###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Report rendering for the preflight cluster overview.

Regression coverage for the overview cell: the Mooncake column must surface a
failing per-GPU VRAM variant (``rdma-gpu{g}``) -- the ionic bare-VRAM RDMA
registration path the whole check exists to catch. Before the fix those
variants matched no overview column and were silently dropped, so a failing GPU
KV transfer stayed green in the at-a-glance overview.
"""

from __future__ import annotations

from infera.tools.preflight.finding import Finding
from infera.tools.preflight.report import _mooncake_group, _overview_cell, render_html


def test_mooncake_group_classifies_gpu_variants_as_rdma():
    assert _mooncake_group("host-a -> host-b rdma") == "rdma"
    assert _mooncake_group("host-a -> host-b rdma-gpu0") == "rdma"
    assert _mooncake_group("host-a -> host-b rdma-gpu7") == "rdma"
    assert _mooncake_group("host-a -> host-b rdma-default") == "rdma-default"
    assert _mooncake_group("host-a -> host-b tcp") == "tcp"
    assert _mooncake_group("host-a -> host-b something-else") is None


def _mooncake_findings(gpu_level: str) -> list[Finding]:
    return [
        Finding("info", "src -> dst rdma", {"GB/s": 40}),
        Finding("warn", "src -> dst rdma-default", {"reason": "link-local"}),
        Finding("info", "src -> dst tcp", {"GB/s": 5}),
        Finding(gpu_level, "src -> dst rdma-gpu3", {"reason": "register errno 14 (EFAULT)"}),
    ]


def test_overview_cell_rolls_gpu_failure_into_rdma_badge():
    cell = _overview_cell("mooncake", _mooncake_findings("fail"), "anchor")
    # The RDMA column reflects the worst of the CPU rdma + all per-GPU rdma
    # variants, so a failing rdma-gpu turns the RDMA badge FAIL.
    assert 'RDMA: <span class="badge fail">FAIL</span>' in cell
    assert 'RDMA(default): <span class="badge warn">WARN</span>' in cell
    assert 'TCP: <span class="badge ok">OK</span>' in cell


def test_overview_cell_rdma_ok_when_all_rdma_pass():
    cell = _overview_cell("mooncake", _mooncake_findings("info"), "anchor")
    assert 'RDMA: <span class="badge ok">OK</span>' in cell


def test_render_html_overview_shows_gpu_failure():
    results = [
        {
            "host": "node-a",
            "image": "img",
            "generated": "t",
            "sections": {"mooncake": _mooncake_findings("fail")},
        }
    ]
    html = render_html(results)
    # A FAIL badge must appear (the overview, not only the detail table), and the
    # failing variant must be named somewhere in the report.
    assert 'class="badge fail"' in html
    assert "rdma-gpu3" in html
