###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""netperf degraded-link detection.

netperf FAILs only a fully unreachable (None) link; a link degraded to a
fraction of line rate would otherwise be reported green. ``_slow_link_findings``
warns on gross relative outliers below the node median, mirroring the GPU p2p
probe's within-node comparison.
"""

from __future__ import annotations

from infera.tools.preflight.network.netperf import _slow_link_findings


def _rec(s_nic: int, c_nic: int, gb_s):
    return {
        "server": "srv",
        "s_nic": s_nic,
        "client": "cli",
        "c_nic": c_nic,
        "gb_s": gb_s,
    }


def test_no_warn_when_links_uniform():
    recs = [_rec(i, i, 40.0) for i in range(8)]
    assert _slow_link_findings(recs) == []


def test_warns_on_link_below_half_median():
    # Seven healthy ~40 GB/s links + one degraded 14 GB/s link (median 40).
    recs = [_rec(i, i, 40.0) for i in range(7)] + [_rec(7, 7, 14.0)]
    out = _slow_link_findings(recs)
    assert len(out) == 1
    assert out[0].level == "warn"
    assert "slow" in out[0].message
    assert out[0].details["gb_s"] == 14.0
    assert out[0].details["node_median"] == 40.0


def test_ignores_unreachable_links_for_median():
    # A None link is handled by the FAIL path, not here; it must not crash the
    # median computation nor be reported as "slow".
    recs = [_rec(i, i, 40.0) for i in range(6)] + [_rec(6, 6, None), _rec(7, 7, 15.0)]
    out = _slow_link_findings(recs)
    assert [f.details["gb_s"] for f in out] == [15.0]


def test_too_few_samples_no_warn():
    recs = [_rec(0, 0, 40.0), _rec(1, 1, 1.0)]
    assert _slow_link_findings(recs) == []
