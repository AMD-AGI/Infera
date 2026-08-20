# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
from env_mgr.outcome import Outcome, status_from, worst_level


def test_outcome_defaults_empty_details():
    o = Outcome("ok", "present")
    assert o.details == {}


def test_status_from_empty_is_ok():
    assert status_from([]) == "OK"


def test_status_from_warn_beats_ok_info():
    outs = [Outcome("ok", "a"), Outcome("info", "b"), Outcome("warn", "c")]
    assert status_from(outs) == "WARN"


def test_status_from_fail_beats_all():
    outs = [Outcome("warn", "a"), Outcome("fail", "b")]
    assert status_from(outs) == "FAIL"


def test_worst_level_empty_is_ok():
    assert worst_level([]) == "ok"


def test_worst_level_picks_highest():
    assert worst_level([Outcome("ok", "a"), Outcome("fail", "b")]) == "fail"
