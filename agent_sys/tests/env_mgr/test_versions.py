# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
from env_mgr.versions import constraints_conflict, satisfies


def test_no_constraint_is_always_satisfied():
    assert satisfies("1.0.0", None) is True
    assert satisfies(None, None) is True


def test_missing_actual_fails_a_constraint():
    assert satisfies(None, ">=1.0") is False


def test_specifier_constraint():
    assert satisfies("1.2.0", ">=1.1") is True
    assert satisfies("1.0.0", ">=1.1") is False


def test_malformed_actual_version_fails():
    assert satisfies("not-a-version", ">=1.0") is False


def test_malformed_constraint_fails_instead_of_raising():
    assert satisfies("1.2.0", "v1.0.0") is False
    assert satisfies("1.2.0", "not-a-constraint") is False


def test_bare_version_means_gte():
    assert satisfies("1.2.0", "1.1.0") is True
    assert satisfies("1.0.0", "1.1.0") is False


def test_conflict_detection():
    assert constraints_conflict(">=0.5", ">=0.6") is True
    assert constraints_conflict(">=0.5", ">=0.5") is False
    assert constraints_conflict(None, ">=0.5") is False
    assert constraints_conflict(">=0.5", None) is False


def test_conflict_semantic_equivalence_is_not_a_conflict():
    # bare version normalizes to >=, so these are equal, not conflicts
    assert constraints_conflict("0.5", ">=0.5") is False
    assert constraints_conflict(">=0.5", ">=0.5.0") is False


def test_conflict_unparseable_falls_back_to_text():
    assert constraints_conflict("garbage", "other-garbage") is True
    assert constraints_conflict("garbage", "garbage") is False
