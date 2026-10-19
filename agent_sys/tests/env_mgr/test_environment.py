# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mechanism is a declared input, not a discovered condition. Design §14.1.

Spec §10: *"When no sandbox mechanism is available, the suite fails. It does not
skip."* Nobody achieves that by probing at run time — the kernel ships
``CONFIG_SECURITY_LANDLOCK=y`` with its selftests, and `rust-landlock` pins
``LANDLOCK_CRATE_TEST_ABI`` per CI runner and asserts the kernel matches it,
booting UML kernels to cover the rest.

So CI declares `ENV_MGR_TEST_MECHANISM` and, for Landlock, `ENV_MGR_TEST_ABI`,
and this asserts the machine matches. Absent the variables — a developer's
machine — the harness auto-detects and runs. **The hard failure is CI-side**,
exactly as `rust-landlock` arranges it.
"""

from __future__ import annotations

import os

import pytest

from env_mgr.isolation.probe import Availability

from .conftest import ABI_VAR, MECHANISM_VAR


def test_declared_mechanism_is_the_one_present(availability: Availability) -> None:
    declared = os.environ.get(MECHANISM_VAR)
    if declared is None:
        pytest.skip(
            f"{MECHANISM_VAR} is unset: a developer's machine auto-detects. "
            f"This skip is for environmental variation orthogonal to the property "
            f"under test — the property itself is never skipped, and the session "
            f"fixture has already failed if no mechanism exists at all."
        )
    if declared == "bwrap":
        assert availability.bwrap, f"{MECHANISM_VAR}=bwrap but bwrap is not on PATH"
    elif declared == "landlock":
        assert availability.landlock_abi, f"{MECHANISM_VAR}=landlock but Landlock is unavailable"
    else:
        pytest.fail(f"{MECHANISM_VAR}={declared!r} is not a mechanism this chain has")


def test_declared_abi_is_the_one_present(availability: Availability) -> None:
    declared = os.environ.get(ABI_VAR)
    if declared is None:
        pytest.skip(f"{ABI_VAR} is unset")
    assert availability.landlock_abi == int(declared), (
        f"{ABI_VAR}={declared} but this kernel reports {availability.landlock_abi}"
    )


def test_the_gate_lives_in_one_place() -> None:
    """bubblewrap's ``BWRAP_MUST_WORK`` is the right shape and leaks: the
    variable appears in exactly one file, and the Python half of its own suite
    skips unconditionally — passing green in the CI job that sets it. A single
    session-scoped fixture is the whole defence against repeating that, so this
    asserts there is exactly one."""
    from pathlib import Path

    here = Path(__file__).parent
    # A top-level `def`, so the mention of the name inside this test does not
    # count itself.
    definitions = [p.name for p in here.glob("*.py") if "\ndef availability(" in p.read_text()]
    assert definitions == ["conftest.py"], (
        f"the mechanism gate is defined in {definitions}; it must be one fixture "
        f"that every confinement test traverses"
    )
