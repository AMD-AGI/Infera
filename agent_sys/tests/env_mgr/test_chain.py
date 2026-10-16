# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Criteria 8 and 9 — no sandbox means no start, and the chain degrades in order.

**Criterion 9 cannot be satisfied as written, and design §14.6 says so rather
than approximating it.** No machine runs all three branches: ``bwrap`` is absent
here so rung 1 cannot be exercised, and there is no ordinary way to make a
Landlock-capable kernel look incapable so rung 3 cannot be exercised wherever
rung 2 works. The composition is tested with an injected `Availability`; the
three-way degradation on one host is not, and cannot be.
"""

from __future__ import annotations

import pytest

from env_mgr.isolation.apply import apply, bwrap_argv, confinement_for
from env_mgr.isolation.policy import DEFAULT_SYSTEM_SET, Granted, Mode, Policy
from env_mgr.isolation.probe import Availability, probe, select
from env_mgr.protocols import NoConfinement, Tier

NOTHING = Availability(bwrap=None, landlock_abi=None)
ONLY_LANDLOCK = Availability(bwrap=None, landlock_abi=3)
BOTH = Availability(bwrap="/usr/bin/bwrap", landlock_abi=3)
ONLY_BWRAP = Availability(bwrap="/usr/bin/bwrap", landlock_abi=None)


# ----------------------------------------------------------- criterion 9


def test_prefers_bwrap() -> None:
    assert select(BOTH) == "bwrap"
    assert select(ONLY_BWRAP) == "bwrap"


def test_falls_back_to_landlock() -> None:
    assert select(ONLY_LANDLOCK) == "landlock"


def test_refuses_when_neither() -> None:
    with pytest.raises(NoConfinement):
        select(NOTHING)


def test_end_to_end_against_the_declared_mechanism(availability: Availability) -> None:
    """The one branch this machine can run, whichever it is.

    This is the half of criterion 9 that is not a unit test, and the session
    fixture is what makes it fail rather than skip when the machine has nothing.
    """
    assert select(availability) in ("bwrap", "landlock")


# ----------------------------------------------------------- criterion 8


def test_no_mechanism_refuses_to_start() -> None:
    """Not "warn and continue". An agent started without confinement is an agent
    running with the operator's full privileges while the system reports it is
    sandboxed."""
    with pytest.raises(NoConfinement):
        apply(Policy(DEFAULT_SYSTEM_SET), NOTHING, tier=Tier.PRODUCTION)


def test_refusal_names_the_reason() -> None:
    with pytest.raises(NoConfinement) as excinfo:
        apply(Policy(DEFAULT_SYSTEM_SET), NOTHING, tier=Tier.PRODUCTION)
    message = str(excinfo.value)
    assert "bwrap" in message and "Landlock" in message
    assert "does not start" in message


def test_nothing_in_the_package_catches_noconfinement() -> None:
    """Criterion 8 is only a rule if nothing converts the refusal into a warning.

    Checked structurally, because the failure mode is a ``try`` somebody adds
    later in good faith. The survey's most useful negative result is that the
    project closest to this one decided the other way, in the weakest available
    form: Codex's sandbox tests begin with a bare early return and report green
    on a machine with no sandbox at all.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "env_mgr"
    for source in root.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            names = ast.unparse(node.type)
            assert "NoConfinement" not in names, (
                f"{source.name} catches NoConfinement; the task must not start"
            )


# --------------------------------------- the two rungs are not the same thing


def test_the_chain_degrades_in_properties_not_only_preference() -> None:
    """Falling from rung 1 to rung 2 silently drops network and PID isolation
    unless something says so. `Confinement` is what says so."""
    rung1 = confinement_for("bwrap", None)
    rung2 = confinement_for("landlock", 3)
    assert (rung1.network, rung1.pid) == (True, True)
    assert (rung2.network, rung2.pid) == (False, False)
    assert confinement_for("landlock", 4).network is True
    assert rung1.filesystem and rung2.filesystem


def test_bwrap_argv_carries_the_policy() -> None:
    """``bwrap`` is **absent on this machine**, so rung 1 is tested as the argv
    it produces. The two unshares are where its extra properties come from."""
    policy = Policy(
        (
            Granted("/usr", Mode.READ_EXEC),
            Granted("/zone", Mode.READ_WRITE),
            Granted("/lib64", Mode.READ_EXEC, optional=True),
        )
    )
    argv = bwrap_argv(policy, BOTH, ("true",))
    assert argv[0] == "/usr/bin/bwrap"
    assert "--unshare-net" in argv and "--unshare-pid" in argv
    assert argv[argv.index("--ro-bind") + 1 : argv.index("--ro-bind") + 3] == ["/usr", "/usr"]
    assert "--bind" in argv and argv[argv.index("--bind") + 1] == "/zone"
    # `--ro-bind-try` is bubblewrap's own distinction, used in its own tests for
    # exactly the paths that legitimately do not exist on some hosts.
    assert "--ro-bind-try" in argv
    assert argv[-2:] == ["--", "true"]


def test_probe_reports_this_machine() -> None:
    av = probe()
    assert av.bwrap is None or isinstance(av.bwrap, str)
    assert av.landlock_abi is None or av.landlock_abi >= 1
