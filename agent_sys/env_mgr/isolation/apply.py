# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The chain, and the two-tier gate. Design §4.5.

The sequence *is* the decision, so it appears as a body rather than as a
signature and a sentence.
"""

from __future__ import annotations

import threading

from env_mgr.isolation import bwrap as _bwrap
from env_mgr.isolation import landlock
from env_mgr.isolation.landlock import Enforced
from env_mgr.isolation.policy import Policy
from env_mgr.isolation.probe import Availability, select
from env_mgr.protocols import Confinement, NoConfinement, Tier

__all__ = ["apply", "bwrap_argv", "confinement_for"]


def confinement_for(mechanism: str, abi: int | None) -> Confinement:
    """What the chain achieved, reported rather than assumed.

    bubblewrap also isolates network and PID; Landlock below ABI 4 isolates
    neither and cannot touch the network at all. Falling from rung 1 to rung 2
    silently drops both unless something says so.
    """
    if mechanism == "bwrap":
        return Confinement("bwrap", True, True, True, None)
    return Confinement("landlock", True, bool(abi and abi >= 4), False, abi)


def bwrap_argv(policy: Policy, av: Availability, command: tuple[str, ...] = ()) -> list[str]:
    """Rung 1 is applied by ``exec``, not in this process, so it is an argv."""
    if not av.bwrap:
        raise NoConfinement("bwrap was selected but is not present")
    return _bwrap.argv(policy, bwrap=av.bwrap, command=command)


def apply(policy: Policy, av: Availability, *, tier: Tier) -> Confinement:
    """Confine **this process** and every descendant. Irreversible.

    `NoConfinement` is never caught inside this package: `prepare` lets it
    propagate and the task does not start. The survey's most useful negative
    result is that the project closest to this one decided the other way in the
    weakest available form — Codex's sandbox tests begin with a bare early
    return, so they report green on a machine with no sandbox at all.
    """
    mechanism = select(av)  # raises NoConfinement
    if mechanism == "bwrap":
        # Applied by exec, by whoever runs `bwrap_argv`. Nothing to do here, and
        # nothing to claim: the argv carries the policy.
        return confinement_for("bwrap", None)

    if threading.active_count() > 1:
        # Below ABI 8 `restrict_self` restricts only the calling thread, and
        # `all_threads()` does not exist to ask for more. Measured, with the
        # guard removed, from a worker thread (p3_confine_from_a_thread.py):
        #
        #   the worker thread, writing outside   denied
        #   the MAIN thread, writing outside     WRITABLE
        #   a subprocess of the worker           denied
        #
        # Two separate problems, and the first is the one that stops a caller
        # working rather than merely misreporting. **The thread that applied it
        # is itself confined, irreversibly** — so a runner thread that must
        # record an outcome afterwards can no longer do so. And the process is
        # not confined while the status says the filesystem is, which is true of
        # the executor subprocess and false of everything else.
        raise NoConfinement(
            f"{threading.active_count()} threads are running. Landlock below ABI 8 "
            f"restricts only the calling thread, so this would (a) confine this "
            f"thread irreversibly, including against work it still has to do, and "
            f"(b) leave every sibling thread unrestricted while reporting enforced. "
            f"Confinement belongs in the process that becomes the executor, applied "
            f"between fork and exec"
        )

    ruleset = landlock.build(policy)  # (1) every entry, or an error
    status = landlock.restrict(ruleset)  # (2) irreversible from here
    if status.enforced is Enforced.NOTHING:  # (3) the hard gate, both tiers
        raise NoConfinement("ruleset enforced nothing")
    if tier is Tier.STRICT and status.enforced is not Enforced.FULLY:  # (4)
        raise NoConfinement(f"partial enforcement: dropped {list(status.dropped)}")
    return confinement_for("landlock", status.abi)
