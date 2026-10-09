# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Design §4.4(c) and §14.3 — restriction must precede any thread start.

**A live hazard on this kernel.** ``landlock_restrict_self()`` restricts only
the *calling thread* below ABI 8; ``all_threads()`` arrives at ABI 8 and does not
exist at ABI 3. A best-effort implementation therefore leaves sibling threads
unrestricted **while the status still reports enforced** — the failure is silent,
which is the whole reason this is a test and not a comment.
"""

from __future__ import annotations

import errno
import os
import threading
from pathlib import Path

import pytest

from env_mgr.isolation.policy import Granted, Mode
from env_mgr.protocols import NoConfinement, Tier

from .conftest import attempt, base_policy

pytestmark = pytest.mark.usefixtures("landlock_abi")


def test_restriction_precedes_any_thread(tmp_path: Path) -> None:
    """`apply` refuses when a sibling thread is already running.

    Below ABI 8 there is nothing to ask the kernel for, so the only correct
    answer is to refuse — and `prepare` restricts at step 7, before the executor
    exists, which is what makes the refusal never fire in production.
    """
    from env_mgr.isolation import apply as _apply
    from env_mgr.isolation.probe import probe

    zone = tmp_path / "zone"
    zone.mkdir()
    policy = base_policy(Granted(str(zone), Mode.READ_WRITE))
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")

    stop = threading.Event()
    sibling = threading.Thread(target=stop.wait, daemon=True)

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never reports coverage
        os.close(read_fd)
        sibling.start()
        try:
            _apply.apply(policy, probe(), tier=Tier.PRODUCTION)
            os.write(write_fd, b"applied")
        except NoConfinement as e:
            os.write(write_fd, f"refused: {e}".encode())
        finally:
            stop.set()
            os.close(write_fd)
        os._exit(0)

    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as fh:
        message = fh.read().decode()
    os.waitpid(pid, 0)
    assert message.startswith("refused:"), (
        "confinement was applied with a sibling thread running; below ABI 8 that "
        "leaves the sibling unrestricted while the status reports enforced"
    )
    assert "restricts only the calling thread" in message
    # The refusal names **both** consequences, because the first is the one that
    # stops a caller working and the message used to state only the second.
    # Measured with the guard removed (p3_confine_from_a_thread.py): the thread
    # that applies it can no longer write outside the zone — so a runner thread
    # that must record an outcome afterwards is permanently crippled — while the
    # main thread stays writable and the status would report enforced.
    assert "irreversibly" in message
    assert "between fork and exec" in message


def test_single_threaded_confinement_still_works(tmp_path: Path) -> None:
    """The positive control for the test above: the same policy, no sibling."""
    from .conftest import run_confined

    zone = tmp_path / "zone"
    zone.mkdir()
    (zone / "mine.txt").write_text("mine")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    policy = base_policy(Granted(str(zone), Mode.READ_WRITE))

    inside, out = run_confined(
        policy, lambda: (attempt(str(zone / "mine.txt"), "r"), attempt(str(outside), "r"))
    )
    assert inside == 0
    assert out == errno.EACCES
