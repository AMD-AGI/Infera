###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Which GPU architecture an e2e run targets.

WHAT: resolves one gfx name for the whole run. It selects the engine image
(``run_tests.sh``) and the per-case knobs the matrix applies (:mod:`.matrix`).

WHY: CI is gfx950 (MI355X) while the local SLURM fleet is gfx942 (MI300X), so
one suite has to launch two recipes without forking the case tables.

HOW: three tiers — an explicit ``INFERA_E2E_GFX_ARCH`` (an *intent*), else the
live GPU (a *fact*, and normally the answer), else :data:`DEFAULT_ARCH`.
``run_tests.sh`` and the test process probe the same node and so agree without
being told. Only an intent is exported: ``srun`` re-runs that script on the
compute node, and the host that builds must be free to probe for itself rather
than inherit a GPU-less login host's answer.

CONTEXT: tier 1 exists for the one case tier 2 cannot serve — the PD-disagg
orchestrator runs pytest on a GPU-less login host and drives containers onto
nodes SLURM picks *after* the image was chosen. Since only a declaration can
contradict the hardware, that is exactly what :func:`check_arch` verifies, and
it fails rather than auto-corrects: the image is already built by then, so
switching the launch config would just move the failure somewhere less legible.
"""

from __future__ import annotations

import functools
import os
import subprocess

# A local rocminfo answers in ~0.3s; anything slower is a wedged driver, not an answer.
_PROBE_TIMEOUT = 30.0

# Architectures the case tables know how to target; anything else is a typo.
SUPPORTED_ARCHS = ("gfx950", "gfx942")

# Where a host with no GPU to ask lands — the CI fleet.
DEFAULT_ARCH = "gfx950"

ARCH_ENV = "INFERA_E2E_GFX_ARCH"


def declared_arch() -> str:
    """The declared target arch, or ``""`` if unset. Raises on an unsupported value,
    so a typo surfaces at collection instead of silently running the default matrix."""
    declared = (os.environ.get(ARCH_ENV) or "").strip()
    if declared and declared not in SUPPORTED_ARCHS:
        raise RuntimeError(f"{ARCH_ENV}={declared!r} is not one of {', '.join(SUPPORTED_ARCHS)}")
    return declared


def parse_rocminfo(text: str) -> str | None:
    """The first GPU agent's gfx name in ``rocminfo`` output, else None. Also used for
    hosts we can only reach by shelling out (a remote SLURM node, see cluster.py)."""
    for line in (text or "").splitlines():
        name = line.strip()
        if name.startswith("Name:"):
            value = name.split(":", 1)[1].strip()
            if value.startswith("gfx"):
                return value
    return None


@functools.cache
def probe_arch() -> str | None:
    """The gfx name of the GPU on THIS host, or None when there is none to ask (login
    host, CPU-only box) — callers read None as "cannot tell".

    ``rocminfo`` rather than torch, though torch stays as the fallback: reading one
    string through torch initialises HIP, which costs ~14s and then holds a primary
    context — ~0.5 GiB of device 0 — for the life of the process. This runs in the
    pytest ORCHESTRATOR, which otherwise never touches a card, so that context would
    sit there all session inside the budget the engine under test was promised.
    """
    try:
        done = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=_PROBE_TIMEOUT)
        if done.returncode == 0:
            found = parse_rocminfo(done.stdout)
            if found:
                return found
    except (OSError, subprocess.SubprocessError):
        pass  # not installed, or a wedged driver — let torch have a go
    try:
        from infera.common.arch import gpu_arch

        return gpu_arch()
    except Exception:
        return None


def target_arch() -> str:
    """Declaration, else live GPU, else default. Itself uncached, so the env var stays
    easy to test; the probe behind it is what carries the cache."""
    return declared_arch() or probe_arch() or DEFAULT_ARCH


def check_arch() -> str | None:
    """Describes a declared-vs-actual mismatch, else None. A probed arch agrees with
    the card by definition, so only a declaration is worth checking."""
    declared = declared_arch()
    if not declared:
        return None
    actual = probe_arch()
    if actual is None or actual == declared:
        return None
    return (
        f"this e2e run was told to target {declared} but the visible GPU is {actual}. The "
        f"engine image and the launch config were both chosen for {declared}, so they would "
        f"disagree with the hardware. Unset {ARCH_ENV} to just use the card in front of you, "
        f"or run this tier on {declared}."
    )
