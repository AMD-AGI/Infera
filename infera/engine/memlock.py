###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Lift the RLIMIT_MEMLOCK soft limit to the hard limit before an engine starts.

The RDMA drivers register every KV buffer through the kernel's ``ib_umem_get``
first: it charges the region's pages against RLIMIT_MEMLOCK before it tries to
pin them, and only after the pin fails does the driver fall back to peer memory
for VRAM. Without CAP_IPC_LOCK (a container root remapped by Pyxis/enroot has
none) the charge is held to the soft limit, so a few concurrent 1 GiB
registrations from the transfer engine's threads exceed a limit such as the
8 GiB a Slurm launcher propagates, and those registrations fail with ENOMEM.
The engine subprocess inherits the raised limit.
"""

from __future__ import annotations

import logging
import resource

logger = logging.getLogger(__name__)


def raise_memlock_soft_limit() -> int | None:
    """Set the soft RLIMIT_MEMLOCK to the hard limit. Returns the new soft limit, or None if unchanged."""
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft == hard:
        return None
    try:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
    except (OSError, ValueError) as exc:
        logger.warning(
            "could not raise RLIMIT_MEMLOCK soft limit %s to the hard limit %s: %s", soft, hard, exc
        )
        return None
    logger.info("raised RLIMIT_MEMLOCK soft limit from %s to %s", _fmt(soft), _fmt(hard))
    return hard


def _fmt(limit: int) -> str:
    return "unlimited" if limit == resource.RLIM_INFINITY else str(limit)
