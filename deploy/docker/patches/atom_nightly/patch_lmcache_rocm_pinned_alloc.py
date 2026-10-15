#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Back LMCache's NUMA CPU pool with exact-size hipHostMalloc (GTT) on ROCm.

With ``LMCACHE_NUMA_MODE`` set, LMCache allocates its CPU pool with
``alloc_pinned_numa_ptr``: mmap + mbind + hipHostRegister. On ROCm that is a
KFD USERPTR mapping whose pages the kernel may still migrate (compaction,
move_pages), and every migration evicts all GPU queues of the process until
KFD revalidates the range: multi-second stalls per event on a 256 GiB pool.

Same approach as sgl-project/sglang#42553 for HiCache: when the engine runs
with ``HSA_USERPTR_FOR_PAGED_MEM=0`` (set before HIP initializes, i.e. in the
process environment), use LMCache's own ``alloc_pinned_ptr`` /
``free_pinned_ptr`` pair instead: hipHostMalloc(size, 0) / hipHostFree, exact
size, which with =0 is driver-owned GTT that cannot migrate. Hugepage pools are
left unchanged. Without =0 the pool keeps the upstream path and a warning says
so. GTT lands on the NUMA node of the first GPU the process can open, not
numa_id.

Written against lmcache 0.5.6.dev139 (f22dec28) in
rocm/atom-dev:nightly_202610131505. Drop it once LMCache's non-MP CPU pool
stops using hipHostRegister on ROCm. Strict: see anchor_patch.py.
"""

import sys

from anchor_patch import apply

HELPERS = '''\
def _rocm_gtt_pool() -> bool:
    """ROCm with HSA_USERPTR_FOR_PAGED_MEM=0: hipHostMalloc returns GTT memory."""
    return (
        torch.version.hip is not None
        and os.environ.get("HSA_USERPTR_FOR_PAGED_MEM") == "0"
    )


def _log_rocm_numa_pool(use_hugepages: bool) -> None:
    """Say which memory backs a NUMA CPU pool on ROCm (alloc path only)."""
    if torch.version.hip is None or use_hugepages:
        return
    if _rocm_gtt_pool():
        logger.info(
            "ROCm: LMCache CPU pool uses hipHostMalloc (GTT, not migratable); it lands "
            "on the NUMA node of the first GPU the process can open, not numa_id."
        )
    else:
        logger.warning(
            "ROCm: LMCache NUMA CPU pool uses hipHostRegister (KFD USERPTR); a page "
            "migration stalls every GPU queue. Set HSA_USERPTR_FOR_PAGED_MEM=0 in the "
            "engine environment to use GTT memory instead."
        )


'''

PATCH = {
    "v1/memory_management.py": (
        "def _rocm_gtt_pool() -> bool:",
        [
            # The helpers use these module-level names; require them.
            ("\nimport os\n", "\nimport os\n"),
            ("\nimport torch\n", "\nimport torch\n"),
            ("logger = init_logger(__name__)\n", "logger = init_logger(__name__)\n"),
            ("def _resolve_pinned_alloc_free(\n", HELPERS + "def _resolve_pinned_alloc_free(\n"),
            (
                "    elif numa_mapping:\n",
                "    elif numa_mapping and not use_hugepages and _rocm_gtt_pool():\n"
                "        # Exact-size hipHostMalloc, freed by hipHostFree (sglang#42553).\n"
                "        return PinnedAllocFree(\n"
                "            alloc_fn=device_ops.alloc_pinned_ptr,\n"
                "            alloc_args=(size, 0),\n"
                "            free_fn=device_ops.free_pinned_ptr,\n"
                "            free_args=(),\n"
                "        )\n"
                "    elif numa_mapping:\n",
            ),
            (
                "    try:\n        ptr = resolved.alloc()\n",
                "    if numa_mapping and not shm_name:\n"
                "        _log_rocm_numa_pool(use_hugepages)\n"
                "    try:\n        ptr = resolved.alloc()\n",
            ),
        ],
    ),
}

if __name__ == "__main__":
    sys.exit(apply("lmcache-patch rocm-pinned-alloc", PATCH, package="lmcache"))
