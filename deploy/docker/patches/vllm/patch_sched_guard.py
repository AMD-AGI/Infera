###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Apply the vLLM scheduler KV-xfer-finished guard inside an engine container.

Workaround for AMD-AGI/Infera#69: under concurrent PD load the decode EngineCore
dies on `assert req_id in self.requests` in scheduler._update_from_kv_xfer_finished
(a KV-transfer-finished event arrives for a req already removed). This guard skips
the stale event instead of asserting, so the engine stays up. It does NOT fix the
underlying correctness race (the racing request still loses its KV transfer) — so it
is applied for THROUGHPUT (perf) runs, where correctness is off, not for correctness
tests (which should surface #69). Idempotent. Run inside the container:
    docker exec <ctr> python3 <repo>/pd/patch_sched_guard.py
"""

import os
import sys

import vllm

f = os.path.join(os.path.dirname(vllm.__file__), "v1/core/sched/scheduler.py")
src = open(f).read()

if "KV xfer finished for unknown req" in src:
    print("sched-guard: already patched")
    sys.exit(0)

reps = [
    (
        '            logger.debug("Finished recving KV transfer for request %s", req_id)\n'
        "            assert req_id in self.requests\n",
        '            logger.debug("Finished recving KV transfer for request %s", req_id)\n'
        "            if req_id not in self.requests:\n"
        '                logger.warning("KV xfer finished for unknown req %s "\n'
        '                               "(recving); skipping stale event", req_id)\n'
        "                continue\n",
    ),
    (
        '            logger.debug("Finished sending KV transfer for request %s", req_id)\n'
        "            assert req_id in self.requests\n",
        '            logger.debug("Finished sending KV transfer for request %s", req_id)\n'
        "            if req_id not in self.requests:\n"
        '                logger.warning("KV xfer finished for unknown req %s "\n'
        '                               "(sending); skipping stale event", req_id)\n'
        "                continue\n",
    ),
]

for old, new in reps:
    n = src.count(old)
    if n != 1:
        print(f"sched-guard: expected 1 occurrence, found {n} — vLLM layout changed?")
        sys.exit(1)
    src = src.replace(old, new)

open(f, "w").write(src)
print("sched-guard: patched", f)
