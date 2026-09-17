#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Do not implicitly enable EAGLE rejection sampling on a PD decode worker.

WHAT BREAKS. v0.5.19 auto-enables ``speculative_use_rejection_sampling`` on HIP
for any plain-EAGLE run, via ``_should_auto_enable_hip_rejection_sampling``. A
disaggregated *decode* worker cannot take that path: the PD handoff rebuilds
``EagleDraftInput`` from the transferred metadata, which carries topk_p,
topk_index, hidden_states, bonus tokens and dsa_topk_indices but no
``draft_probs``. ``EagleDraftWorker.draft()`` then seeds
``draft_probs_list`` with that ``None`` and dies at ``torch.stack``. Startup
warmup posts a real request through the PD path, so the worker does not survive
boot -- and every subsequent request would hit the same first draft step.

WHY THIS IS NOW AN UPSTREAM DEFECT, NOT A FORK ONE. This patch was first cut
against the xiaobochen GLM-5.2 fork and deliberately kept out of the shared
patch glob, because official v0.5.18 had no implicit enable at all: ``is_hip()``
appears once in its speculative hook, to pick a draft attention backend. v0.5.19
adopted the same auto-enable upstream. The helper enumerates the configurations
it must not flip -- EAGLE3, a draft token map, topk != 1, non-default accept
thresholds, deterministic inference -- and omits PD decode. Verified absent from
upstream ``main`` as well, so this is worth filing rather than just carrying.

WHAT IT COSTS. Nothing for a greedy deployment. Rejection sampling exists so
EAGLE verify can honour temperature and top_p; ROCm has no sampling-verify
kernel, so verify falls back to argmax, which is already exactly right at
temperature 0. An operator who wants it can still pass
``--speculative-use-rejection-sampling`` explicitly -- this only removes the
implicit selection, and only on the decode leg of a PD pair.

WHY THE GUARD RIDES ``is_hip=``. The helper's first conjunct is already
``is_hip``, so narrowing that keyword adds the condition without changing the
helper's signature or its documented contract. ``cfg`` is the live resolving
view bound at the top of ``_handle_eagle_family``; it falls through to the raw
field, and nothing declares ``disaggregation_mode`` during resolution.

DROP THIS SCRIPT when a base either guards the auto-enable on
``disaggregation_mode`` itself or transfers ``draft_probs`` across the PD
handoff. The anchor stops matching in the first case, so the drop is not silent.

Self-locating and idempotent.
"""

from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path

TAG = "[pd-rocm-rejection-sampling]"
MARKER = "PD decode does not receive draft_probs from prefill"
OLD = """    if _should_auto_enable_hip_rejection_sampling(
        is_hip=get_platform().is_hip,
"""
NEW = f"""    if _should_auto_enable_hip_rejection_sampling(
        # {MARKER}; do not implicitly select the incompatible path.
        is_hip=get_platform().is_hip and cfg.disaggregation_mode != "decode",
"""


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        raise SystemExit(f"{TAG} cannot locate sglang")
    path = Path(spec.origin).parent / "srt" / "arg_groups" / "speculative_hook.py"
    source = path.read_text()
    if MARKER in source:
        print(f"{TAG} already present — skipping")
        return 0
    if source.count(OLD) != 1:
        raise SystemExit(f"{TAG} expected one ROCm default-enable anchor in {path}")
    path.write_text(source.replace(OLD, NEW, 1))
    py_compile.compile(str(path), doraise=True)
    print(f"{TAG} implicit rejection sampling is disabled for PD decode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
