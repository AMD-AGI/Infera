#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Idempotent, self-locating, STRICT-match ATOM source patch.

Bug : **hybrid GDN models (e.g. Qwen3.5-397B-A17B) produce garbage /
cannot recall in-context facts in PD disaggregation.**

Root cause: ``GDNAttentionMetadataBuilder`` inherits
``get_kv_transfer_tensors`` from ``AiterAttentionMetadataBuilder``, which only
registers the **full-attention paged KV cache** as ``block_regions`` and returns
``slot_regions=[]``. The GatedDeltaNet (linear-attention) layers' **recurrent
state** (conv_state + temporal/ssm state, stored per request in
``mamba_k_cache`` / ``mamba_v_cache``) is therefore NEVER transferred from the
prefill (producer) worker to the decode (consumer) worker. The decode worker
decodes from a ZERO mamba state, so it has no memory of the prompt context.

Symptom: world-knowledge facts (Paris/Tokyo/2+2 — stored in weights) look fine,
which masked the bug; but anything that depends on the prompt context (counting
continuation, in-context recall) comes out as token-salad / off-topic. Verified
with memory probes: single-node MIXED 5/5 recall, 2-node PD 0/5 recall.

Fix: override ``get_kv_transfer_tensors`` on ``GDNAttentionMetadataBuilder`` to
ALSO publish the mamba per-request state as **direct slot_regions** (one region
per GDN layer for K and V). The Mooncake connector already supports this exact
path — ``register_kv_caches`` sets ``_has_slot_regions=True``, and
``_execute_block_slot_transfer`` Phase 2a does a direct per-slot RDMA write
(``src_base + src_slot*unit_bytes`` -> ``dst_base + dst_slot*unit_bytes``) with
NO staging/gather/scatter needed, because one request's state is contiguous
within each layer buffer (shape ``[n_gdn, num_slots, *state_shape]``). We size
``unit_bytes`` per *cache group* (= ``slots_per_req`` contiguous tensor slots)
because ``local_slot_index`` carried in ``kv_transfer_params`` is the
per-request cache-group index. With speculative decoding off, slots_per_req==1
and a group == a single slot.

STRICT MATCHING: applied only if the exact, unique anchor is present, and only
once (marker guard). If ATOM changes the GDN builder so the anchor is gone, the
hunk is SKIPPED with a loud WARNING and the build continues — never edits code
it doesn't recognise.
"""

import importlib.util
import os
import sys

REL = "model_ops/attentions/gdn_attn.py"
MARKER = "publish GDN mamba state as slot_regions"

ANCHOR = "        }\n\n    def compute_block_bytes(self) -> int:\n"

INJECT = """        }

    def get_kv_transfer_tensors(self):
        # publish GDN mamba state as slot_regions so PD disaggregation
        # transfers the GatedDeltaNet recurrent (conv+ssm) state from prefill to
        # decode. Without this, decode runs from a zero mamba state and loses all
        # prompt context (garbage / no in-context recall). The Mooncake connector
        # transfers these via its direct per-slot RDMA path (Phase 2a, no staging).
        from atom.kv_transfer.disaggregation.types import (
            KVTransferRegion,
            KVTransferTensors,
        )

        base = super().get_kv_transfer_tensors()
        if base is None:
            return None
        runner = self.model_runner
        mk = getattr(runner, "mamba_k_cache", None)
        mv = getattr(runner, "mamba_v_cache", None)
        if mk is None or mv is None:
            # No per-request state allocated (not in PD / not yet built) -> leave
            # the inherited block-only transfer tensors untouched.
            return base

        num_slots = int(getattr(runner, "max_per_req_cache_slots", 0))
        spr = max(1, self.slots_per_req())
        num_groups = num_slots // spr if spr > 0 else num_slots

        slot_regions = list(base.slot_regions)
        n = mk.shape[0]
        for layer in range(n):
            for t in (mk[layer], mv[layer]):
                # t: [num_slots, *state_shape], contiguous in the slot dim, so a
                # request group's state is contiguous; unit = bytes per GROUP
                # (slots_per_req slots) because local_slot_index is the group idx.
                per_slot_bytes = t.stride(0) * t.element_size()
                slot_regions.append(
                    KVTransferRegion(
                        base_addr=t.data_ptr(),
                        total_bytes=t.numel() * t.element_size(),
                        unit_bytes=per_slot_bytes * spr,
                    )
                )

        return KVTransferTensors(
            block_regions=base.block_regions,
            slot_regions=slot_regions,
            num_blocks=base.num_blocks,
            num_slots=num_groups,
        )

    def compute_block_bytes(self) -> int:
"""


def _find_target():
    spec = importlib.util.find_spec("atom")
    roots = (
        list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    )
    roots.append("/app/ATOM/atom")
    for root in roots:
        cand = os.path.join(root, REL)
        if os.path.isfile(cand):
            return cand
    return None


def main() -> int:
    tag = "atom-patch gdn-pd-state-transfer"
    target = _find_target()
    if not target:
        print(f"[{tag}] WARNING: target {REL} not found — GDN PD fix NOT applied")
        return 0

    src = open(target).read()
    if MARKER in src:
        print(f"[{tag}] already applied")
        return 0

    n = src.count(ANCHOR)
    if n == 0:
        print(
            f"[{tag}] WARNING: anchor not found — ATOM source likely changed; "
            f"hunk SKIPPED (re-derive the patch)"
        )
        return 0
    if n != 1:
        print(f"[{tag}] WARNING: anchor not unique ({n}x) — hunk SKIPPED to avoid a wrong edit")
        return 0

    src = src.replace(ANCHOR, INJECT, 1)
    open(target, "w").write(src)
    print(f"[{tag}] patched {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
