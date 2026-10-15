#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Give padded rows an empty KV range before sparse DCP persistent MLA planning.

With sparse MLA, DCP and MTP, the P/D producer runs an eager q_len=1 decode
step after prefill. Its rows are padded to a batch bucket (e.g. 6 real rows to
8), but the fused DCP candidate exchange writes the compact
``dcp_sparse_kv_indptr_buffer`` for the real rows only; the padded rows keep the
previous step's offsets. ``_forward_decode`` passes ``[:B + 1]`` of that buffer
to ``_rebuild_sparse_dcp_persistent_metadata``. A stale offset below the real
end gives a negative-length row, ``get_mla_metadata_v1`` plans a work with
kv_end < kv_start, and the persistent ASM decode kernel reads kv_indices past
the written region as slot ids: "Memory access fault by GPU" on the prefill
node under long-prefix load.

Fix: before planning, set every row with q_len == 0 to end where the last real
row ends (length 0). GPU-only, no host sync, graph-safe; assumes padded rows
are trailing, which holds for bucket padding.

Written against ATOM 661ef766 in rocm/atom-dev:nightly_202610131505. Drop it
once ATOM writes or clamps the padded rows itself. Strict: see anchor_patch.py.
"""

import sys

from anchor_patch import apply

PATCH = {
    "model_ops/attention_mla.py": (
        "# infera: empty KV range for padded rows",
        [
            (
                "        assert q.shape[1] == gathered_num_heads\n"
                "        get_mla_metadata_v1(\n",
                "        assert q.shape[1] == gathered_num_heads\n"
                "        # infera: empty KV range for padded rows (q_len == 0); the\n"
                "        # DCP exchange leaves stale offsets there, and a negative\n"
                "        # length makes the persistent kernel read past kv_indices.\n"
                "        n_rows = paged_cu_seqlens_q.numel() - 1\n"
                "        if 0 < n_rows < paged_kv_indptr.numel():\n"
                "            q_lens = paged_cu_seqlens_q[1:] - paged_cu_seqlens_q[:-1]\n"
                "            n_real = (q_lens > 0).sum().view(1)\n"
                "            real_end = paged_kv_indptr.index_select(0, n_real)\n"
                "            row = torch.arange(1, n_rows + 1, device=paged_kv_indptr.device)\n"
                "            ends = paged_kv_indptr[1 : n_rows + 1]\n"
                "            ends.copy_(torch.where(row > n_real, real_end, ends))\n"
                "        get_mla_metadata_v1(\n",
            ),
        ],
    ),
}

if __name__ == "__main__":
    sys.exit(apply("atom-patch dcp-sparse-pad-rows", PATCH, package="atom"))
