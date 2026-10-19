#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Give padded rows an empty KV range in sparse DCP decode.

With sparse MLA and DCP, the indexer's fused DCP candidate exchange writes the
compact ``dcp_sparse_kv_indptr_buffer`` for the scheduled rows only
(``scheduled_bs * max_seqlen_q``), and ``_forward_decode`` reads ``[:B + 1]`` of
it, B being the padded row count. A captured graph runs the exchange over all B
rows, so only an eager step that is still padded to a graph bucket reads the
previous step's offsets in its padded rows. The P/D producer runs every q_len=1
decode that way (its graphs hold only the MTP verify shape), e.g. 6 real rows
padded to 8. A stale offset below the real end gives a negative-length row, the
persistent MLA plan gets a work with kv_end < kv_start, and the ASM decode
kernel reads kv_indices past the written region as slot ids: "Memory access
fault by GPU" on the prefill node under long-prefix load.

Fix: in such a step, end every padded row where the last real row ends
(length 0), as ATOM's ``prepare_mtp_decode`` already does for ``kv_indptr``.
Both counts are host ints, so a replayed graph runs nothing extra. Covers
q_len=1 decode and MTP verify, persistent and non-persistent alike.

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
                "                    paged_kv_indptr = self.dcp_sparse_kv_indptr_buffer[: B + 1]\n",
                "                    paged_kv_indptr = self.dcp_sparse_kv_indptr_buffer[: B + 1]\n"
                "                    # infera: empty KV range for padded rows. The DCP\n"
                "                    # exchange wrote the scheduled rows only; an eager\n"
                "                    # step padded past them holds stale offsets there.\n"
                "                    n_real = (\n"
                "                        get_forward_context().context.scheduled_bs\n"
                "                        * attn_metadata.max_seqlen_q\n"
                "                    )\n"
                "                    if n_real < B:\n"
                "                        paged_kv_indptr[n_real + 1 :] = paged_kv_indptr[n_real]\n",
            ),
        ],
    ),
}

if __name__ == "__main__":
    sys.exit(apply("atom-patch dcp-sparse-pad-rows", PATCH, package="atom"))
