#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""DCP decode merge with the quantized combine in aiter's all-to-all kernel.

When the DCP merge also quantizes for o_proj (PBM + per-token FP8, the decode
of the GLM-5.2 kit), cp_lse_a2a runs pack -> all-to-all -> unpack + combine +
quant. With patch_aiter_custom_alltoall.py the last two are one aiter kernel
(alltoall_combine_quant) that reads each rank's partials straight from the
peers: the pack writes the split layout that kernel reads (data, then the fp32
LSE), and the unpack launch and the received copy go away. Where the aiter
kernel does not apply (no custom communicator, other dtypes, sizes or world
sizes) the path is unchanged.

Written against ATOM 661ef766 in rocm/atom-dev:nightly_202610131505, on top of
patch_atom_dcp_custom_a2a.py. Strict: see anchor_patch.py.
"""

import sys

from anchor_patch import apply

HELPERS = '''# infera: one-kernel DCP merge (see _infera_dcp_merge_quant).
@triton.jit
def _infera_dcp_pack_split_kernel(
    out_ptr,  # [B, H, D]
    lse_ptr,  # [B, H] fp32
    owned_counts_ptr,  # [B], or unused
    send_ptr,  # [N, B, H_LOCAL, D]
    lse_send_ptr,  # [N, B, H_LOCAL] fp32
    out_stride_b,
    out_stride_h,
    lse_stride_b,
    lse_stride_h,
    B,
    H_LOCAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_OWNED_COUNTS: tl.constexpr,
):
    """_dcp_a2a_pack_kernel for the split layout of aiter's merge kernel: the
    same records, with the LSE in its own fp32 array after the data."""
    b = tl.program_id(axis=0).to(tl.int64)
    h = tl.program_id(axis=1).to(tl.int64)
    row = ((h // H_LOCAL) * B + b) * H_LOCAL + h % H_LOCAL
    d = tl.arange(0, HEAD_DIM)
    src = tl.load(out_ptr + b * out_stride_b + h * out_stride_h + d)
    tl.store(send_ptr + row * HEAD_DIM + d, src)
    lse = tl.load(lse_ptr + b * lse_stride_b + h * lse_stride_h)
    if HAS_OWNED_COUNTS:
        lse = tl.where(tl.load(owned_counts_ptr + b) == 0, float("-inf"), lse)
    tl.store(lse_send_ptr + row, lse)


def _infera_dcp_merge_quant(
    cp_group, cp_attn_out, cp_attn_lse, owned_counts, has_owned_counts, h_local, quant_dtype
):
    """Pack + aiter's one-kernel merge (all-to-all, LSE combine, per-token
    quant) -> (out [B, H_local, D], scale [B, 1]); None where it does not apply."""
    device_comm = getattr(cp_group, "device_communicator", None)
    ca = getattr(device_comm, "ca_comm", None)
    if ca is None or ca.disabled or not hasattr(ca, "supports_a2a_combine_quant"):
        return None
    n_ranks = cp_group.world_size
    b, h_total, head_dim = cp_attn_out.shape
    send = torch.empty(
        (n_ranks * b * h_local, head_dim + 2),
        dtype=cp_attn_out.dtype,
        device=cp_attn_out.device,
    )
    if not ca.supports_a2a_combine_quant(send, h_local, head_dim, quant_dtype):
        return None
    flat = send.view(-1)
    _infera_dcp_pack_split_kernel[(b, h_total)](
        cp_attn_out,
        cp_attn_lse,
        owned_counts,
        flat,
        flat[n_ranks * b * h_local * head_dim :].view(torch.float32),
        cp_attn_out.stride(0),
        cp_attn_out.stride(1),
        cp_attn_lse.stride(0),
        cp_attn_lse.stride(1),
        b,
        H_LOCAL=h_local,
        HEAD_DIM=head_dim,
        HAS_OWNED_COUNTS=has_owned_counts,
    )
    return ca.custom_a2a_combine_quant(send, h_local, head_dim)


def cp_lse_a2a(
'''

PATCH = {
    "model_ops/dcp_ops.py": (
        "# infera: one-kernel DCP merge",
        [
            ("def cp_lse_a2a(\n", HELPERS),
            (
                "    send = torch.empty((n_ranks, b, h_local, head_dim + pack), dtype=dtype, device=dev)\n",
                "    # infera: one-kernel DCP merge when the combine also quantizes.\n"
                "    if quant_dtype is not None:\n"
                "        merged = _infera_dcp_merge_quant(\n"
                "            cp_group,\n"
                "            cp_attn_out,\n"
                "            cp_attn_lse,\n"
                "            owned_counts,\n"
                "            has_owned_counts,\n"
                "            h_local,\n"
                "            quant_dtype,\n"
                "        )\n"
                "        if merged is not None:\n"
                "            return merged\n"
                "\n"
                "    send = torch.empty((n_ranks, b, h_local, head_dim + pack), dtype=dtype, device=dev)\n",
            ),
        ],
    ),
}

if __name__ == "__main__":
    sys.exit(apply("atom-patch dcp-custom-a2a-combine", PATCH, package="atom"))
