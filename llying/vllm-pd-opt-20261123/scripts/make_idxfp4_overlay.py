#!/usr/bin/env python3
"""Build the MXFP4 indexer K cache overlay (vLLM PR #61985, backported to v0.34.0, target layers only) from the image's
own files.

#61985 (gfx950 + AITER) lets GLM-5.x / DeepSeek-V3.2 take `--attention-config '{"indexer_kv_dtype":"mxfp4"}'`: the
fused norm/RoPE kernel writes the indexer K cache as paged MXFP4 (64 B packed e2m1 + 4 B e8m0 per token, in the order
aiter's `paged_mxfp4_mqa_logits` reads; FP8 is 128 + 4 B), fused_q emits an MXFP4 index query, and the existing
DeepSeek-V4 MXFP4 path (#58671 / #60834, both in v0.34.0) scores the cache in place and runs the top-k. GLM-5.2 has 21
target indexer layers: KV cache about +2.9% tokens per decode rank, indexer logits read about half the bytes.
Its prerequisites (#52550, #58671, #60834, #61943) are all in v0.34.0; the hunks that did not apply touch #61942
(short-prefill skip) and PCP context only and are resolved here by hand.
Differences from the PR:
  - MTP draft excluded (InferenceX: the draft keeps its default KV dtype). DeepseekV32ROCmIndexerCache keeps FP8 for an
    MTP-architecture model or a layer `layers.N.` with N >= VPD_IDX_MXFP4_DRAFT_MIN_LAYER (78); VPD_IDX_MXFP4_DRAFT=1
    restores the PR's behaviour. Each indexer cache logs "[vpd] indexer K cache <prefix>: mxfp4|fp8" when mxfp4 is on.
  - The MXFP4 path does not go through rocm_aiter_sparse_attn_indexer, so our Triton decode top-k overlay
    (make_tritontopk_overlay.py) would not apply to it and k = 2048 would fall back to aiter's one-block top-k. The
    same hook is added to `_dense_decode` in ops/rocm_paged_mxfp4_indexer.py, with the same env (VPD_TOPK_IMPL /
    VPD_TRITON_TOPK, VPD_TOPK_SCOPE, VPD_TOPK_DRAFT_MIN_LAYER, VPD_TOPK_SHADOW) and the same guards (uncompressed
    indexer, fp32 unit-stride logits, <= 128 rows, k a power of two <= 2048, one int32 length per row);
    vpd_triton_topk_decode.py must be importable when VPD_TOPK_IMPL=triton (mounted next to vllm, as S_TOPK=1 does).
  - fused_q's docstring hunk is left out.
With indexer_kv_dtype unset (FP8) every file behaves as v0.34.0's apart from the module-level imports. Prefill and
decode must both mount it and pass the attention config: Mooncake moves the indexer K pages, sized from the KV spec
(4352 B instead of 8448 B per 64-token page), and rejects a mismatch at the handshake.
Each image file must be v0.34.0's (sha256 below); each output is checked against the sha256 of the reviewed result.
  make_idxfp4_overlay.py SITE_PACKAGES OUT_DIR
      -> OUT_DIR/{rocm.py,kernels.py,indexer.py,rocm_paged_mxfp4_indexer_backend.py,rocm_paged_mxfp4_indexer_ops.py}
         and OUT_DIR/mounts.txt (one `file:site-packages path` per line)
"""
import hashlib
import os
import sys

ROCM_1 = (
    '# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n'
    '\n'
    'from typing import cast\n'
    '\n',
    '# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\n'
    '\n'
    'import os\n'
    'import re\n'
    'from typing import cast\n'
    '\n',
)

ROCM_2 = (
    'from vllm._aiter_ops import rocm_aiter_ops\n'
    'from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n'
    'from vllm.forward_context import get_forward_context\n'
    'from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata\n'
    'from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer\n',
    'from vllm._aiter_ops import rocm_aiter_ops\n'
    'from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n'
    'from vllm.config import CacheConfig, get_current_vllm_config\n'
    'from vllm.forward_context import get_forward_context\n'
    'from vllm.logger import init_logger\n'
    'from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata\n'
    'from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer\n',
)

ROCM_3 = (
    'from vllm.models.deepseek_v32.common.kernels import fused_norm_rope, fused_q\n'
    'from vllm.platforms import current_platform\n'
    'from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend\n'
    'from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (\n'
    '    ROCMAiterMLASparseBackend,\n'
    ')\n'
    'from vllm.v1.kv_cache_interface import is_quantized_kv_cache\n'
    '\n'
    '\n'
    'class DeepseekV32ROCmIndexerCache(DeepseekV32IndexerCache):\n'
    '    def get_attn_backend(self):\n'
    '        return DeepseekV32IndexerBackend\n'
    '\n',
    'from vllm.models.deepseek_v32.common.kernels import fused_norm_rope, fused_q\n'
    'from vllm.platforms import current_platform\n'
    'from vllm.v1.attention.backend import AttentionBackend\n'
    'from vllm.v1.attention.backends.mla.indexer import (\n'
    '    DeepseekV32IndexerBackend,\n'
    '    dsa_indexer_uses_fp4,\n'
    ')\n'
    'from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (\n'
    '    ROCMAiterMLASparseBackend,\n'
    ')\n'
    'from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (\n'
    '    MXFP4_BLOCK_SIZE,\n'
    '    rocm_paged_mxfp4_cache_layout,\n'
    ')\n'
    'from vllm.v1.kv_cache_interface import is_quantized_kv_cache\n'
    '\n'
    '\n'
    '_vpd_logger = init_logger(__name__)\n'
    '_VPD_IDX_MXFP4_DRAFT = os.environ.get("VPD_IDX_MXFP4_DRAFT", "0") == "1"\n'
    '_VPD_DRAFT_MIN_LAYER = int(os.environ.get("VPD_IDX_MXFP4_DRAFT_MIN_LAYER", "78"))\n'
    '\n'
    '\n'
    'def _vpd_is_draft_indexer(vllm_config, prefix: str) -> bool:\n'
    '    """MTP draft indexers keep the FP8 cache (InferenceX: no MXFP4 on the draft).\n'
    '\n'
    "    A layer is the draft's when the model being built is an MTP model or its\n"
    '    layer index is >= the target\'s layer count (GLM-5.2: layers.78)."""\n'
    '    if _VPD_IDX_MXFP4_DRAFT:\n'
    '        return False\n'
    '    model_config = vllm_config.model_config\n'
    '    archs = set(getattr(model_config, "architectures", None) or ())\n'
    '    if any("MTP" in a for a in archs):\n'
    '        return True\n'
    '    m = re.search(r"layers\\.(\\d+)\\.", prefix)\n'
    '    return m is not None and int(m.group(1)) >= _VPD_DRAFT_MIN_LAYER\n'
    '\n'
    '\n'
    'class DeepseekV32ROCmIndexerCache(DeepseekV32IndexerCache):\n'
    '    def __init__(\n'
    '        self, head_dim: int, dtype: torch.dtype, prefix: str, cache_config: CacheConfig\n'
    '    ):\n'
    '        vllm_config = get_current_vllm_config()\n'
    '        requested = dsa_indexer_uses_fp4(vllm_config)\n'
    '        mxfp4 = requested and not _vpd_is_draft_indexer(vllm_config, prefix)\n'
    '        if requested:\n'
    '            _vpd_logger.info(\n'
    '                "[vpd] indexer K cache %s: %s", prefix, "mxfp4" if mxfp4 else "fp8"\n'
    '            )\n'
    '        if mxfp4:\n'
    '            # Packed e2m1 values, then one e8m0 byte per 32 of them.\n'
    '            dim = vllm_config.model_config.hf_text_config.index_head_dim\n'
    '            head_dim = dim // 2 + dim // MXFP4_BLOCK_SIZE\n'
    '        super().__init__(head_dim, dtype, prefix, cache_config)\n'
    '        self.mxfp4 = mxfp4\n'
    '\n'
    '    def get_attn_backend(self) -> type[AttentionBackend]:\n'
    '        if self.mxfp4:\n'
    '            from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (\n'
    '                DeepseekV32RocmMxfp4IndexerBackend,\n'
    '            )\n'
    '\n'
    '            return DeepseekV32RocmMxfp4IndexerBackend\n'
    '        return DeepseekV32IndexerBackend\n'
    '\n',
)

ROCM_4 = (
    '\n'
    '        self.indexer_op: SparseAttnIndexer | None = None\n'
    '        if self.indexer is not None:\n'
    '            self.indexer_op = SparseAttnIndexer(\n'
    '                self.indexer.k_cache,\n'
    '                self.indexer.quant_block_size,\n',
    '\n'
    '        self.indexer_op: SparseAttnIndexer | None = None\n'
    '        self._indexer_mxfp4 = (\n'
    '            self.indexer is not None and self.indexer.k_cache.mxfp4  # type: ignore[attr-defined]\n'
    '        )\n'
    '        if self.indexer is not None:\n'
    '            indexer_op_cls = SparseAttnIndexer\n'
    '            if self._indexer_mxfp4:\n'
    '                from vllm.model_executor.layers.rocm_paged_mxfp4_indexer import (\n'
    '                    RocmSparseAttnIndexer,\n'
    '                )\n'
    '\n'
    '                indexer_op_cls = RocmSparseAttnIndexer\n'
    '            self.indexer_op = indexer_op_cls(\n'
    '                self.indexer.k_cache,\n'
    '                self.indexer.quant_block_size,\n',
)

ROCM_5 = (
    '        self,\n'
    '        q_c: torch.Tensor,\n'
    '        index_q_fp8: torch.Tensor | None,\n'
    '        index_weights_out: torch.Tensor | None,\n'
    '    ) -> None:\n',
    '        self,\n'
    '        q_c: torch.Tensor,\n'
    '        index_q_fp8: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,\n'
    '        index_weights_out: torch.Tensor | None,\n'
    '    ) -> None:\n',
)

ROCM_6 = (
    '            indexer_k_cache = self.indexer.k_cache.kv_cache\n'
    '            indexer_cache_shuffled = self.indexer.k_cache.uses_shuffled_layout\n'
    '            indexer_softmax_scale = self.indexer.softmax_scale\n'
    '            indexer_n_head_scale = self.indexer.n_head**-0.5\n',
    '            indexer_k_cache = self.indexer.k_cache.kv_cache\n'
    '            indexer_cache_shuffled = self.indexer.k_cache.uses_shuffled_layout\n'
    '            indexer_cache_mxfp4 = None\n'
    '            if self._indexer_mxfp4 and indexer_k_cache.numel():\n'
    '                indexer_cache_mxfp4 = rocm_paged_mxfp4_cache_layout(\n'
    '                    self.indexer.n_head,\n'
    '                    self.indexer.head_dim,\n'
    '                    indexer_k_cache.shape[1],\n'
    '                )\n'
    '            indexer_softmax_scale = self.indexer.softmax_scale\n'
    '            indexer_n_head_scale = self.indexer.n_head**-0.5\n',
)

ROCM_7 = (
    '            indexer_k_cache = None\n'
    '            indexer_cache_shuffled = False\n'
    '            indexer_softmax_scale = 0.0\n'
    '            indexer_n_head_scale = 0.0\n',
    '            indexer_k_cache = None\n'
    '            indexer_cache_shuffled = False\n'
    '            indexer_cache_mxfp4 = None\n'
    '            indexer_softmax_scale = 0.0\n'
    '            indexer_n_head_scale = 0.0\n',
)

ROCM_8 = (
    '            indexer_k_cache=indexer_k_cache,\n'
    '            indexer_cache_shuffled=indexer_cache_shuffled,\n'
    '            mla_kv_cache=mla_kv_cache,\n'
    '            mla_kv_cache_dtype=self.kv_cache_dtype,\n',
    '            indexer_k_cache=indexer_k_cache,\n'
    '            indexer_cache_shuffled=indexer_cache_shuffled,\n'
    '            indexer_cache_mxfp4=indexer_cache_mxfp4,\n'
    '            mla_kv_cache=mla_kv_cache,\n'
    '            mla_kv_cache_dtype=self.kv_cache_dtype,\n',
)

ROCM_9 = (
    '            index_rope_interleave=self._index_rope_interleave,\n'
    '            quantize_mqa=self._fp8_kv and not use_mha,\n'
    '        )\n'
    '\n',
    '            index_rope_interleave=self._index_rope_interleave,\n'
    '            quantize_mqa=self._fp8_kv and not use_mha,\n'
    '            index_q_mxfp4=index_q is not None and self._indexer_mxfp4,\n'
    '        )\n'
    '\n',
)

ROCM_EDITS = [ROCM_1, ROCM_2, ROCM_3, ROCM_4, ROCM_5, ROCM_6, ROCM_7, ROCM_8, ROCM_9]


KERN_1 = (
    '\n'
    '@triton.jit\n'
    'def _fused_norm_rope_kernel(\n'
    '    pos_ptr,\n',
    '\n'
    '@triton.jit\n'
    'def _mxfp4_quantize(vals, HEAD_DIM: tl.constexpr):\n'
    '    """[HEAD_DIM] fp32 to MXFP4: ([HEAD_DIM // 2] e2m1 pairs, low nibble\n'
    '    first, [HEAD_DIM // 32] e8m0 scales of 2^ceil(log2(amax / 6))), so no\n'
    '    value saturates. gfx950 only: one v_cvt_scalef32_pk_fp4_f32 per pair."""\n'
    '    N_BLOCKS: tl.constexpr = HEAD_DIM // 32\n'
    '    blocks = tl.reshape(vals, (N_BLOCKS, 32))\n'
    '    ratio = tl.maximum(tl.max(tl.abs(blocks), axis=1), 6.0 * (2**-126)) / 6.0\n'
    '    # Exact ceil(log2): a nonzero mantissa carries into the exponent.\n'
    '    e8m0 = (ratio.to(tl.uint32, bitcast=True) + 0x7FFFFF) >> 23\n'
    '    inv_scale = ((254 - e8m0) << 23).to(tl.float32, bitcast=True)\n'
    '    lo, hi = tl.split(tl.reshape(blocks * inv_scale[:, None], (N_BLOCKS, 16, 2)))\n'
    '    packed = tl.inline_asm_elementwise(\n'
    '        "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",\n'
    '        "=v,v,v,v",\n'
    '        args=[lo, hi, 1.0],\n'
    '        dtype=tl.int32,\n'
    '        is_pure=True,\n'
    '        pack=1,\n'
    '    ).to(tl.uint8)\n'
    '    return tl.reshape(packed, (HEAD_DIM // 2,)), e8m0.to(tl.uint8)\n'
    '\n'
    '\n'
    '@triton.jit\n'
    'def _mxfp4_quant_and_cache_write(\n'
    '    vals,\n'
    '    slot_idx,\n'
    '    kv_cache_ptr,\n'
    '    cache_block_size,\n'
    '    cache_block_stride,\n'
    '    HEAD_DIM: tl.constexpr,\n'
    '    N_PER_TILE: tl.constexpr,\n'
    '    D_PER_TILE: tl.constexpr,\n'
    '    SCALE_LANES: tl.constexpr,\n'
    '):\n'
    '    """Store one key in the paged MXFP4 order aiter\'s MQA-logits kernel reads\n'
    '    (see `RocmPagedMxfp4CacheLayout`): a page\'s values, then its scales."""\n'
    '    packed, e8m0 = _mxfp4_quantize(vals, HEAD_DIM)\n'
    '    TOKEN_BYTES: tl.constexpr = HEAD_DIM // 2\n'
    '    N_BLOCKS: tl.constexpr = HEAD_DIM // 32\n'
    '    S_HI: tl.constexpr = N_BLOCKS // SCALE_LANES\n'
    '    page = kv_cache_ptr + (slot_idx // cache_block_size) * cache_block_stride\n'
    '    pos = slot_idx % cache_block_size\n'
    '    group = pos // N_PER_TILE\n'
    '    lane = pos % N_PER_TILE\n'
    '    byte = tl.arange(0, TOKEN_BYTES)\n'
    '    scale_idx = tl.arange(0, N_BLOCKS)\n'
    '    value_off = (\n'
    '        group * (N_PER_TILE * TOKEN_BYTES)\n'
    '        + byte // D_PER_TILE * (N_PER_TILE * D_PER_TILE)\n'
    '        + lane * D_PER_TILE\n'
    '        + byte % D_PER_TILE\n'
    '    )\n'
    '    scale_off = (\n'
    '        group * (N_PER_TILE * N_BLOCKS)\n'
    '        + scale_idx % SCALE_LANES * (N_PER_TILE * S_HI)\n'
    '        + lane * S_HI\n'
    '        + scale_idx // SCALE_LANES\n'
    '    )\n'
    '    tl.store(page + value_off, packed)\n'
    '    tl.store(page + cache_block_size * TOKEN_BYTES + scale_off, e8m0)\n'
    '\n'
    '\n'
    '@triton.jit\n'
    'def _fused_norm_rope_kernel(\n'
    '    pos_ptr,\n',
)

KERN_2 = (
    '    INDEXER_CACHE_HEAD_TILE: tl.constexpr,\n'
    '    INDEXER_CACHE_MX_SCALE: tl.constexpr,\n'
    '    # MLA KV cache (concat kv_c_normed + k_pe_roped, uses slot_mapping_ptr)\n'
    '    mla_cache_ptr,\n',
    '    INDEXER_CACHE_HEAD_TILE: tl.constexpr,\n'
    '    INDEXER_CACHE_MX_SCALE: tl.constexpr,\n'
    '    # Index K MXFP4 cache (ROCm), in place of the FP8 one\n'
    '    INDEXER_CACHE_MXFP4: tl.constexpr,\n'
    '    MXFP4_N_PER_TILE: tl.constexpr,\n'
    '    MXFP4_D_PER_TILE: tl.constexpr,\n'
    '    MXFP4_SCALE_LANES: tl.constexpr,\n'
    '    # MLA KV cache (concat kv_c_normed + k_pe_roped, uses slot_mapping_ptr)\n'
    '    mla_cache_ptr,\n',
)

KERN_3 = (
    '        if indexer_cache_ptr is not None and indexer_slot_mapping_ptr is not None:\n'
    '            slot_idx = tl.load(indexer_slot_mapping_ptr + tok_idx)\n'
    '            _fp8_quant_and_cache_write(\n'
    '                result,\n'
    '                index_k_mask,\n'
    '                slot_idx,\n'
    '                indexer_cache_ptr,\n'
    '                indexer_cache_scale_ptr,\n'
    '                indexer_cache_block_size,\n'
    '                indexer_cache_stride,\n'
    '                index_k_block,\n'
    '                INDEX_K_DIM,\n'
    '                INDEXER_CACHE_SHUFFLE,\n'
    '                INDEXER_CACHE_BLOCK_TILE,\n'
    '                INDEXER_CACHE_HEAD_TILE,\n'
    '                INDEXER_CACHE_MX_SCALE,\n'
    '                FP8_MAX,\n'
    '                USE_FNUZ,\n'
    '            )\n'
    '\n'
    '\n',
    '        if indexer_cache_ptr is not None and indexer_slot_mapping_ptr is not None:\n'
    '            slot_idx = tl.load(indexer_slot_mapping_ptr + tok_idx)\n'
    '            if INDEXER_CACHE_MXFP4:\n'
    '                if slot_idx >= 0:\n'
    '                    _mxfp4_quant_and_cache_write(\n'
    '                        result,\n'
    '                        slot_idx,\n'
    '                        indexer_cache_ptr,\n'
    '                        indexer_cache_block_size,\n'
    '                        indexer_cache_stride,\n'
    '                        INDEX_K_DIM,\n'
    '                        MXFP4_N_PER_TILE,\n'
    '                        MXFP4_D_PER_TILE,\n'
    '                        MXFP4_SCALE_LANES,\n'
    '                    )\n'
    '            else:\n'
    '                _fp8_quant_and_cache_write(\n'
    '                    result,\n'
    '                    index_k_mask,\n'
    '                    slot_idx,\n'
    '                    indexer_cache_ptr,\n'
    '                    indexer_cache_scale_ptr,\n'
    '                    indexer_cache_block_size,\n'
    '                    indexer_cache_stride,\n'
    '                    index_k_block,\n'
    '                    INDEX_K_DIM,\n'
    '                    INDEXER_CACHE_SHUFFLE,\n'
    '                    INDEXER_CACHE_BLOCK_TILE,\n'
    '                    INDEXER_CACHE_HEAD_TILE,\n'
    '                    INDEXER_CACHE_MX_SCALE,\n'
    '                    FP8_MAX,\n'
    '                    USE_FNUZ,\n'
    '                )\n'
    '\n'
    '\n',
)

KERN_4 = (
    '    indexer_k_cache: torch.Tensor | None = None,\n'
    '    indexer_cache_shuffled: bool = False,\n'
    '    mla_kv_cache: torch.Tensor | None = None,\n'
    '    mla_kv_cache_dtype: str = "auto",\n',
    '    indexer_k_cache: torch.Tensor | None = None,\n'
    '    indexer_cache_shuffled: bool = False,\n'
    '    indexer_cache_mxfp4: tuple[int, int, int] | None = None,\n'
    '    mla_kv_cache: torch.Tensor | None = None,\n'
    '    mla_kv_cache_dtype: str = "auto",\n',
)

KERN_5 = (
    '\n'
    '    # --- Indexer K cache setup ---\n'
    '    if indexer_k_cache is not None:\n'
    '        assert slot_mapping is not None\n'
    '        if indexer_slot_mapping is None:\n',
    '\n'
    '    # --- Indexer K cache setup ---\n'
    '    # (n_per_tile, d_per_tile, scale_lanes) of a paged MXFP4 cache, see\n'
    '    # RocmPagedMxfp4CacheLayout; (1, 1, 1) when the cache is FP8.\n'
    '    mxfp4_layout = (1, 1, 1)\n'
    '    if indexer_k_cache is not None and indexer_cache_mxfp4 is not None:\n'
    '        assert indexer_k_cache.dtype == torch.uint8\n'
    '        assert index_k.shape[-1] == 128, "the MXFP4 cache is written for D=128"\n'
    '        if indexer_slot_mapping is None:\n'
    '            indexer_slot_mapping = slot_mapping\n'
    '        mxfp4_layout = indexer_cache_mxfp4\n'
    '        idx_cache_scale_view = None\n'
    '        idx_cache_mx_scale = False\n'
    '        idx_cache_block_size = indexer_k_cache.shape[1]\n'
    '        idx_cache_stride = indexer_k_cache.stride(0)\n'
    '        idx_cache_shuffle = False\n'
    '        idx_cache_head_tile = _INDEXER_CACHE_HEAD_TILE_BYTES\n'
    '    elif indexer_k_cache is not None:\n'
    '        assert slot_mapping is not None\n'
    '        if indexer_slot_mapping is None:\n',
)

KERN_6 = (
    '        idx_cache_head_tile,\n'
    '        idx_cache_mx_scale,\n'
    '        # MLA KV cache (uses same slot_mapping)\n'
    '        mla_kv_cache,\n',
    '        idx_cache_head_tile,\n'
    '        idx_cache_mx_scale,\n'
    '        indexer_cache_mxfp4 is not None,\n'
    '        *mxfp4_layout,\n'
    '        # MLA KV cache (uses same slot_mapping)\n'
    '        mla_kv_cache,\n',
)

KERN_7 = (
    '    index_q_fp8_stride0,\n'
    '    index_q_fp8_stride1,\n'
    '    INDEX_Q_HEAD_DIM: tl.constexpr,\n'
    '    # MQA query pack: quantize ql_nope and RoPE+quantize q_pe into mqa_q_fp8\n',
    '    index_q_fp8_stride0,\n'
    '    index_q_fp8_stride1,\n'
    '    index_q_scale_ptr,\n'
    '    index_q_scale_stride0,\n'
    '    index_q_scale_stride1,\n'
    '    INDEX_Q_HEAD_DIM: tl.constexpr,\n'
    '    # MQA query pack: quantize ql_nope and RoPE+quantize q_pe into mqa_q_fp8\n',
)

KERN_8 = (
    '    INDEX_ROPE_INTERLEAVE: tl.constexpr,\n'
    '    QUANTIZE_MQA: tl.constexpr,\n'
    '    USE_PDL: tl.constexpr,\n'
    '    FP8_MAX: tl.constexpr,\n',
    '    INDEX_ROPE_INTERLEAVE: tl.constexpr,\n'
    '    QUANTIZE_MQA: tl.constexpr,\n'
    '    INDEX_Q_MXFP4: tl.constexpr,\n'
    '    USE_PDL: tl.constexpr,\n'
    '    FP8_MAX: tl.constexpr,\n',
)

KERN_9 = (
    '        index_q = tl.where(in_rope, roped, index_q)\n'
    '\n'
    '        # Index Q Quantize (from registers)\n'
    '        index_q_fp8, index_q_scale = _fp8_ue8m0_quantize(index_q, FP8_MAX, USE_FNUZ)\n'
    '        tl.store(\n'
    '            index_q_fp8_ptr\n'
    '            + tok_idx * index_q_fp8_stride0\n'
    '            + head_idx * index_q_fp8_stride1\n'
    '            + index_q_block,\n'
    '            index_q_fp8,\n'
    '        )\n'
    '\n'
    '        # Index weights update\n'
    '        index_weights = tl.load(\n',
    '        index_q = tl.where(in_rope, roped, index_q)\n'
    '\n'
    '        # Index weights update\n'
    '        index_weights = tl.load(\n',
)

KERN_10 = (
    '        )\n'
    '        index_weights = index_weights.to(tl.float32)\n'
    '        index_weights *= index_q_scale\n'
    '        index_weights *= index_weights_softmax_scale\n'
    '        index_weights *= index_weights_head_scale\n',
    '        )\n'
    '        index_weights = index_weights.to(tl.float32)\n'
    '\n'
    '        # Index Q Quantize (from registers)\n'
    '        if INDEX_Q_MXFP4:\n'
    '            # The scorer takes the block scales, so none fold into the weights.\n'
    '            index_q_fp4, index_q_e8m0 = _mxfp4_quantize(index_q, INDEX_Q_HEAD_DIM)\n'
    '            tl.store(\n'
    '                index_q_fp8_ptr\n'
    '                + tok_idx * index_q_fp8_stride0\n'
    '                + head_idx * index_q_fp8_stride1\n'
    '                + tl.arange(0, INDEX_Q_HEAD_DIM // 2),\n'
    '                index_q_fp4,\n'
    '            )\n'
    '            tl.store(\n'
    '                index_q_scale_ptr\n'
    '                + tok_idx * index_q_scale_stride0\n'
    '                + head_idx * index_q_scale_stride1\n'
    '                + tl.arange(0, INDEX_Q_HEAD_DIM // 32),\n'
    '                index_q_e8m0,\n'
    '            )\n'
    '        else:\n'
    '            index_q_fp8, index_q_scale = _fp8_ue8m0_quantize(index_q, FP8_MAX, USE_FNUZ)\n'
    '            tl.store(\n'
    '                index_q_fp8_ptr\n'
    '                + tok_idx * index_q_fp8_stride0\n'
    '                + head_idx * index_q_fp8_stride1\n'
    '                + index_q_block,\n'
    '                index_q_fp8,\n'
    '            )\n'
    '            index_weights *= index_q_scale\n'
    '        index_weights *= index_weights_softmax_scale\n'
    '        index_weights *= index_weights_head_scale\n',
)

KERN_11 = (
    '    index_rope_interleave: bool = False,\n'
    '    quantize_mqa: bool = True,\n'
    ') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:\n'
    '    """Fuse the MQA-query and indexer-query RoPE/quantization.\n'
    '\n',
    '    index_rope_interleave: bool = False,\n'
    '    quantize_mqa: bool = True,\n'
    '    index_q_mxfp4: bool = False,\n'
    ') -> tuple[\n'
    '    torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor\n'
    ']:\n'
    '    """Fuse the MQA-query and indexer-query RoPE/quantization.\n'
    '\n',
)

KERN_12 = (
    '    # cutlass at module scope, so only reach for it on CUDA.\n'
    '    cutedsl_kernel: Callable[..., None] | None = None\n'
    '    if current_platform.is_cuda():\n'
    '        from vllm.models.deepseek_v32.nvidia.ops.fused_q_cutedsl import (\n'
    '            fused_q_cutedsl,\n',
    '    # cutlass at module scope, so only reach for it on CUDA.\n'
    '    cutedsl_kernel: Callable[..., None] | None = None\n'
    '    if current_platform.is_cuda() and not index_q_mxfp4:\n'
    '        from vllm.models.deepseek_v32.nvidia.ops.fused_q_cutedsl import (\n'
    '            fused_q_cutedsl,\n',
)

KERN_13 = (
    '        mqa_q = q_pe_out\n'
    '\n'
    '    index_q_fp8 = torch.empty_like(index_q, dtype=_FP8_DTYPE)\n'
    '    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)\n'
    '    if cutedsl_kernel is not None:\n',
    '        mqa_q = q_pe_out\n'
    '\n'
    '    if index_q_mxfp4:\n'
    '        assert has_indexer and index_q_head_dim == 128, (\n'
    '            "MXFP4 scales pack as one int32 per head"\n'
    '        )\n'
    '        index_q_fp8 = index_q.new_empty(\n'
    '            (num_tokens, num_index_q_heads, index_q_head_dim // 2), dtype=torch.uint8\n'
    '        )\n'
    '        index_q_scale = index_q.new_empty(\n'
    '            (num_tokens, num_index_q_heads, index_q_head_dim // 32), dtype=torch.uint8\n'
    '        )\n'
    '    else:\n'
    '        index_q_fp8 = torch.empty_like(index_q, dtype=_FP8_DTYPE)\n'
    '        index_q_scale = index_q_fp8\n'
    '    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)\n'
    '    if cutedsl_kernel is not None:\n',
)

KERN_14 = (
    '        index_q_fp8.stride(0),\n'
    '        index_q_fp8.stride(1),\n'
    '        index_q_head_dim,\n'
    '        ql_nope,\n',
    '        index_q_fp8.stride(0),\n'
    '        index_q_fp8.stride(1),\n'
    '        index_q_scale,\n'
    '        index_q_scale.stride(0),\n'
    '        index_q_scale.stride(1),\n'
    '        index_q_head_dim,\n'
    '        ql_nope,\n',
)

KERN_15 = (
    '        INDEX_ROPE_INTERLEAVE=index_rope_interleave,\n'
    '        QUANTIZE_MQA=quantize_mqa,\n'
    '        FP8_MAX=_FP8_MAX,\n'
    '        USE_FNUZ=_USE_FNUZ,\n',
    '        INDEX_ROPE_INTERLEAVE=index_rope_interleave,\n'
    '        QUANTIZE_MQA=quantize_mqa,\n'
    '        INDEX_Q_MXFP4=index_q_mxfp4,\n'
    '        FP8_MAX=_FP8_MAX,\n'
    '        USE_FNUZ=_USE_FNUZ,\n',
)

KERN_16 = (
    '        num_warps=1,\n'
    '    )\n'
    '    return index_q_fp8, index_weights_out, mqa_q\n'
    '\n',
    '        num_warps=1,\n'
    '    )\n'
    '    if index_q_mxfp4:\n'
    '        q_scale_i32 = index_q_scale.view(torch.int32).squeeze(-1)\n'
    '        return (index_q_fp8, q_scale_i32), index_weights_out, mqa_q\n'
    '    return index_q_fp8, index_weights_out, mqa_q\n'
    '\n',
)

KERN_EDITS = [KERN_1, KERN_2, KERN_3, KERN_4, KERN_5, KERN_6, KERN_7, KERN_8, KERN_9, KERN_10, KERN_11, KERN_12, KERN_13, KERN_14, KERN_15, KERN_16]


IDX_1 = (
    '# and mxfp4 is the opt-in Blackwell and gfx950 path.\n'
    'DSA_INDEXER_KV_DTYPES = ("fp8", "mxfp4")\n'
    '\n'
    '\n',
    '# and mxfp4 is the opt-in Blackwell and gfx950 path.\n'
    'DSA_INDEXER_KV_DTYPES = ("fp8", "mxfp4")\n'
    '# Served by vllm/models/deepseek_v32/amd, whose indexer writes and scores the\n'
    '# ROCm paged MXFP4 cache.\n'
    '_ROCM_MXFP4_V32_ARCHS = frozenset(\n'
    '    ("DeepseekV32ForCausalLM", "GlmMoeDsaForCausalLM", "DeepseekV32MTPModel")\n'
    ')\n'
    '\n'
    '\n',
)

IDX_2 = (
    '        from vllm.platforms.rocm import get_cdna_version\n'
    '\n'
    '        # Only DeepSeek-V4 and V4.1 are wired to the ROCm MXFP4 cache; other\n'
    '        # DSA models would silently keep their FP8 one.\n'
    '        model_config = vllm_config.model_config\n'
    '        if model_config is None or model_config.hf_config.model_type not in (\n'
    '            "deepseek_v4",\n'
    '            "deepseek_v41",\n'
    '        ):\n'
    '            raise ValueError(\n',
    '        from vllm.platforms.rocm import get_cdna_version\n'
    '\n'
    '        # Only DeepSeek-V4, V4.1 and the V3.2-style DSA models (GLM-5.x, MTP\n'
    '        # drafts included) are wired to the ROCm MXFP4 cache.\n'
    '        model_config = vllm_config.model_config\n'
    '        if model_config is None or not (\n'
    '            model_config.hf_config.model_type in ("deepseek_v4", "deepseek_v41")\n'
    '            or set(model_config.architectures) & _ROCM_MXFP4_V32_ARCHS\n'
    '        ):\n'
    '            raise ValueError(\n',
)

IDX_EDITS = [IDX_1, IDX_2]


BACKEND_1 = (
    'from vllm.v1.attention.backends.mla.indexer import (\n'
    '    DeepseekV4IndexerBackend,\n'
    '    DeepseekV32IndexerMetadata,\n'
    '    DeepseekV32IndexerMetadataBuilder,\n',
    'from vllm.v1.attention.backends.mla.indexer import (\n'
    '    DeepseekV4IndexerBackend,\n'
    '    DeepseekV32IndexerBackend,\n'
    '    DeepseekV32IndexerMetadata,\n'
    '    DeepseekV32IndexerMetadataBuilder,\n',
)

BACKEND_2 = (
    '\n'
    '\n'
    'class RocmMxfp4IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):\n'
    '    """Dense indexer metadata plus the in-place paged launches\' layout.\n',
    '\n'
    '\n'
    'class DeepseekV32RocmMxfp4IndexerBackend(DeepseekV32IndexerBackend):\n'
    '    """The DeepSeek-V3.2 / GLM-5 indexer on the paged MXFP4 cache."""\n'
    '\n'
    '    @staticmethod\n'
    '    def get_name() -> str:\n'
    '        return "DEEPSEEK_V32_ROCM_MXFP4_INDEXER"\n'
    '\n'
    '    @staticmethod\n'
    '    def get_supported_kernel_block_sizes(\n'
    '        kv_cache_spec: KVCacheSpec | None = None,\n'
    '    ) -> list[int | MultipleOf]:\n'
    "        # aiter's kernel reads 64-entry KV tiles that may not straddle a page.\n"
    '        return [64, 128]\n'
    '\n'
    '    @staticmethod\n'
    '    def get_builder_cls() -> type["RocmMxfp4IndexerMetadataBuilder"]:\n'
    '        return RocmMxfp4IndexerMetadataBuilder\n'
    '\n'
    '\n'
    'class RocmMxfp4IndexerMetadataBuilder(DeepseekV32IndexerMetadataBuilder):\n'
    '    """Dense indexer metadata plus the in-place paged launches\' layout.\n',
)

BACKEND_EDITS = [BACKEND_1, BACKEND_2]


OPS_1 = (
    '        RocmMxfp4PrefillPlan,\n'
    '    )\n'
    '\n'
    'MXFP4_BLOCK_SIZE = 32\n',
    '        RocmMxfp4PrefillPlan,\n'
    '    )\n'
    '\n'
    'import os as _vpd_os\n'
    '\n'
    '_VPD_TOPK_IMPL = _vpd_os.environ.get("VPD_TOPK_IMPL") or (\n'
    '    "triton" if _vpd_os.environ.get("VPD_TRITON_TOPK") == "1" else ""\n'
    ')\n'
    '_VPD_TRITON_TOPK = _VPD_TOPK_IMPL == "triton"\n'
    '_VPD_TOPK_SCOPE = _vpd_os.environ.get("VPD_TOPK_SCOPE", "all")  # all | target | draft\n'
    '_VPD_DRAFT_MIN_LAYER = int(_vpd_os.environ.get("VPD_TOPK_DRAFT_MIN_LAYER", "78"))\n'
    '_VPD_TOPK_SHADOW = _vpd_os.environ.get("VPD_TOPK_SHADOW") == "1"\n'
    '\n'
    '\n'
    'def _vpd_is_draft(prefix) -> bool:\n'
    '    import re as _re\n'
    '\n'
    '    m = _re.search(r"layers\\.(\\d+)\\.", str(prefix))\n'
    '    return m is not None and int(m.group(1)) >= _VPD_DRAFT_MIN_LAYER\n'
    '\n'
    '\n'
    'def _vpd_scope_ok(prefix) -> bool:\n'
    '    if _VPD_TOPK_SCOPE == "all":\n'
    '        return True\n'
    '    draft = _vpd_is_draft(prefix)\n'
    '    return draft if _VPD_TOPK_SCOPE == "draft" else not draft\n'
    '\n'
    '\n'
    'if _VPD_TRITON_TOPK:\n'
    '    import vpd_triton_topk_decode as _vpd_tk\n'
    '\n'
    '\n'
    'def _vpd_decode_topk(\n'
    '    layer: "_Layer", logits: torch.Tensor, lengths: torch.Tensor, prefix: str\n'
    ') -> bool:\n'
    '    """The decode top-k of a dense uncompressed layer through the Triton port\n'
    "    (VPD_TOPK_IMPL=triton) or vLLM's in-tree kernel (native), as the FP8 path's\n"
    '    overlay (make_tritontopk_overlay.py) does. False: take `_topk`."""\n'
    '    if not _VPD_TOPK_IMPL or layer.compress_ratio != 1 or layer.candidates is not None:\n'
    '        return False\n'
    '    rows, k = logits.shape[0], layer.topk_tokens\n'
    '    out = layer.topk_buffer[:rows, :k]\n'
    '    lens = lengths.reshape(-1)\n'
    '    if _VPD_TOPK_IMPL == "native":\n'
    '        torch.ops._C.top_k_per_row_decode(\n'
    '            logits, 1, lens, out, rows, logits.stride(0), logits.stride(1), k\n'
    '        )\n'
    '        return True\n'
    '    if not (\n'
    '        _VPD_TRITON_TOPK\n'
    '        and lens.numel() == rows\n'
    '        and lens.dtype == torch.int32\n'
    '        and out.dtype == torch.int32\n'
    '        and out.stride(1) == 1\n'
    '        and _vpd_tk.supported(logits, k)\n'
    '        and _vpd_scope_ok(prefix)\n'
    '    ):\n'
    '        return False\n'
    '    _vpd_tk.triton_top_k_per_row_decode(logits, lens, out, k)\n'
    '    if _VPD_TOPK_SHADOW:\n'
    '        _vpd_tk.shadow_compare(\n'
    '            logits,\n'
    '            lens,\n'
    '            out,\n'
    '            k,\n'
    '            _vpd_is_draft(prefix),\n'
    '            lambda _o: torch.ops._C.top_k_per_row_decode(\n'
    '                logits, 1, lens, _o, rows, logits.stride(0), logits.stride(1), k\n'
    '            ),\n'
    '            lambda _o: _aiter_topk()(\n'
    '                logits, 1, lens, _o, rows, logits.stride(0), logits.stride(1), k=k\n'
    '            ),\n'
    '        )\n'
    '    return True\n'
    '\n'
    '\n'
    'MXFP4_BLOCK_SIZE = 32\n',
)

OPS_2 = (
    '\n'
    '\n'
    'def _dense_decode(layer: _Layer, logits_width: int, candidate_write: bool) -> None:\n'
    '    metadata = layer.metadata\n'
    '    lengths = metadata.decode_row_lens\n',
    '\n'
    '\n'
    'def _dense_decode(\n'
    '    layer: _Layer, logits_width: int, candidate_write: bool, prefix: str = ""\n'
    ') -> None:\n'
    '    metadata = layer.metadata\n'
    '    lengths = metadata.decode_row_lens\n',
)

OPS_3 = (
    '    elif candidates is not None:\n'
    '        _apply_candidate_mask_strided(logits, None, lengths, candidates, layer.block)\n'
    '    _topk(\n'
    '        logits,\n',
    '    elif candidates is not None:\n'
    '        _apply_candidate_mask_strided(logits, None, lengths, candidates, layer.block)\n'
    '    if _vpd_decode_topk(layer, logits, lengths, prefix):\n'
    '        return\n'
    '    _topk(\n'
    '        logits,\n',
)

OPS_4 = (
    '            _dense_prefill(layer, chunk, plan, candidate_write)\n'
    '    if metadata.decode is not None:\n'
    '        _dense_decode(layer, max_model_len, candidate_write)\n'
    '    return topk_indices_buffer\n'
    '\n',
    '            _dense_prefill(layer, chunk, plan, candidate_write)\n'
    '    if metadata.decode is not None:\n'
    '        _dense_decode(layer, max_model_len, candidate_write, k_cache_prefix)\n'
    '    return topk_indices_buffer\n'
    '\n',
)

OPS_EDITS = [OPS_1, OPS_2, OPS_3, OPS_4]


# out name: (path under site-packages, edits, sha256 of v0.34.0's file, sha256 of the result)
FILES = {
    "rocm.py": (
        "vllm/models/deepseek_v32/amd/rocm.py",
        ROCM_EDITS,
        "f1beac0306a7c4763507a039a038dc397c20b1bc07cd1ecade713ff541407c51",
        "4f6cc0133fa15b612f79ae47b006fd3bc015170b2ec74ba7abb73fe5b02933d8",
    ),
    "kernels.py": (
        "vllm/models/deepseek_v32/common/kernels.py",
        KERN_EDITS,
        "23be4ed95ae476fcb4193c4e253c19fdf4b5ae951c50fedf6a3299f4c25ba25a",
        "23481ff7a55b8672524b4ac1efaf82b6a31c177494672a77d85a3241e114a6d1",
    ),
    "indexer.py": (
        "vllm/v1/attention/backends/mla/indexer.py",
        IDX_EDITS,
        "a521dc1fbe7ec03354de024f9bee8db8a3ad19cbc17f4ba719f5b915ec1b420f",
        "dbdd8e1fc96434967d12cb494492c30ebe7ba167332edce374c3ed58137948f9",
    ),
    "rocm_paged_mxfp4_indexer_backend.py": (
        "vllm/v1/attention/backends/mla/rocm_paged_mxfp4_indexer.py",
        BACKEND_EDITS,
        "4ae266ee06c0a815dbbfe0c46fa00c34ce1f64234340ad09a31d3544bcd96369",
        "4dba360562486aecb684fcae0cf0928f9d31fe69a8cfbc241b57791e788927d3",
    ),
    "rocm_paged_mxfp4_indexer_ops.py": (
        "vllm/v1/attention/ops/rocm_paged_mxfp4_indexer.py",
        OPS_EDITS,
        "7648f1160b7a171de11547f7cc3e34cc039a55e0373d4fc2ed9caef7efe2ca44",
        "82fbac1fa976f119798d132312532ad9967e92550d70e6d4443a96f9bd3da093",
    ),
}


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main():
    sp, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    mounts = []
    for name, (rel, edits, base_sha, out_sha) in FILES.items():
        src = os.path.join(sp, rel)
        s = open(src).read()
        if sha256(s) != base_sha:
            sys.exit(f"{src} is not v0.34.0's (sha256 {sha256(s)}); the #61985 backport may not fit")
        for old, new in edits:
            n = s.count(old)
            if n != 1:
                sys.exit(f"expected exactly one {old.strip()[:80]!r} in {src}, found {n}")
            s = s.replace(old, new)
        if sha256(s) != out_sha:
            sys.exit(f"{name}: result sha256 {sha256(s)} is not the reviewed one {out_sha}")
        open(os.path.join(out, name), "w").write(s)
        mounts.append(f"{name}:{rel}")
        print(f"wrote {out}/{name} (from {rel})")
    open(os.path.join(out, "mounts.txt"), "w").write("\n".join(mounts) + "\n")
    print(f"wrote {out}/mounts.txt")


if __name__ == "__main__":
    main()
