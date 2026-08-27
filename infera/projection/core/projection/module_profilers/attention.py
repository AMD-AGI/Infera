###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Optional

from infera.projection.core.projection.base_module_profiler import BaseModuleProfiler


# Bytes per element the KV cache is stored at. The cache dtype is set
# independently of the compute dtype (fp8 KV with bf16 activations is common),
# and at decode the cache read is the dominant attention cost.
_KV_CACHE_BYTES = {
    "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0, "int8": 1.0,
    "bf16": 2.0, "fp16": 2.0, "auto": 2.0, "": 2.0,
    "fp32": 4.0,
}


class AttentionProfiler(BaseModuleProfiler):
    def __init__(self, config, sub_profilers=None):
        super().__init__(config, sub_profilers)
        self.module = None  # Will be set during benchmarking
        self._cached_results = None  # Cache for (forward_time, activation_memory)
        self._cache_key = None  # Cache key (batch_size, seq_len)
        self._gemm_backend = None  # Optional: GEMM simulation backend
        self._sdpa_backend = None  # Optional: SDPA simulation backend
        # Inference phase: None (default), "prefill" or "decode".
        # When set, the SDPA simulation uses a distinct KV-cache length for
        # forward-only serving.
        self._inference_phase = None
        self._kv_seq_len = None

    def set_module(self, module):
        """Set the actual attention module for benchmarking."""
        self.module = module
        # Invalidate cache when module changes
        self._cached_results = None
        self._cache_key = None

    def set_gemm_backend(self, backend):
        """Set a GEMM simulation backend for attention linear projections."""
        self._gemm_backend = backend
        self._cached_results = None
        self._cache_key = None

    def set_sdpa_backend(self, backend):
        """Set an SDPA simulation backend for attention computation."""
        self._sdpa_backend = backend
        self._cached_results = None
        self._cache_key = None

    def set_inference_phase(self, phase, kv_seq_len=None):
        """Configure forward-only inference SDPA.

        Args:
            phase: ``"prefill"`` (query = chunk of prompt, attends to KV of
                length ``kv_seq_len``) or ``"decode"`` (query length is the
                ``seq_len`` passed to the profiler — usually 1 — attending to
                a KV cache of length ``kv_seq_len``).  ``None`` restores the
                default training behaviour.
            kv_seq_len: Length of the resident KV cache the query attends to.
        """
        self._inference_phase = phase
        self._kv_seq_len = kv_seq_len
        self._cached_results = None
        self._cache_key = None

    def estimated_num_params(self, rank: Optional[int] = None) -> int:
        args = self.config.model_config
        # Group-query & multi-latent attention support.
        # If GQA not enabled, fall back to per-head queries.
        num_query_groups = (
            args.num_query_groups
            if args.group_query_attention and args.num_query_groups
            else args.num_attention_heads
        )

        # Projection ratio: (kv_channels * n_heads) / hidden_size
        query_proj_to_hidden = (args.kv_channels * args.num_attention_heads) / args.hidden_size

        if args.multi_latent_attention:
            # q_term: either dense or LoRA factored Q with RoPE/Q-norm
            if args.q_lora_rank is None:
                q_term = (
                    args.hidden_size
                    * args.num_attention_heads
                    * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                )
            else:
                q_term = args.q_lora_rank * (
                    args.hidden_size
                    + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                    + 1
                )
            attn = (
                q_term
                # kv lora + rope + kv norm
                + args.kv_lora_rank
                * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim) + 1)
                # pos emb
                + args.hidden_size * args.qk_pos_emb_head_dim
                # out proj
                + (args.num_attention_heads * args.v_head_dim) * args.hidden_size
            )
            return attn

        # Standard attention path (Q,K,V,O projections)
        return (
            2
            * args.hidden_size
            * args.hidden_size
            * ((1 + (num_query_groups / args.num_attention_heads)) * query_proj_to_hidden)
        )

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        args = self.config.model_config
        mp = self.config.model_parallel_config

        tp_size = max(1, mp.tensor_model_parallel_size)
        cp_size = max(1, mp.context_model_parallel_size)

        tokens_per_rank = batch_size * seq_len // tp_size // cp_size
        if tokens_per_rank == 0:
            return 0

        bytes_per_value = 2  # assume bf16 activations

        def _num_query_groups() -> int:
            if args.group_query_attention and args.num_query_groups:
                return args.num_query_groups
            return args.num_attention_heads

        ln_width = 0

        if args.multi_latent_attention:
            # MLA uses separate latent dimensions for Q/K and V plus optional LoRA ranks.
            heads = args.num_attention_heads
            q_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
            v_head_dim = args.v_head_dim

            q_width = heads * q_head_dim
            k_width = q_width  # key stores the same latent + positional dims
            v_width = heads * v_head_dim
            context_width = v_width  # attention output before the final projection
            query_projection_size = q_width  # For softmax width calculation

            if args.qk_layernorm:
                ln_width += q_width
                ln_width += k_width

            activation_width = q_width + k_width + v_width + context_width
        else:
            query_projection_size = args.kv_channels * args.num_attention_heads
            kv_projection_size = args.kv_channels * _num_query_groups()

            # Need to retain Q, K, V as well as the projected context/output.
            activation_width = query_projection_size + 2 * kv_projection_size + args.hidden_size

            if args.qk_layernorm:
                ln_width += kv_projection_size * 2

        heads_per_partition = max(1, args.num_attention_heads // tp_size)
        seqlen_per_cp = max(1, (seq_len + cp_size - 1) // cp_size)
        if getattr(args, "use_flash_attn", False):
            softmax_width = query_projection_size
        else:
            softmax_width = heads_per_partition * seqlen_per_cp
        activation_width += softmax_width

        return tokens_per_rank * (activation_width + ln_width) * bytes_per_value

    def _simulate_mla_gemms(self, batch_tokens: int, dtype: str, tp_size: int = 1) -> float:
        """Simulate MLA (Multi-Latent Attention) projection GEMMs (forward-only).

        MLA uses LoRA-factored Q and compressed KV projections instead of
        standard Q/K/V projections:
          Forward  (6 GEMMs): Q_down, Q_up, KV_down, KV_up, RoPE_proj, O_proj

        Tensor parallelism shards the head axis only. The down-projections
        produce the shared latent and are replicated on every rank; the
        up-projections are column-parallel over heads and the output projection
        is row-parallel over them, so only those carry ``heads // tp``.
        """
        args = self.config.model_config
        backend = self._gemm_backend

        hidden = args.hidden_size
        heads = max(1, args.num_attention_heads // max(1, tp_size))
        q_lora_rank = args.q_lora_rank
        kv_lora_rank = args.kv_lora_rank
        qk_head_dim = args.qk_head_dim
        qk_pos_emb_head_dim = args.qk_pos_emb_head_dim
        v_head_dim = args.v_head_dim

        fwd_time = 0.0
        T = batch_tokens

        # ---------- Forward ----------
        if q_lora_rank is not None:
            # Q down-proj: [T, hidden] × [hidden, q_lora_rank]
            q_down_out = q_lora_rank
            r = backend.simulate_gemm(T, q_down_out, hidden, dtype)
            fwd_time += r.forward_time_ms
            # Q up-proj: [T, q_lora_rank] × [q_lora_rank, heads*(qk_hd+qk_pe_hd)]
            q_up_out = heads * (qk_head_dim + qk_pos_emb_head_dim)
            r = backend.simulate_gemm(T, q_up_out, q_lora_rank, dtype)
            fwd_time += r.forward_time_ms
        else:
            # Direct Q projection (no LoRA): [T, hidden] × [hidden, heads*(qk_hd+qk_pe_hd)]
            q_up_out = heads * (qk_head_dim + qk_pos_emb_head_dim)
            r = backend.simulate_gemm(T, q_up_out, hidden, dtype)
            fwd_time += r.forward_time_ms

        # KV down-proj: [T, hidden] × [hidden, kv_lora_rank]
        kv_down_out = kv_lora_rank
        r = backend.simulate_gemm(T, kv_down_out, hidden, dtype)
        fwd_time += r.forward_time_ms
        # KV up-proj: [T, kv_lora_rank] × [kv_lora_rank, heads*(qk_hd+v_hd)]
        kv_up_out = heads * (qk_head_dim + v_head_dim)
        r = backend.simulate_gemm(T, kv_up_out, kv_lora_rank, dtype)
        fwd_time += r.forward_time_ms

        # RoPE positional embedding projection: [T, hidden] × [hidden, qk_pos_emb_head_dim]
        r = backend.simulate_gemm(T, qk_pos_emb_head_dim, hidden, dtype)
        fwd_time += r.forward_time_ms

        # Output projection: [T, heads*v_hd] × [heads*v_hd, hidden]
        o_in = heads * v_head_dim
        r = backend.simulate_gemm(T, hidden, o_in, dtype)
        fwd_time += r.forward_time_ms

        return fwd_time

    def _get_simulated_results(self, batch_size: int, seq_len: int) -> tuple[float, int]:
        """Get simulated results from GEMM + SDPA simulation backends."""
        args = self.config.model_config
        mp = self.config.model_parallel_config
        tp_size = max(1, mp.tensor_model_parallel_size)
        cp_size = max(1, mp.context_model_parallel_size)

        # Data-parallel attention splits the in-flight requests across dp groups
        # and tensor-parallelises attention only inside one, so a rank sees 1/dp
        # of the batch across tp/dp of the heads. Everything below -- projection
        # GEMMs, the SDPA roofline, the KV bytes read, the activation working
        # set -- follows from those two numbers. For head-parallel work the two
        # divisions cancel, which is why GQA gains nothing here; MLA's latent is
        # shared by every head rather than sharded across them, so for it they
        # do not.
        attn_dp = max(1, getattr(mp, "attention_data_parallel_size", 1) or 1)
        if attn_dp > 1:
            tp_size = max(1, tp_size // attn_dp)
            # Rounded up, matching the KV-cache estimate: a batch that does not
            # divide evenly leaves some rank holding the extra request, and the
            # engines pad every rank to the longest one rather than letting a
            # step finish early. Rounding down charged a batch of 58 over 8 dp
            # ranks for 7 sequences while memory had already provisioned 8.
            batch_size = max(1, -(-batch_size // attn_dp))

        # Tensor parallelism shards attention *heads*, not tokens: every rank
        # holds the whole token axis and a 1/tp slice of the heads. Dividing the
        # token count by ``tp`` instead (and then sizing the projection GEMMs
        # with the *full* head count, as below) gets the FLOPs right by accident
        # but the shape and the weight bytes wrong -- each rank was charged for
        # streaming the entire Q/K/V/O weight matrices. At decode those bytes
        # are the whole cost, which is why the modelled attention time barely
        # moved from TP=1 to TP=8 while the measured ladder shards strongly.
        # Context parallelism does shard the token axis, so it stays.
        batch_tokens = max(1, batch_size * seq_len // cp_size)
        slen_per_cp = max(1, seq_len // cp_size)

        fwd_time = 0.0

        # 1. Simulate linear projection GEMMs using GEMM backend
        if self._gemm_backend is not None:
            gemm_dtype = "fp8" if getattr(args, "fp8", None) else "bf16"

            if getattr(args, "multi_latent_attention", False):
                # MLA: LoRA-factored Q and compressed KV projections (6 GEMMs)
                fwd_time += self._simulate_mla_gemms(batch_tokens, gemm_dtype, tp_size)
            else:
                # Standard attention: Q, K, V, O projections (4 GEMMs)
                num_query_groups = (
                    args.num_query_groups
                    if args.group_query_attention and args.num_query_groups
                    else args.num_attention_heads
                )
                # Per-rank head slices: TP splits Q heads and KV groups, so each
                # rank's projection weights are 1/tp of the full matrices.
                gemm_result = self._gemm_backend.simulate_attention_gemms(
                    batch_tokens=batch_tokens,
                    hidden_size=args.hidden_size,
                    num_attention_heads=max(1, args.num_attention_heads // tp_size),
                    kv_channels=args.kv_channels,
                    num_query_groups=max(1, num_query_groups // tp_size),
                    dtype=gemm_dtype,
                )
                fwd_time += gemm_result.forward_time_ms

        # 2. Simulate SDPA core computation using SDPA backend
        if self._sdpa_backend is not None:
            heads_per_rank = max(1, args.num_attention_heads // tp_size)

            if getattr(args, "multi_latent_attention", False):
                # MLA: Q·Kᵀ uses qk_head_dim + qk_pos_emb_head_dim (e.g. 192),
                #       P·V  uses v_head_dim (e.g. 128).
                sdpa_head_dim = args.qk_head_dim + args.qk_pos_emb_head_dim
                sdpa_head_dim_v = args.v_head_dim
            else:
                sdpa_head_dim = args.kv_channels
                sdpa_head_dim_v = None  # same as head_dim

            if self._inference_phase is not None:
                # Forward-only serving: query length = slen_per_cp (1 for
                # decode, chunk size for chunked prefill), KV length = the
                # resident cache.  GQA stores fewer KV heads.
                num_query_groups = (
                    args.num_query_groups
                    if args.group_query_attention and args.num_query_groups
                    else args.num_attention_heads
                )
                kv_heads_per_rank = max(1, num_query_groups // tp_size)
                kv_len = self._kv_seq_len if self._kv_seq_len is not None else slen_per_cp
                # Bytes the KV cache holds per token on this rank, for the SDPA
                # HBM roofline. MLA caches one compressed latent
                # (``kv_lora_rank`` + the RoPE dims) that every head shares and
                # that TP replicates rather than shards, so it is nothing like
                # the per-head K+V footprint the head counts would imply --
                # for DeepSeek-R1 at TP=8 the difference is ~9x.
                kv_bpe = _KV_CACHE_BYTES.get(
                    str(getattr(args, "kv_cache_dtype", "") or "").lower(), 2.0
                )
                if getattr(args, "multi_latent_attention", False):
                    kv_bytes_per_token = (
                        args.kv_lora_rank + args.qk_pos_emb_head_dim
                    ) * kv_bpe
                else:
                    kv_bytes_per_token = (
                        kv_heads_per_rank * (sdpa_head_dim + (sdpa_head_dim_v or sdpa_head_dim)) * kv_bpe
                    )
                # Prefill keeps causal masking; decode (single query) attends
                # to the whole cache so masking is irrelevant.
                causal = self._inference_phase == "prefill"
                sdpa_result = self._sdpa_backend.simulate_sdpa(
                    batch_size=batch_size,
                    num_heads=heads_per_rank,
                    seq_len=slen_per_cp,
                    head_dim=sdpa_head_dim,
                    causal=causal,
                    dtype="bf16",
                    seq_len_kv=int(kv_len),
                    num_heads_kv=kv_heads_per_rank,
                    kv_bytes_per_token=kv_bytes_per_token,
                    head_dim_v=sdpa_head_dim_v,
                )
                fwd_time += sdpa_result.forward_time_ms
            else:
                sdpa_result = self._sdpa_backend.simulate_sdpa(
                    batch_size=batch_size,
                    num_heads=heads_per_rank,
                    seq_len=slen_per_cp,
                    head_dim=sdpa_head_dim,
                    causal=True,
                    dtype="bf16",
                    head_dim_v=sdpa_head_dim_v,
                )
                fwd_time += sdpa_result.forward_time_ms

        activation_memory = self.estimated_activation_memory(batch_size, seq_len)
        return (fwd_time, activation_memory)

    def _get_benchmark_results(self, batch_size: int, seq_len: int) -> tuple[float, int]:
        """Get or compute benchmark results (cached)."""
        cache_key = (batch_size, seq_len, self._inference_phase, self._kv_seq_len)

        if self._cached_results is None or self._cache_key != cache_key:
            if self._gemm_backend is not None or self._sdpa_backend is not None:
                # Use simulation mode
                self._cached_results = self._get_simulated_results(batch_size, seq_len)
            else:
                # Use actual GPU benchmarking. torch is imported here rather than
                # at module scope because it costs ~0.66 s and only this branch
                # needs it; see the note in embedding.py.
                import torch

                # Context parallel / Sequence parallel adjustment
                cp_size = self.config.model_parallel_config.context_model_parallel_size
                # Effective sequence length per rank if CP is used
                slen_per_cp = seq_len // cp_size


                # Imported here, not at module scope: this pulls in torch, which costs
                # ~0.66 s and is only needed to benchmark on a real GPU. A simulate-only
                # projection should not pay for it -- Hyperloom spawns one process per
                # config, where that import dwarfed the ~28 ms the projection takes.
                from .utils import benchmark_layer
                self._cached_results = benchmark_layer(
                    self.module,
                    [
                        (seq_len, batch_size, self.config.model_config.hidden_size),
                        ((1, 1, slen_per_cp, seq_len), torch.bool),
                    ],
                )
            self._cache_key = cache_key
        return self._cached_results

    def measured_forward_time(self, batch_size: int, seq_len: int) -> float:
        forward_time, _ = self._get_benchmark_results(batch_size, seq_len)
        return forward_time

    def measured_activation_memory(self, batch_size: int, seq_len: int) -> int:
        _, activation_memory = self._get_benchmark_results(batch_size, seq_len)
        return activation_memory
