###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional


@dataclass
class RuntimeConfig:
    global_batch_size: int = 1
    micro_batch_size: int = 1
    sequence_length: int = 0
    data_parallel_size: int = 1


@dataclass
class ModelParallelConfig:
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: int = 1
    context_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    use_torch_fsdp2: bool = False
    use_distributed_optimizer: bool = False
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = False
    # Pipeline stage layer distribution
    decoder_first_pipeline_num_layers: int = None
    decoder_last_pipeline_num_layers: int = None
    pipeline_model_parallel_layout: str = None
    # Recomputation settings
    recompute_granularity: str = None  # "full" or "selective"
    recompute_num_layers: int = 0
    # Megatron selective block recompute: global transformer layer indices (0..num_layers-1)
    recompute_layer_ids: Optional[List[int]] = None
    # Precision-aware optimizer (Megatron `--use-precision-aware-optimizer`).
    # When enabled the optimizer state dtypes follow the *_dtype fields below;
    # the projection's bytes-per-param formula uses these to size the static
    # block correctly instead of assuming default fp32 main params + fp32 m + fp32 v.
    use_precision_aware_optimizer: bool = False
    main_grads_dtype: str = "fp32"  # fp32 | bf16 | fp16
    exp_avg_dtype: str = "fp32"  # 1st moment dtype (fp32 | bf16 | fp16)
    exp_avg_sq_dtype: str = "fp32"  # 2nd moment dtype (fp32 | bf16 | fp16)


@dataclass
class ModelConfig:
    num_layers: int = 0
    hidden_size: int = 0
    padded_vocab_size: int = 0
    ffn_hidden_size: int = 0
    # attention
    num_attention_heads: int = 0
    kv_channels: int = 0
    group_query_attention: bool = False
    num_query_groups: int = 0
    qk_layernorm: bool = False
    multi_latent_attention: bool = False
    use_flash_attn: bool = False
    qk_head_dim: int = 0
    qk_pos_emb_head_dim: int = 0
    v_head_dim: int = 0
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    # Sliding-window (local) attention: each query attends to at most the last
    # ``sink_sliding_window`` KV tokens (0 = full/global attention). Several
    # models (gpt-oss, Mistral, Gemma-2/3, Qwen2.5) cap the attention span this
    # way, which bounds both decode attention compute and KV-cache footprint at
    # long context. ``sink_window_even_layers_only`` models the common
    # interleave where only alternating layers are windowed (the rest stay
    # full-attention), i.e. half the layers.
    sink_sliding_window: int = 0
    sink_window_even_layers_only: bool = False
    # FFN & MoE
    swiglu: bool = False
    num_experts: int = 0
    moe_ffn_hidden_size: int = 0
    moe_pattern: list = None
    moe_router_topk: int = 0
    moe_shared_expert_intermediate_size: int = 0
    # How unevenly the router spreads tokens over experts, as the exponent of a
    # Zipf popularity law (0 = uniform). It sets how many *distinct* experts a
    # decode step touches, which is what a weight-bandwidth-bound MoE step costs.
    # Relates to a measured max/mean expert load by ``I(s) = N / H_N(s)``, so an
    # imbalance figure from a profile converts straight into this.
    #
    # Default 0 -- uniform -- because for a load-balanced router that is very
    # nearly right, and the alternative is worse than it looks. Real gpt-oss vLLM
    # sweeps under forced Zipf routing confirm a decode step is linear in this
    # count to +-1.5% at ~0.06 ms per expert, so the count is the whole story.
    # But at gpt-oss's 128 experts and top-4, Zipf s=0.3 (a 3.1x max/mean load,
    # already more skew than a router with an aux balance loss shows) gives 49.4
    # distinct experts at batch 16 against uniform's 51.0 -- a 3% difference.
    # Realistic imbalance simply does not move the distinct count much.
    #
    # Worth stating because fitting this to the measured TP ladder *does* find a
    # clean optimum, at s=0.8 for 6.7% MAPE against 9.5% at uniform. That is a
    # 14.6x max/mean expert load, which no load-balanced router has, so what the
    # fit is really doing is absorbing an unrelated error into a knob that
    # happens to have the right shape. The mid-batch residual is not routing skew.
    moe_routing_skew: float = 0.0
    # Misc
    share_embeddings_and_output_weights: bool = False
    # Precision – None means bf16, "hybrid" means FP8-hybrid (linear GEMMs in FP8)
    fp8: str = None

    # Primus Turbo flags — used to select the grouped-GEMM performance model
    enable_primus_turbo: bool = False
    use_turbo_grouped_mlp: bool = False
    # MoE simulate-kernel model: None -> auto (turbo/legacy); "vllm_fused" models
    # the vLLM fused-MoE decode kernel; "legacy"/"turbo" force the Megatron paths.
    moe_sim_kernel: str = None
    use_turbo_deepep: bool = False  # DeepEP enables async A2A with compute overlap
    turbo_sync_free_moe_stage: int = 0  # 0=off, 1=fused router, 2=+DeepEP+grouped, 3=+fused act
    # MoE routing imbalance (hottest EP-rank / mean token load, >= 1.0) and its
    # redundant-experts mitigation, mirrored from the inference request config so
    # the expert-GEMM simulator can apply it *inside* the roofline (scales the
    # busiest rank's token count M, a near no-op at weight-bound decode and full
    # at compute-bound prefill). EP-gated: only meaningful when EP > 1.
    ep_load_balance: float = 1.0
    redundant_experts: int = 0

    # Loss fusion – fuses cross-entropy with output layer avoiding full logits materialisation
    cross_entropy_loss_fusion: bool = False


@dataclass
class TrainingConfig:
    """
    Configuration for training the profiler models.
    """

    model_config: ModelConfig
    runtime_config: RuntimeConfig
    model_parallel_config: ModelParallelConfig


# ─────────────────────────────────────────────────────────────────────────────
# Inference / serving configuration
# ─────────────────────────────────────────────────────────────────────────────


def dtype_num_bytes(dtype: Optional[str]) -> float:
    """Return the byte width of a (loosely-named) tensor dtype.

    Accepts the informal names used throughout the projection layer
    (``bf16``, ``fp16``, ``fp8``, ``int8``, ``fp32``).  Unknown / ``None``
    values fall back to bf16 (2 bytes) which is the projection default.

    ``mxfp4`` is block-scaled: 4-bit elements plus one E8M0 scale per block of
    32, i.e. 4 + 8/32 bits = 0.53125 bytes per element. That 6% over the raw
    nibble is the difference between a checkpoint fitting and not.
    """
    if dtype is None:
        return 2.0
    key = str(dtype).lower().strip()
    return {
        "fp32": 4.0,
        "float32": 4.0,
        "bf16": 2.0,
        "bfloat16": 2.0,
        "fp16": 2.0,
        "float16": 2.0,
        "half": 2.0,
        "fp8": 1.0,
        "fp8_e4m3": 1.0,
        "fp8_e5m2": 1.0,
        "int8": 1.0,
        "uint8": 1.0,
        "fp4": 0.5,
        "int4": 0.5,
        "mxfp4": 0.53125,
        "mxfp8": 1.03125,
    }.get(key, 2.0)


@dataclass
class InferenceRequestConfig:
    """Describes the *serving* workload an inference projection targets.

    Unlike :class:`RuntimeConfig` (which models a training microbatch /
    global-batch / gradient-accumulation pipeline) this captures the
    request profile that drives prefill + autoregressive decode.
    """

    # Prompt (prefill) and generation (decode) lengths, in tokens.
    input_seq_len: int = 1024
    output_seq_len: int = 128
    # Number of sequences processed together in one forward (a decode batch).
    batch_size: int = 1
    # Max number of sequences whose KV cache is resident at once (for memory
    # sizing / continuous batching).  Defaults to ``batch_size``.
    max_concurrency: Optional[int] = None
    # Largest context (prompt + generated) any sequence can reach.  Drives
    # KV-cache capacity.  Defaults to ``input_seq_len + output_seq_len``.
    max_context_len: Optional[int] = None
    # Fraction of per-GPU HBM the serving engine may use (vLLM
    # ``gpu_memory_utilization`` / SGLang ``mem_fraction_static``).  Bounds the
    # usable HBM for weights + KV + activations and therefore the max concurrent
    # sequences.  ``None`` = use the full HBM capacity (legacy behaviour).
    kv_cache_memory_fraction: Optional[float] = None
    # Paged-KV block (page) size in tokens.  Real serving engines allocate KV in
    # fixed-size blocks (vLLM ``block_size``, typically 16), so a sequence's
    # context is rounded UP to a whole number of blocks — the last partially
    # filled block still reserves a full block.  This fragmentation inflates the
    # per-sequence KV footprint and lowers max concurrency.  ``0`` = no paging
    # (continuous/contiguous allocation, legacy behaviour).
    kv_block_size: int = 0

    # ---- Precision ----
    weight_dtype: str = "bf16"     # weights kept resident (bf16 | fp8 | ...)
    kv_cache_dtype: str = "bf16"   # KV cache precision (bf16 | fp8 | int8 | ...)

    # ---- Inference features ----
    # Chunked prefill: split a long prompt into chunks of this many tokens
    # (0 disables; the whole prompt is one forward).
    chunked_prefill_size: int = 0
    # Prefix-cache hit rate in [0, 1): the fraction of each prompt's tokens
    # already resident in the paged KV cache (an automatically-cached / reused
    # shared prefix). Cached tokens skip prefill compute -- only the non-cached
    # suffix is run through the network -- so TTFT and the prefill share of
    # continuous-batching pollution scale with ``(1 - prefix_cache_hit_rate)``.
    # The suffix tokens still attend over the FULL context (cached prefix +
    # suffix), so decode/KV sizing is unchanged. ``0`` = cold cache (legacy
    # behaviour). Models vLLM/SGLang automatic prefix caching and Infera's
    # cross-request prefix reuse (shared system prompts, agentic multi-turn).
    prefix_cache_hit_rate: float = 0.0
    # Speculative decoding: number of draft tokens proposed per verify step
    # (0 disables) and the expected acceptance rate in [0, 1].
    speculative_num_tokens: int = 0
    speculative_acceptance_rate: float = 0.0

    # ---- Serving / continuous-batching dynamics ----
    # How decode latency is modelled:
    #   "continuous" — continuous batching with mixed prefill+decode steps
    #                  (models the TPOT "pollution" real servers like vLLM see).
    #   "static"     — an idealized batch doing pure decode (legacy behaviour;
    #                  prefill is charged once as TTFT only).
    serving_model: str = "continuous"
    # Fixed per-decode-step host/launch overhead (microseconds). At low decode
    # batch the step is launch-bound; CUDA-graph capture shrinks this. Added to
    # every decode/mixed step. 0 = ignore (pure kernel-compute model).
    decode_step_overhead_us: float = 0.0
    # Per-kernel launch latency (microseconds) for the small-tensor launch-bound
    # floor used by the pure-simulate (GPU-free) decode path. A decode step
    # issues ~``kernels_per_layer`` kernels per layer plus a handful for the
    # head/sampling; at low batch the roofline GEMM/attention times underflow
    # the real step, which is set by host-side kernel-launch dispatch. The floor
    # is ``n_kernels * kernel_launch_latency_us`` and is applied as
    # ``step = max(step, floor)`` in simulation only. CUDA-graph replay collapses
    # the launches, so a "full"/"piecewise" ``cudagraph_mode`` disables it and
    # ``fused_kernels`` shrinks it. 0 = disabled (legacy behaviour).
    kernel_launch_latency_us: float = 0.0
    # Representative number of distinct kernel launches per transformer layer
    # (norms, projections, attention, activation, residuals, MoE router/permute/
    # grouped-GEMM/combine). Only consulted for the launch-latency floor above.
    kernels_per_layer: int = 12
    # Per-kernel GPU *occupancy* (microseconds): the minimum wall time a kernel
    # holds the GPU regardless of how little data it touches (wave launch,
    # memory latency, barriers). This is distinct from
    # ``kernel_launch_latency_us`` above in two ways that matter physically:
    #
    #   * it survives CUDA-graph capture. Graph replay removes the *host-side*
    #     dispatch, but the kernels still execute and still occupy the device.
    #   * it is *additive*, not a ``max``. A decode step runs a few large
    #     data-bound kernels (the expert GEMMs) alongside many small
    #     latency-bound ones (norms, rope, KV write, router, activation,
    #     residuals). The large ones are priced by bytes and the small ones by
    #     occupancy, so the two costs are paid together. Taking ``max`` of the
    #     totals makes the small-kernel cost vanish exactly when the batch is
    #     large, which is where it was measured to still be present.
    #
    # Measured on MI355X by solving ``step(tp) = floor + compute(1)/tp`` across
    # TP=1,2,4,8 with real weights (bench/hyperloom_validation/
    # measure_step_floor.sh), after subtracting the *measured* per-step
    # all-reduce from each TP>1 rung. Subtracting it matters: TP=1 runs no
    # collective at all, so a floor fit that leaves comm in absorbs it and the
    # same milliseconds then get charged twice once comm is modelled explicitly.
    # That is what the earlier 5.96 us did -- it came from a 2.61 ms floor that
    # silently contained ~0.8 ms of all-reduce.
    #
    # With comm removed the floor is flat where it should be: 1.83-2.07 ms over
    # batch 1..64 on the TP=8 rung, which is the most floor-dominated one. The
    # default is that mean, 1.91 ms, over the kernel count
    # ``36 * kernels_per_layer + 6 = 438``: 4.35 us.
    #
    # This is one architecture's ladder divided by an assumed kernel count, so
    # it is a per-kernel figure only if the count is right. measure_kernel_floor.py
    # measures the count and the per-kernel minimum directly and should replace
    # it; until then, treat cross-model agreement as the test of whether this is
    # hardware physics or one model's tuning. 0 = disabled.
    decode_kernel_occupancy_us: float = 4.35
    # Per-output-token host cost for detokenization + response streaming
    # (microseconds/token). The serving harness (vLLM / InferenceX) measures ITL
    # client-side, so its per-token latency carries detok+stream that the GPU
    # decode step does not. Added to ITL/TPOT and end-to-end latency only -- it
    # overlaps the next server step, so aggregate throughput is unchanged.
    # 0 = ignore (pure GPU-step model).
    detokenize_overhead_us: float = 0.0
    # Per-prompt-token host cost for tokenizing the request text (microseconds/
    # token). The serving harness (vLLM / InferenceX) sends the prompt as text
    # and the server tokenizes it after the TTFT clock starts, so it lands in
    # TTFT. Added to TTFT + end-to-end latency only. 0 = ignore (GPU-prefill
    # model). Symmetric with ``detokenize_overhead_us`` on the decode side.
    tokenize_overhead_us: float = 0.0
    # Output tokens buffered per streaming flush (vLLM/SGLang
    # ``--stream-interval``). The server detokenizes and flushes the SSE stream
    # only every N tokens, so the client's *first* token -- and therefore the
    # measured TTFT -- lands only after N tokens have been decoded. Serving
    # harnesses raise this to cut host detokenization cost, and it is a leading
    # TTFT term at high N: SGLang's PD default of 30 adds ~29 TPOTs. ``1`` =
    # flush every token (legacy behaviour, no TTFT contribution).
    stream_interval: int = 1
    # Decode-scheduler admission granularity, in decode steps. A request whose
    # prefill is done is not admitted to the running batch immediately: the
    # scheduler polls its receive queue only every ``--scheduler-recv-interval``
    # loop iterations, and each iteration advances ``--num-continuous-decode-
    # steps`` decode steps, so the product bounds the admission delay. The
    # request waits a uniform fraction of that window on average. This is a pure
    # TTFT term (throughput is unaffected -- that is the point of the knob) and
    # it dominates TTFT in disaggregated serving, where prefill compute is off
    # the critical path. ``0`` = admit immediately (legacy behaviour).
    decode_admission_steps: int = 0
    # Extra cost fraction applied to a *mixed* (prefill+decode) step to model
    # vLLM's less-efficient PIECEWISE CUDA-graph path vs the FULL graph used for
    # uniform pure-decode steps. 0 = no penalty.
    mixed_batch_penalty: float = 0.0
    # CUDA-graph capture strategy. A friendly preset over the two low-level
    # knobs above:
    #   "none"      — eager, launch-bound per step (high per-step overhead).
    #   "piecewise" — piecewise graphs; mixed prefill+decode steps fall off the
    #                 captured graph (moderate overhead + a mixed-step penalty).
    #   "full"      — one full graph capture (minimal overhead, no penalty).
    # ``None`` leaves ``decode_step_overhead_us`` / ``mixed_batch_penalty`` at
    # their explicit values (legacy behaviour). An explicit non-zero value of
    # either low-level knob overrides the preset.
    cudagraph_mode: Optional[str] = None
    # Scheduler per-step token budget (vLLM ``--max-num-batched-tokens``).  The
    # scheduler caps the total tokens processed in one engine step: a mixed step
    # carries a prefill chunk plus the decode tokens of every running sequence,
    # and that sum may not exceed this cap.  Oversized steps are split, raising
    # the number of prefill steps (and the mixed-step fraction) → higher TPOT /
    # lower throughput.  ``0`` = unlimited (legacy behaviour).
    max_num_batched_tokens: int = 0
    # MoE expert routing imbalance.  On an EP-sharded model the MoE step time is
    # set by the BUSIEST rank, not the average: ``ep_load_balance`` is the ratio
    # of the hottest rank's token load to the average (1.0 = perfectly
    # balanced; 1.3 = the hottest rank does 1.3x the mean).  Inflates the
    # realized MoE expert-compute (grouped-GEMM) time.
    ep_load_balance: float = 1.0
    # Extra replicated expert slots (EPLB / redundant experts) that spread the
    # hottest experts' tokens and so reduce the realized imbalance above.
    # ``0`` = no redundancy (legacy behaviour).
    redundant_experts: int = 0

    # ---- Offered load / request rate (open-loop arrivals) ----
    # Offered load in requests/sec. ``0`` keeps the legacy *closed-loop* model
    # (a fixed resident ``concurrency`` with no arrival queue). When > 0 (and
    # ``arrival_model`` != "closed") a first-order queueing-delay term is added
    # to TTFT: the engine sustains a finite request-completion rate, and offered
    # load approaching that rate inflates waiting time (the latency–throughput
    # knee). This is the primary latency/throughput trade-off knob.
    request_rate: float = 0.0
    # Arrival process for ``request_rate``: "closed" (no queue; legacy),
    # "poisson" (M/M/1-style waiting time), or "deterministic" (D/M/1-ish,
    # lighter queueing). Only consulted when ``request_rate`` > 0.
    arrival_model: str = "closed"

    # ---- Kernel backend selection (ROCm) ----
    # Selectable attention/kernel library. Best choice is shape/recipe
    # dependent on ROCm; modelled as a representative multiplier on attention
    # compute time (simulation path only). ``None`` = engine default (= triton
    # baseline, 1.0). Values: aiter | triton | ck | hip.
    attention_backend: Optional[str] = None

    # ---- Native sparse attention (DeepSeek V3.2 / V4 NSA) ----
    # Number of KV tokens each query attends to under native sparse attention
    # (NSA + indexer top-k selection). ``0`` = dense attention (legacy). When
    # > 0 and the context exceeds it, attention compute/KV-read scales toward
    # ``topk / context`` (plus a small indexer overhead), making long-context
    # attention roughly constant in context length.
    sparse_attention_topk: int = 0

    # ---- Sliding-window / local attention ----
    # Override the model's sliding-window size (KV tokens each query attends to;
    # 0 = full attention). ``None`` follows the model config's
    # ``sink_sliding_window``. When set (or inherited) and the context exceeds
    # the window, decode/prefill attention compute and KV-cache footprint are
    # bounded by the window instead of the full context.
    sliding_window: Optional[int] = None
    # Fraction of attention layers that are windowed (the remainder use full
    # attention) for models that interleave local/global layers. ``None``
    # follows the model's ``sink_window_even_layers_only`` flag (0.5 when set,
    # else 1.0 = every layer windowed).
    sliding_window_layer_fraction: Optional[float] = None

    # ---- MoE expert compute precision ----
    # Expert grouped-GEMM compute dtype (separate from ``weight_dtype`` which
    # sizes resident weights). Models the expert-MLP speedup of low-precision
    # expert kernels: mxfp4 | fp8 | bf16. ``None`` = follow bf16 (no speedup).
    moe_expert_dtype: Optional[str] = None

    # ---- Runtime activation quantization / cast ----
    # Precision the runtime casts activations to before each low-precision GEMM
    # (fp8 | mxfp4). These cast kernels (read bf16, amax, write packed + scale)
    # are memory-bound overhead the GEMM simulator does not see. ``None``
    # auto-detects from ``weight_dtype`` / model fp8; ``"bf16"`` / ``"none"``
    # drops the cast term (e.g. a bf16 serving path).
    act_quant_dtype: Optional[str] = None

    # ---- Speculative decoding draft cost ----
    # Draft-model forward cost per proposed draft token, as a fraction of one
    # target decode step. The draft runs ``speculative_num_tokens`` times per
    # verify step; this charges that extra compute (``0`` = ignore the draft
    # cost, the legacy behaviour that only credited the accepted-token speedup).
    speculative_draft_cost_factor: float = 0.0

    # ---- Token sampling / logits post-processing ----
    # The per-step cost of turning LM-head logits into sampled tokens (a
    # memory-bound reduction over the vocabulary). ``sampling_enabled=False``
    # drops the term (e.g. to compare against a logits-free step). ``top_k`` /
    # ``top_p`` / ``temperature`` shape how many streaming passes the sampler
    # makes over the logits (greedy/argmax is the cheapest).
    sampling_enabled: bool = True
    sampling_top_k: int = 0
    sampling_top_p: float = 1.0
    sampling_temperature: float = 1.0

    # ---- Fused custom ops (RMSNorm / RoPE / quant / KV-store+quant) ----
    # Fused elementwise kernels mainly cut per-step kernel-launch overhead.
    # When set, the resolved per-decode-step launch overhead is reduced by
    # ``_FUSED_KERNEL_OVERHEAD_FACTOR`` (representative). No-op when there is no
    # per-step overhead to cut (e.g. full CUDA graph already amortises it).
    fused_kernels: bool = False

    def resolved_max_context_len(self) -> int:
        if self.max_context_len is not None:
            return int(self.max_context_len)
        return int(self.input_seq_len) + int(self.output_seq_len)

    def resolved_max_concurrency(self) -> int:
        if self.max_concurrency is not None:
            return int(self.max_concurrency)
        return int(self.batch_size)

    def resolved_prefix_cache_hit_rate(self) -> float:
        """Prefix-cache hit rate clamped to ``[0, 0.999]``.

        Capped just below 1.0 so at least one prompt token is always prefilled
        (a request with a 100%-cached prompt still runs one forward to produce
        its first token). ``0`` = cold cache.
        """
        h = float(self.prefix_cache_hit_rate or 0.0)
        if h <= 0.0:
            return 0.0
        return min(h, 0.999)

    def resolved_decode_step_overhead_us(self) -> float:
        """Per-decode-step launch overhead, honoring the cudagraph preset.

        An explicit non-zero ``decode_step_overhead_us`` always wins; otherwise
        the ``cudagraph_mode`` preset (if any) supplies a representative value.
        Fused elementwise kernels (``fused_kernels``) further cut whatever
        launch overhead remains.
        """
        if self.decode_step_overhead_us:
            base = float(self.decode_step_overhead_us)
        else:
            base = _CUDAGRAPH_PRESETS.get(self.cudagraph_mode, (0.0, 0.0))[0]
        if self.fused_kernels:
            base *= _FUSED_KERNEL_OVERHEAD_FACTOR
        return base

    def resolved_kernel_launch_floor_ms(self, num_layers: int) -> float:
        """Small-tensor launch-latency floor (ms) for the pure-simulate decode.

        A launch-bound decode step's wall time is set by dispatching one kernel
        per op rather than the (tiny) small-M GEMM/attention compute. Estimated
        as ``n_kernels * kernel_launch_latency_us`` where ``n_kernels`` scales
        with model depth. Returns 0.0 when disabled, or when CUDA-graph capture
        collapses the launches (``cudagraph_mode`` full/piecewise). Fused
        elementwise kernels shrink the launch count.
        """
        lat_us = float(self.kernel_launch_latency_us or 0.0)
        if lat_us <= 0.0:
            return 0.0
        # Graph replay issues a single launch for the captured region.
        if str(self.cudagraph_mode or "").lower() in ("full", "piecewise"):
            return 0.0
        kpl = max(1, int(self.kernels_per_layer or 12))
        # + a handful for embedding / final norm / LM head / sampling.
        n_kernels = max(1, int(num_layers)) * kpl + 6
        if self.fused_kernels:
            n_kernels *= _FUSED_KERNEL_OVERHEAD_FACTOR
        return n_kernels * lat_us / 1000.0

    def resolved_decode_occupancy_ms(self, num_layers: int) -> float:
        """Additive per-kernel GPU occupancy for one decode step (ms).

        Same kernel count as the launch floor, but priced at the device-side
        occupancy minimum and *not* cancelled by graph capture: replay removes
        the host dispatch, not the kernels. Fusing genuinely removes kernels, so
        ``fused_kernels`` still shrinks the count.
        """
        occ_us = float(self.decode_kernel_occupancy_us or 0.0)
        if occ_us <= 0.0:
            return 0.0
        kpl = max(1, int(self.kernels_per_layer or 12))
        n_kernels = max(1, int(num_layers)) * kpl + 6
        if self.fused_kernels:
            n_kernels *= _FUSED_KERNEL_OVERHEAD_FACTOR
        return n_kernels * occ_us / 1000.0

    def resolved_mixed_batch_penalty(self) -> float:
        """Mixed-step penalty fraction, honoring the cudagraph preset."""
        if self.mixed_batch_penalty:
            return float(self.mixed_batch_penalty)
        return _CUDAGRAPH_PRESETS.get(self.cudagraph_mode, (0.0, 0.0))[1]

    def resolved_ep_imbalance(self, num_experts: int = 0) -> float:
        """Effective MoE expert-compute imbalance multiplier (always >= 1.0).

        ``ep_load_balance`` is the hottest-rank / mean token-load ratio; the MoE
        step is gated by that busiest rank, so its expert-compute time scales by
        this factor.  ``redundant_experts`` replicates the hottest expert slots,
        diluting the surplus the hot rank carries: the excess ``(ratio - 1)`` is
        shrunk by ``num_experts / (num_experts + redundant_experts)`` (the share
        of routed mass that still lands on a single, non-replicated slot).

        ``1.0`` (perfectly balanced) is a no-op.  ``num_experts`` is supplied by
        the caller (it lives on :class:`ModelConfig`, not the request config).
        """
        bal = float(self.ep_load_balance or 1.0)
        if bal <= 1.0:
            return 1.0
        red = max(0, int(self.redundant_experts or 0))
        if num_experts > 0 and red > 0:
            bal = 1.0 + (bal - 1.0) * num_experts / (num_experts + red)
        return max(1.0, bal)

    def resolved_attention_backend_multiplier(self) -> float:
        """Attention-compute time multiplier for the selected kernel backend.

        ``None`` / unknown → 1.0 (no change). Representative ROCm ratios
        relative to the Triton baseline; override by measuring (benchmark mode)
        for a real number.
        """
        if not self.attention_backend:
            return 1.0
        return _ATTENTION_BACKEND_PRESETS.get(str(self.attention_backend).lower(), 1.0)

    def resolved_sparse_attention_scale(self, context_len: int) -> float:
        """Fraction of dense attention cost under NSA top-k selection.

        Dense (``sparse_attention_topk == 0``) or short context (≤ topk) → 1.0.
        Otherwise attention work scales toward ``topk / context`` with a small
        floor (projections + indexer that do not shrink with the KV window).
        """
        topk = int(self.sparse_attention_topk or 0)
        if topk <= 0 or context_len <= topk:
            return 1.0
        return max(_SPARSE_ATTENTION_FLOOR, float(topk) / float(max(1, context_len)))

    def resolved_sliding_window(self, model_window: int = 0) -> int:
        """Sliding-window size in KV tokens (0 = full attention).

        An explicit ``sliding_window`` override wins; otherwise inherit the
        model's ``sink_sliding_window``. Non-positive values disable windowing.
        """
        w = self.sliding_window
        if w is None:
            w = model_window
        w = int(w or 0)
        return w if w > 0 else 0

    def resolved_sliding_window_fraction(self, even_layers_only: bool = False) -> float:
        """Fraction of attention layers that are windowed, clamped to [0, 1].

        ``sliding_window_layer_fraction`` wins when set; otherwise 0.5 for an
        even-layers-only interleave (e.g. gpt-oss) or 1.0 when every layer is
        windowed.
        """
        f = self.sliding_window_layer_fraction
        if f is None:
            f = 0.5 if even_layers_only else 1.0
        return min(1.0, max(0.0, float(f)))

    def effective_attn_kv(
        self, kv_len: int, model_window: int = 0, even_layers_only: bool = False
    ) -> int:
        """KV length an average attention layer reads under sliding-window.

        Full attention (or short context) returns ``kv_len`` unchanged. When a
        window is active and the context exceeds it, windowed layers read only
        ``window`` tokens while full layers still read ``kv_len``; the two are
        blended by the windowed-layer fraction so a single length represents the
        average layer (attention decode cost is ~linear in KV read). Used for
        both decode and prefill attention compute.
        """
        kv = int(max(1, kv_len))
        w = self.resolved_sliding_window(model_window)
        if w <= 0 or kv <= w:
            return kv
        frac = self.resolved_sliding_window_fraction(even_layers_only)
        return int(round(frac * w + (1.0 - frac) * kv))

    def resolved_moe_expert_dtype_speedup(self) -> float:
        """Expert grouped-GEMM compute multiplier for the expert dtype.

        ``None`` / bf16 → 1.0. fp8 ≈ 0.55 (~2x), mxfp4 ≈ 0.35; representative,
        override via benchmark calibration for a measured number.
        """
        if not self.moe_expert_dtype:
            return 1.0
        return _MOE_EXPERT_DTYPE_SPEEDUP.get(str(self.moe_expert_dtype).lower(), 1.0)

    def resolved_act_quant_dtype(self, model_fp8=None) -> Optional[str]:
        """Precision of the runtime activation cast, or ``None`` to disable.

        An explicit ``act_quant_dtype`` wins (``"bf16"`` / ``"none"`` disables);
        otherwise infer from ``weight_dtype`` (fp8 / mxfp4) or the model's fp8
        flag. bf16 serving has no runtime activation cast.
        """
        v = self.act_quant_dtype
        if v is None:
            wd = str(self.weight_dtype or "").lower()
            if "mxfp4" in wd:
                return "mxfp4"
            if "fp8" in wd:
                return "fp8"
            if model_fp8:
                return "fp8"
            return None
        v = str(v).lower()
        return None if v in ("none", "bf16", "") else v


# Representative attention-backend compute multipliers (relative to Triton).
_ATTENTION_BACKEND_PRESETS = {
    "aiter": 0.85,
    "ck": 0.90,
    "triton": 1.0,
    "hip": 1.10,
}

# Floor on the sparse-attention scale (projections + indexer don't shrink).
_SPARSE_ATTENTION_FLOOR = 0.15

# Representative expert grouped-GEMM compute multipliers by expert dtype.
_MOE_EXPERT_DTYPE_SPEEDUP = {
    "bf16": 1.0,
    "fp8": 0.55,
    "fp8_e4m3": 0.55,
    "mxfp4": 0.35,
    "fp4": 0.35,
}

# Fused elementwise kernels cut per-step launch overhead by this factor.
_FUSED_KERNEL_OVERHEAD_FACTOR = 0.7


# CUDA-graph presets → (decode_step_overhead_us, mixed_batch_penalty).
# Representative, ROCm-order-of-magnitude values; override with the explicit
# low-level knobs for a measured number.
_CUDAGRAPH_PRESETS = {
    "none": (40.0, 0.0),
    "piecewise": (8.0, 0.15),
    "full": (3.0, 0.0),
}


@dataclass
class InferenceCollectiveConfig:
    """Knobs for the explicit inference communication model (feature B).

    When :attr:`enabled` the inference performance projector replaces the
    layer profiler's *implicit* TP-AllReduce / EP-AllToAll cost with an
    explicit, reportable communication model.  At default values it
    reproduces the implicit cost; the knobs below let a user model
    **custom collective ops** — forcing a specific algorithm, hiding comm
    behind compute (overlap), or applying a fused-op speedup (e.g.
    AllReduce+RMSNorm fusion, DeepEP-style overlapped dispatch/combine).
    """

    enabled: bool = True
    # Algorithm override for the TP AllReduce / EP AllToAll. ``auto`` lets the
    # collective model pick the fastest; otherwise force a specific algorithm.
    tp_allreduce_algo: str = "auto"  # auto | ring | one_shot | two_shot | hierarchical
    ep_a2a_algo: str = "auto"  # auto | direct | single_shot | hierarchical
    # Fraction of communication hidden behind compute (0 = none, fully
    # exposed; 1 = fully overlapped). Set per phase.
    prefill_overlap: float = 0.0
    decode_overlap: float = 0.0
    # Custom fused-op efficiency multipliers applied to comm time
    # (<1.0 = faster, models kernel fusion / better algorithms). 1.0 = none.
    tp_allreduce_efficiency: float = 1.0
    ep_a2a_efficiency: float = 1.0
    # ROCm "quick reduce": low-latency quantized all-reduce for small messages.
    # When set, applies an extra speedup multiplier to the TP AllReduce time
    # (models the small-message latency win). ``False`` = standard all-reduce.
    quick_reduce: bool = False
    # Fused RMSNorm + AllReduce: communication–computation fusion that hides
    # part of the TP all-reduce latency behind the norm. When set, applies an
    # extra speedup multiplier to the TP AllReduce time. ``False`` = unfused.
    fuse_rmsnorm_allreduce: bool = False
    # Whether to charge pipeline-stage P2P (send/recv) latency. Only nonzero
    # when pipeline_model_parallel_size > 1.
    include_pp_p2p: bool = True
    # Optional hardware overrides forwarded to ``get_default_args`` (node_bw,
    # pod_bw, bw_eff, latencies, ...). ``None`` uses the model defaults.
    hardware_config: Optional[Dict] = None


@dataclass
class DisaggregationConfig:
    """Prefill/decode disaggregation (feature A).

    Models separate prefill and decode worker pools, each with its own
    parallelism, plus the KV-cache transfer cost incurred when a request
    migrates from a prefill worker to a decode worker.  When disabled the
    projector runs the standard colocated two-phase model.
    """

    enabled: bool = False
    # Per-pool parallelism overrides. ``None`` falls back to the shared
    # ``model_parallel_config`` values.
    prefill_tp: Optional[int] = None
    prefill_pp: Optional[int] = None
    prefill_ep: Optional[int] = None
    decode_tp: Optional[int] = None
    decode_pp: Optional[int] = None
    decode_ep: Optional[int] = None
    # Number of replicas in each pool (for aggregate-throughput / GPU split).
    prefill_replicas: int = 1
    decode_replicas: int = 1
    # KV-cache transfer link. ``None`` bw uses the inter-node (pod) bandwidth
    # from the collective model; latency is a fixed per-transfer overhead (us).
    kv_transfer_bw_gbps: Optional[float] = None
    kv_transfer_latency_us: float = 0.0
    # Friendly preset over the two link knobs above, naming the KV-transfer
    # engine: "nixl", "mooncake", or "mori".  ``None`` leaves the explicit link
    # values untouched.  An explicit non-zero/non-None link knob overrides the
    # preset value for that field.
    transfer_backend: Optional[str] = None

    def resolved_kv_transfer_bw_gbps(self) -> Optional[float]:
        if self.kv_transfer_bw_gbps:
            return float(self.kv_transfer_bw_gbps)
        return _TRANSFER_BACKEND_PRESETS.get(self.transfer_backend, (None, 0.0))[0]

    def resolved_kv_transfer_latency_us(self) -> float:
        if self.kv_transfer_latency_us:
            return float(self.kv_transfer_latency_us)
        return _TRANSFER_BACKEND_PRESETS.get(self.transfer_backend, (None, 0.0))[1]

    def prefill_parallel(self, mp: ModelParallelConfig) -> "ModelParallelConfig":
        return _override_parallel(mp, self.prefill_tp, self.prefill_pp, self.prefill_ep)

    def decode_parallel(self, mp: ModelParallelConfig) -> "ModelParallelConfig":
        return _override_parallel(mp, self.decode_tp, self.decode_pp, self.decode_ep)


# KV-transfer engine presets → (kv_transfer_bw_gbps, kv_transfer_latency_us).
# Representative effective point-to-point KV link numbers; override with the
# explicit link knobs for a measured fabric.
_TRANSFER_BACKEND_PRESETS = {
    "nixl": (400.0, 5.0),
    "mooncake": (200.0, 10.0),
    "mori": (300.0, 7.0),
}


def _override_parallel(
    mp: ModelParallelConfig, tp: Optional[int], pp: Optional[int], ep: Optional[int]
) -> ModelParallelConfig:
    """Return a copy of ``mp`` with TP/PP/EP optionally overridden."""
    from copy import copy

    out = copy(mp)
    if tp is not None:
        out.tensor_model_parallel_size = int(tp)
    if pp is not None:
        out.pipeline_model_parallel_size = int(pp)
    if ep is not None:
        out.expert_model_parallel_size = int(ep)
    return out


@dataclass
class InferenceConfig:
    """Configuration for inference / serving projection.

    Reuses the same :class:`ModelConfig` / :class:`ModelParallelConfig` as
    training (so the existing profiler tree can estimate forward compute and
    parameter counts) but swaps the training :class:`RuntimeConfig` for an
    :class:`InferenceRequestConfig`.
    """

    model_config: ModelConfig
    request_config: InferenceRequestConfig
    model_parallel_config: ModelParallelConfig
    collective_config: InferenceCollectiveConfig = field(
        default_factory=InferenceCollectiveConfig
    )
    disaggregation_config: DisaggregationConfig = field(
        default_factory=DisaggregationConfig
    )

    def as_training_config(self, *, batch_size: int, seq_len: int) -> TrainingConfig:
        """Build a throwaway :class:`TrainingConfig` view for a given
        (batch, seq_len) so the existing profiler tree can be reused for
        forward-only compute estimation.
        """
        runtime = RuntimeConfig(
            global_batch_size=batch_size,
            micro_batch_size=batch_size,
            sequence_length=seq_len,
            data_parallel_size=1,
        )
        return TrainingConfig(
            model_config=self.model_config,
            runtime_config=runtime,
            model_parallel_config=self.model_parallel_config,
        )


def update_config_from_args(config, args):
    for field in fields(config):
        if hasattr(args, field.name):
            setattr(config, field.name, getattr(args, field.name))
    return config


def megatron_derive_default_args(args):
    world_size = int(os.getenv("NNODES", "1")) * int(os.getenv("GPUS_PER_NODE", "8"))
    if args.kv_channels is None:
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if not args.group_query_attention:
        # If GQA not set, treat as per-head queries
        args.num_query_groups = args.num_attention_heads

    if not hasattr(args, "context_parallel_size") or args.context_parallel_size is None:
        args.context_parallel_size = 1
    if not hasattr(args, "data_parallel_size") or args.data_parallel_size is None:
        args.data_parallel_size = world_size // (
            args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
        )
    if not hasattr(args, "virtual_pipeline_model_parallel_size"):
        args.virtual_pipeline_model_parallel_size = None
    if not hasattr(args, "num_layers_per_virtual_pipeline_stage"):
        args.num_layers_per_virtual_pipeline_stage = None
    if (
        args.num_layers_per_virtual_pipeline_stage is None
        and args.virtual_pipeline_model_parallel_size is None
    ):
        args.virtual_pipeline_model_parallel_size = 1
    elif args.num_layers_per_virtual_pipeline_stage is not None:
        args.virtual_pipeline_model_parallel_size = args.num_layers // (
            args.num_layers_per_virtual_pipeline_stage * args.pipeline_model_parallel_size
        )

    args.share_embeddings_and_output_weights = not args.untie_embeddings_and_output_weights

    if args.num_experts is None:
        args.moe_pattern = [0] * args.num_layers
    else:
        if isinstance(args.moe_layer_freq, int):
            args.moe_pattern = [1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)]
        elif isinstance(args.moe_layer_freq, list):
            args.moe_pattern = args.moe_layer_freq
        elif isinstance(args.moe_layer_freq, str):
            try:
                parsed = eval(args.moe_layer_freq)
            except Exception:
                raise ValueError(f"Invalid moe_layer_freq format: {args.moe_layer_freq}")

            # Handle case where eval returns an int (e.g., "1" -> 1 means all layers are MoE)
            if isinstance(parsed, int):
                if parsed == 1:
                    # All layers are MoE
                    args.moe_pattern = [1] * args.num_layers
                else:
                    # Every Nth layer is MoE
                    args.moe_pattern = [1 if (i % parsed == 0) else 0 for i in range(args.num_layers)]
            elif isinstance(parsed, list):
                # Handle list-based moe_layer_freq pattern
                if len(parsed) > args.num_layers:
                    # Truncate to first num_layers elements (for proxy models with fewer layers)
                    # This is safe: we're using a subset of the pattern for faster profiling
                    args.moe_pattern = parsed[: args.num_layers]
                elif len(parsed) < args.num_layers:
                    # If the pattern is shorter than num_layers, this is likely an error
                    # (config specifies fewer layers than requested)
                    raise ValueError(
                        f"moe_layer_freq pattern has {len(parsed)} elements but num_layers={args.num_layers}. "
                        f"The pattern length must match or exceed num_layers. "
                        f"Pattern: {parsed}"
                    )
                else:
                    # Exact match - use as-is (normal case for full model)
                    args.moe_pattern = parsed
            else:
                raise ValueError(f"Invalid moe_layer_freq format after eval: {type(parsed)}")

    # naming conversion
    args.sequence_length = args.seq_length
    args.context_model_parallel_size = args.context_parallel_size

    # Use model's vocab size if set, otherwise default to 100352
    if not hasattr(args, "padded_vocab_size") or args.padded_vocab_size is None:
        args.padded_vocab_size = 100352

    return args


def convert_config_to_projection_config(config) -> TrainingConfig:
    args = config.get_module_config("pre_trainer")
    framework = getattr(args, "framework", "")
    if framework == "megatron":
        args = megatron_derive_default_args(args)
    else:
        raise NotImplementedError(f"Unsupported framework: {framework}")

    model_config = update_config_from_args(ModelConfig(), args)
    runtime_config = update_config_from_args(RuntimeConfig(), args)
    model_parallel_config = update_config_from_args(ModelParallelConfig(), args)

    training_config = TrainingConfig(
        model_config=model_config,
        runtime_config=runtime_config,
        model_parallel_config=model_parallel_config,
    )

    return training_config


def convert_config_to_inference_config(
    config,
    *,
    inference_overrides: Optional[dict] = None,
) -> InferenceConfig:
    """Build an :class:`InferenceConfig` from an experiment config.

    Reuses :func:`convert_config_to_projection_config` for the model
    and parallelism config, then layers an :class:`InferenceRequestConfig`
    on top.  ``inference_overrides`` (typically parsed from CLI flags) takes
    precedence over any ``inference:`` block embedded in the YAML.
    """
    training_config = convert_config_to_projection_config(config)

    # Allow an optional ``inference:`` block in the pre_trainer module config
    # (so a single workload YAML can carry a default serving profile).
    args = config.get_module_config("pre_trainer")
    yaml_inf = getattr(args, "inference", None) or {}
    if not isinstance(yaml_inf, dict):
        yaml_inf = {}

    request = InferenceRequestConfig()
    # 1) seed from the training seq_length so a bare config still works.
    if getattr(training_config.runtime_config, "sequence_length", 0):
        request.input_seq_len = int(training_config.runtime_config.sequence_length)
    # 2) apply YAML inference block, then 3) CLI overrides.
    overrides = inference_overrides or {}
    for source in (yaml_inf, overrides):
        for f in fields(request):
            if f.name in source and source[f.name] is not None:
                setattr(request, f.name, source[f.name])

    # ---- Collective (feature B) + disaggregation (feature A) configs ----
    collective = InferenceCollectiveConfig()
    disagg = DisaggregationConfig()
    # YAML may carry nested ``collective:`` / ``disaggregation:`` blocks; CLI
    # overrides arrive flattened (e.g. ``collective_*`` / ``disagg_*`` keys).
    yaml_coll = yaml_inf.get("collective") if isinstance(yaml_inf.get("collective"), dict) else {}
    yaml_disagg = (
        yaml_inf.get("disaggregation") if isinstance(yaml_inf.get("disaggregation"), dict) else {}
    )
    _apply_fields(collective, yaml_coll)
    _apply_fields(disagg, yaml_disagg)
    _apply_prefixed(collective, overrides, prefix="collective_")
    _apply_prefixed(disagg, overrides, prefix="disagg_")

    return InferenceConfig(
        model_config=training_config.model_config,
        request_config=request,
        model_parallel_config=training_config.model_parallel_config,
        collective_config=collective,
        disaggregation_config=disagg,
    )


def _apply_fields(target, source: dict) -> None:
    """Set dataclass fields on ``target`` from matching keys in ``source``."""
    if not source:
        return
    valid = {f.name for f in fields(target)}
    for key, val in source.items():
        if key in valid and val is not None:
            setattr(target, key, val)


def _apply_prefixed(target, source: dict, *, prefix: str) -> None:
    """Set dataclass fields from ``source`` keys of the form ``<prefix><field>``."""
    if not source:
        return
    valid = {f.name for f in fields(target)}
    for key, val in source.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix):]
        if name in valid and val is not None:
            setattr(target, name, val)
