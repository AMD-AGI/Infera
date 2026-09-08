"""Inference / serving tuning: trial config, legality, seed plan, objectives.

This is the inference-mode counterpart to ``legality.py`` + ``plan.py`` (which
target distributed *training*).  It defines the serving search space — the
knobs that actually move TTFT / inter-token latency / throughput / KV-cache
capacity — and a deterministic seed sweep over them.

The oracle is ``inferasim projection inference`` (see
``infera.projection.core.projection.inference_projection``); the evaluator builds
the command and parses the metrics (see
``evaluator.Evaluator.evaluate_inference``).

Search axes (vs. the training axes):
  * **tp / pp / ep / cp** — serving parallelism (no backward, no optimizer).
  * **batch_size / max_concurrency** — continuous-batching depth (replaces the
    training GBS/MBS/num_microbatches pipeline-fill identity).
  * **weight_dtype / kv_cache_dtype** — weight + KV quantization (fp8/int8).
  * **chunked_prefill_size** — bound prefill latency / enable batching.
  * **speculative_num_tokens / acceptance_rate** — speculative decoding.

Training-only axes (recompute, pipeline-schedule bubble tuning, FSDP2,
distributed optimizer, overlap_grad_reduce, SyncFree) are intentionally absent.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import OptimizationConfig, TargetCluster
from .workload import ArchitectureRecord


# ---------------------------------------------------------------------------
# Trial config
# ---------------------------------------------------------------------------

@dataclass
class InferenceTrialConfig:
    """All knobs the inference tuner sweeps."""

    # serving parallelism
    tp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1
    # Attention data parallelism: subdivides the TP group so a rank owns whole
    # requests rather than a slice of every request's heads. 1 = off. This is
    # not a minor knob for the models being tuned here -- MLA caches one latent
    # that every head reads, so tensor parallelism replicates it and the axis is
    # what stops a TP8 replica storing the same cache eight times.
    attention_dp: int = 1
    # request / batching profile
    batch_size: int = 1
    # Share of each prompt already in the KV cache from an earlier turn.
    prefix_cache_hit_rate: float = 0.0
    input_len: int = 1024
    output_len: int = 128
    max_concurrency: int | None = None
    # precision
    weight_dtype: str = "bf16"
    kv_cache_dtype: str = "bf16"
    # Precision of the attention projections and the dense MLP, which the
    # resident weight dtype does not settle: a 4-bit checkpoint quantizes those
    # alongside the experts, and without saying so they stream fp8 while memory
    # sizes them at 4. None leaves the projector's own auto-detection alone.
    linear_weight_dtype: str | None = None
    # serving features
    chunked_prefill_size: int = 0
    speculative_num_tokens: int = 0
    speculative_acceptance_rate: float = 0.0
    # feature B: custom collective ops
    tp_allreduce_algo: str = "auto"
    ep_a2a_algo: str = "auto"
    # MoE A2A backend: DeepEP overlaps dispatch/combine behind expert compute
    use_turbo_deepep: bool = False
    # CUDA-graph capture preset: none | piecewise | full (None = engine default)
    cudagraph_mode: str | None = None
    # fraction of HBM the engine may use (bounds usable HBM + max concurrency)
    kv_cache_memory_fraction: float | None = None
    # paged-KV block size in tokens (0 = no paging; 16 = vLLM default)
    kv_block_size: int = 0
    # scheduler per-step token budget (0 = unlimited)
    max_num_batched_tokens: int = 0
    # MoE expert routing imbalance (1.0 = balanced) + redundant-expert mitigation
    ep_load_balance: float = 1.0
    redundant_experts: int = 0
    # feature A: prefill/decode disaggregation
    disaggregate: bool = False
    prefill_tp: int | None = None
    decode_tp: int | None = None
    decode_replicas: int = 1
    # KV-transfer engine preset for disaggregation: nixl | mooncake | mori
    transfer_backend: str | None = None
    # offered load (open-loop): request rate + arrival process
    request_rate: float = 0.0
    arrival_model: str = "closed"  # closed | poisson | deterministic
    # kernel backend (ROCm attention library): aiter | triton | ck | hip
    attention_backend: str | None = None
    # native sparse attention (DeepSeek V3.2/V4 NSA) top-k KV (0 = dense)
    sparse_attention_topk: int = 0
    # MoE expert compute precision (mxfp4 | fp8 | bf16); MoE-only
    moe_expert_dtype: str | None = None
    # fused elementwise kernels (RMSNorm/RoPE/quant/KV-store) cut step overhead
    fused_kernels: bool = False
    # speculative draft-model forward cost per draft token (fraction of a step)
    speculative_draft_cost_factor: float = 0.0
    # custom collective ops (TP>1): quick-reduce + fused RMSNorm+AllReduce
    quick_reduce: bool = False
    fuse_rmsnorm_allreduce: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "InferenceTrialConfig":
        f = cls()
        for k in f.as_dict():
            if k in d and d[k] is not None:
                setattr(f, k, d[k])
        return f

    def signature(self) -> str:
        d = self.as_dict()
        return ",".join(f"{k}={d[k]}" for k in sorted(d))


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------

def _tier_tops(batch: int, dp: int, n: int = 6) -> list[int]:
    """Concurrencies at the top of an attention-DP tier, largest first.

    With attention DP a rank holds ``ceil(concurrency / dp)`` sequences, so TTFT
    steps up at each multiple of ``dp`` and then improves across the tier. The
    best concurrency is therefore always a multiple of ``dp``; anything else
    pays a tier's latency while using less of it.

    The ladder reaches well below the batch size because admission is the knob
    that buys TPOT back. A large batch that misses its TPOT budget at full
    admission is usually not the wrong batch -- it is the right batch admitting
    too many sequences at once, and only a lower tier reveals that.

    It also reaches above it. Concurrency is how many requests are in flight,
    not how many run in a step, and a scheduler holding more than it runs is
    ordinary continuous batching -- the surplus is what keeps the next step full.
    Capping the ladder at the batch size hid the entire region where these
    models are actually fastest, and it is the memory model, not the batch, that
    says how far it can go.
    """
    if dp <= 1 or batch < dp:
        return [batch] if batch > 0 else []
    out: list[int] = []
    for frac in (2.0, 1.5, 1.25, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125):
        top = (int(batch * frac) // dp) * dp
        if top >= dp and top not in out:
            out.append(top)
        if len(out) >= n:
            break
    return out


def min_draft_cost(acceptance: float) -> float:
    """Cheapest draft model that could plausibly hit this acceptance rate.

    Acceptance and draft cost describe one object -- the draft model -- but the
    search treats them as independent, so left uncoupled it buys accuracy for
    nothing: acceptance 0.95 at a draft cost of 0.10 was accepted as legal, and
    the optimizer duly took it.

    What a draft has to buy is the *odds* of acceptance, and those get
    superlinearly expensive near certainty because a draft that is always right
    is the target model. Pricing the odds gives ~0.12 of a step at acceptance
    0.7 and ~0.45 at 0.9, and passes 1.0 (a draft costing a full target step,
    i.e. no reason to speculate) just below 0.96.
    """
    if acceptance <= 0.0:
        return 0.0
    if acceptance >= 1.0:
        return float("inf")
    return 0.05 * acceptance / (1.0 - acceptance)


def _divisors(n: int, max_val: int | None = None) -> list[int]:
    if n <= 0:
        return [1]
    out = [d for d in range(1, n + 1) if n % d == 0]
    if max_val is not None:
        out = [d for d in out if d <= max_val]
    return out


def _powers_of_two(max_val: int) -> list[int]:
    out = [1]
    while out[-1] * 2 <= max_val:
        out.append(out[-1] * 2)
    return out


@dataclass
class InferenceAxisLegality:
    tp: list[int]
    pp: list[int]
    ep: list[int]
    cp: list[int]
    # Candidate degrees across every legal TP; a trial's own value must divide
    # its TP, which ``validate_inference`` checks against ``cfg.tp``.
    attention_dp: list[int]
    batch_size: list[int]
    weight_dtype: list[str]
    kv_cache_dtype: list[str]
    chunked_prefill_size: list[int]
    speculative_num_tokens: list[int]
    linear_weight_dtype: list[str] = field(
        default_factory=lambda: ["bf16", "fp8", "mxfp4", "fp4"]
    )
    tp_allreduce_algo: list[str] = field(
        default_factory=lambda: ["auto", "ring", "one_shot", "two_shot", "hierarchical"]
    )
    ep_a2a_algo: list[str] = field(
        default_factory=lambda: ["auto", "direct", "single_shot", "hierarchical"]
    )
    use_turbo_deepep: list[bool] = field(default_factory=lambda: [False])
    cudagraph_mode: list[str] = field(
        default_factory=lambda: ["none", "piecewise", "full"]
    )
    kv_cache_memory_fraction: list[float] = field(
        default_factory=lambda: [0.8, 0.85, 0.9]
    )
    transfer_backend: list[str] = field(
        default_factory=lambda: ["nixl", "mooncake", "mori"]
    )
    kv_block_size: list[int] = field(default_factory=lambda: [0, 16, 32])
    max_num_batched_tokens: list[int] = field(
        default_factory=lambda: [0, 2048, 8192]
    )
    ep_load_balance: list[float] = field(default_factory=lambda: [1.0, 1.2, 1.5])
    redundant_experts: list[int] = field(default_factory=lambda: [0, 8, 16])
    arrival_model: list[str] = field(
        default_factory=lambda: ["closed", "poisson", "deterministic"]
    )
    attention_backend: list[str] = field(
        default_factory=lambda: ["aiter", "triton", "ck", "hip"]
    )
    sparse_attention_topk: list[int] = field(default_factory=lambda: [0, 512, 2048])
    moe_expert_dtype: list[str] = field(
        default_factory=lambda: ["bf16", "fp8", "mxfp4"]
    )
    # fused elementwise kernels (always available)
    fused_kernels: list[bool] = field(default_factory=lambda: [False, True])
    # TP-collective optimizations (only meaningful when TP>1)
    quick_reduce: list[bool] = field(default_factory=lambda: [False])
    fuse_rmsnorm_allreduce: list[bool] = field(default_factory=lambda: [False])

    def to_prompt_dict(self) -> dict:
        return {
            "tp": self.tp, "pp": self.pp, "ep": self.ep, "cp": self.cp,
            "attention_dp": self.attention_dp,
            "batch_size": self.batch_size,
            "weight_dtype": self.weight_dtype,
            "kv_cache_dtype": self.kv_cache_dtype,
            "linear_weight_dtype": self.linear_weight_dtype,
            "chunked_prefill_size": self.chunked_prefill_size,
            "speculative_num_tokens": self.speculative_num_tokens,
            "tp_allreduce_algo": self.tp_allreduce_algo,
            "ep_a2a_algo": self.ep_a2a_algo,
            "use_turbo_deepep": self.use_turbo_deepep,
            "cudagraph_mode": self.cudagraph_mode,
            "kv_cache_memory_fraction": self.kv_cache_memory_fraction,
            "transfer_backend": self.transfer_backend,
            "kv_block_size": self.kv_block_size,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "ep_load_balance": self.ep_load_balance,
            "redundant_experts": self.redundant_experts,
            "arrival_model": self.arrival_model,
            "attention_backend": self.attention_backend,
            "sparse_attention_topk": self.sparse_attention_topk,
            "moe_expert_dtype": self.moe_expert_dtype,
            "fused_kernels": self.fused_kernels,
            "quick_reduce": self.quick_reduce,
            "fuse_rmsnorm_allreduce": self.fuse_rmsnorm_allreduce,
        }


def derive_inference_legality(
    arch: ArchitectureRecord, cluster: TargetCluster
) -> InferenceAxisLegality:
    world = cluster.num_nodes * cluster.gpus_per_node
    gpn = cluster.gpus_per_node

    tp = sorted(set(_divisors(arch.num_attention_heads, gpn))
                & set(_divisors(arch.hidden_size, gpn))) or [1]
    pp = _divisors(arch.num_layers, world) if arch.num_layers else [1]
    is_moe = bool(getattr(arch, "is_moe", False))
    if is_moe and arch.num_experts:
        ep = _divisors(arch.num_experts, world) or [1]
    else:
        ep = [1]
    cp = [1]  # context parallel is rarely used for serving; keep simple

    # Sparse attention is a property of the checkpoint before it is a knob: a
    # model trained with an indexer is not served dense, and offering only the
    # generic widths would leave its own out of reach.
    # Only the width this model was actually trained with, plus off. The
    # generic 512/2048 ladder used to be offered to every model, which handed a
    # dense-attention checkpoint an indexer it does not have -- at 130k context
    # that is a sixtyfold reduction in attention reads, so the search took it
    # every time and reported a throughput the model cannot reach. Sparsity is
    # an architectural property, not a serving knob to be turned up.
    model_topk = int(getattr(arch, "index_topk", 0) or 0)
    sparse_topk = sorted({0} | ({model_topk} if model_topk else set()))

    # Attention DP subdivides the TP group, so the degrees worth offering are
    # the divisors of the widest TP on the table. A trial pairs one with its own
    # TP and ``validate_inference`` rejects the pairings that do not divide.
    attention_dp = _divisors(max(tp)) if max(tp) > 1 else [1]

    # Concurrency / decode batch depth.
    batch_size = _powers_of_two(256)

    # A frontier MoE at 130k does not fit on this part at 8 or 16 bits, so a
    # search that cannot say "fp4" cannot say anything about it: every trial is
    # rejected on memory before its performance is ever read.
    weight_dtype = ["bf16", "fp8", "fp4"]
    kv_cache_dtype = ["bf16", "fp8", "int8"]
    chunked_prefill_size = [0, 512, 1024, 2048]
    speculative_num_tokens = [0, 2, 4]

    # DeepEP only matters for MoE (it overlaps the EP All-to-All); offer the
    # on/off choice only when the workload is MoE.
    use_turbo_deepep = [False, True] if is_moe else [False]

    # quick-reduce + fused RMSNorm/AllReduce are TP-collective optimizations;
    # only offer the on/off choice when TP can exceed 1.
    tp_collective = [False, True] if max(tp) > 1 else [False]

    return InferenceAxisLegality(
        tp=tp, pp=pp, ep=ep, cp=cp,
        attention_dp=attention_dp,
        batch_size=batch_size,
        weight_dtype=weight_dtype,
        kv_cache_dtype=kv_cache_dtype,
        chunked_prefill_size=chunked_prefill_size,
        speculative_num_tokens=speculative_num_tokens,
        sparse_attention_topk=sparse_topk,
        use_turbo_deepep=use_turbo_deepep,
        quick_reduce=tp_collective,
        fuse_rmsnorm_allreduce=tp_collective,
    )


def validate_inference(
    cfg: InferenceTrialConfig,
    arch: ArchitectureRecord,
    cluster: TargetCluster,
    legality: InferenceAxisLegality,
) -> tuple[bool, str]:
    world = cluster.num_nodes * cluster.gpus_per_node
    if cfg.tp not in legality.tp:
        return False, f"TP={cfg.tp} not in legal set {legality.tp}"
    if cfg.pp not in legality.pp:
        return False, f"PP={cfg.pp} not in legal set {legality.pp}"
    if cfg.ep not in legality.ep:
        return False, f"EP={cfg.ep} not in legal set {legality.ep}"
    if cfg.attention_dp not in legality.attention_dp:
        return False, (
            f"attention_dp={cfg.attention_dp} not in legal set {legality.attention_dp}"
        )
    if cfg.attention_dp > 1 and cfg.tp % cfg.attention_dp != 0:
        # The axis splits the TP group; a degree it does not divide describes no
        # rank layout, and the projector refuses it rather than rounding.
        return False, (
            f"attention_dp={cfg.attention_dp} must divide TP={cfg.tp}"
        )
    if cfg.batch_size <= 0:
        return False, f"batch_size must be positive, got {cfg.batch_size}"
    if cfg.weight_dtype not in legality.weight_dtype:
        return False, f"weight_dtype={cfg.weight_dtype} not in {legality.weight_dtype}"
    if cfg.kv_cache_dtype not in legality.kv_cache_dtype:
        return False, f"kv_cache_dtype={cfg.kv_cache_dtype} not in {legality.kv_cache_dtype}"
    if (cfg.linear_weight_dtype is not None
            and cfg.linear_weight_dtype not in legality.linear_weight_dtype):
        return False, (
            f"linear_weight_dtype={cfg.linear_weight_dtype} not in "
            f"{legality.linear_weight_dtype}"
        )
    if (cfg.linear_weight_dtype is not None
            and cfg.linear_weight_dtype not in legality.linear_weight_dtype):
        return False, (
            f"linear_weight_dtype={cfg.linear_weight_dtype} not in "
            f"{legality.linear_weight_dtype}"
        )
    if cfg.speculative_num_tokens < 0:
        return False, "speculative_num_tokens must be >= 0"
    if not getattr(arch, "is_moe", False) and cfg.ep > 1:
        return False, "EP>1 is only meaningful for MoE workloads"
    if cfg.use_turbo_deepep and not getattr(arch, "is_moe", False):
        return False, "use_turbo_deepep is only meaningful for MoE workloads"
    if cfg.tp_allreduce_algo not in legality.tp_allreduce_algo:
        return False, f"tp_allreduce_algo={cfg.tp_allreduce_algo} not in {legality.tp_allreduce_algo}"
    if cfg.ep_a2a_algo not in legality.ep_a2a_algo:
        return False, f"ep_a2a_algo={cfg.ep_a2a_algo} not in {legality.ep_a2a_algo}"
    if cfg.cudagraph_mode is not None and cfg.cudagraph_mode not in legality.cudagraph_mode:
        return False, f"cudagraph_mode={cfg.cudagraph_mode} not in {legality.cudagraph_mode}"
    if cfg.kv_cache_memory_fraction is not None and not (0.0 < cfg.kv_cache_memory_fraction <= 1.0):
        return False, f"kv_cache_memory_fraction={cfg.kv_cache_memory_fraction} must be in (0, 1]"
    if cfg.transfer_backend is not None:
        if not cfg.disaggregate:
            return False, "transfer_backend only applies when disaggregate is set"
        if cfg.transfer_backend not in legality.transfer_backend:
            return False, f"transfer_backend={cfg.transfer_backend} not in {legality.transfer_backend}"
    if cfg.kv_block_size < 0:
        return False, f"kv_block_size must be >= 0, got {cfg.kv_block_size}"
    if cfg.max_num_batched_tokens < 0:
        return False, f"max_num_batched_tokens must be >= 0, got {cfg.max_num_batched_tokens}"
    is_moe = bool(getattr(arch, "is_moe", False))
    if cfg.ep_load_balance < 1.0:
        return False, f"ep_load_balance must be >= 1.0, got {cfg.ep_load_balance}"
    if cfg.ep_load_balance != 1.0 and not is_moe:
        return False, "ep_load_balance is only meaningful for MoE workloads"
    if cfg.redundant_experts < 0:
        return False, f"redundant_experts must be >= 0, got {cfg.redundant_experts}"
    if cfg.redundant_experts > 0 and not is_moe:
        return False, "redundant_experts is only meaningful for MoE workloads"
    # Offered load / request rate.
    if cfg.request_rate < 0:
        return False, f"request_rate must be >= 0, got {cfg.request_rate}"
    if cfg.arrival_model not in legality.arrival_model:
        return False, f"arrival_model={cfg.arrival_model} not in {legality.arrival_model}"
    if cfg.request_rate > 0 and cfg.arrival_model == "closed":
        return False, "request_rate>0 requires arrival_model in {poisson, deterministic}"
    # Kernel backend (attention library).
    if cfg.attention_backend is not None and cfg.attention_backend not in legality.attention_backend:
        return False, f"attention_backend={cfg.attention_backend} not in {legality.attention_backend}"
    # Native sparse attention.
    if cfg.sparse_attention_topk < 0:
        return False, f"sparse_attention_topk must be >= 0, got {cfg.sparse_attention_topk}"
    # MoE expert compute precision (MoE-only).
    if cfg.moe_expert_dtype is not None:
        if cfg.moe_expert_dtype not in legality.moe_expert_dtype:
            return False, f"moe_expert_dtype={cfg.moe_expert_dtype} not in {legality.moe_expert_dtype}"
        if not is_moe:
            return False, "moe_expert_dtype is only meaningful for MoE workloads"
    # Speculative draft cost requires speculative decoding to be on.
    if cfg.speculative_draft_cost_factor < 0:
        return False, "speculative_draft_cost_factor must be >= 0"
    if cfg.speculative_draft_cost_factor > 0 and cfg.speculative_num_tokens <= 0:
        return False, "speculative_draft_cost_factor requires speculative_num_tokens>0"
    # Speculation has to pay for itself. Acceptance and draft cost are free
    # parameters, so an optimizer handed a perfect draft model that runs for
    # nothing will take it -- and it did, tripling reported throughput on a
    # configuration no draft model can implement. A draft that is always right
    # is the target model, and running it is not free.
    if cfg.speculative_num_tokens > 0:
        if not (0.0 < cfg.speculative_acceptance_rate < 1.0):
            return False, (
                f"speculative_acceptance_rate={cfg.speculative_acceptance_rate} "
                "must be in (0, 1) when speculative decoding is on"
            )
        floor = min_draft_cost(cfg.speculative_acceptance_rate)
        if cfg.speculative_draft_cost_factor <= 0:
            return False, (
                "speculative decoding must charge a draft cost: "
                "speculative_draft_cost_factor must be > 0"
            )
        if cfg.speculative_draft_cost_factor < floor:
            return False, (
                f"speculative_draft_cost_factor={cfg.speculative_draft_cost_factor} "
                f"is too cheap for acceptance_rate={cfg.speculative_acceptance_rate}: "
                f"a draft that accurate costs at least {floor:.2f} of a step"
            )
    # Custom collective ops only matter when TP>1.
    if cfg.quick_reduce and cfg.tp <= 1:
        return False, "quick_reduce requires TP>1"
    if cfg.fuse_rmsnorm_allreduce and cfg.tp <= 1:
        return False, "fuse_rmsnorm_allreduce requires TP>1"

    replica_gpus = cfg.tp * cfg.pp * (cfg.ep if getattr(arch, "is_moe", False) else 1)
    if replica_gpus > world:
        return False, (
            f"replica needs {replica_gpus} GPUs but only {world} available "
            f"({cluster.num_nodes}×{cluster.gpus_per_node})"
        )

    # Feature A: disaggregation — prefill/decode pools each need to fit.
    if cfg.disaggregate:
        p_tp = cfg.prefill_tp if cfg.prefill_tp else cfg.tp
        d_tp = cfg.decode_tp if cfg.decode_tp else cfg.tp
        if p_tp not in legality.tp:
            return False, f"prefill_tp={p_tp} not in legal TP set {legality.tp}"
        if d_tp not in legality.tp:
            return False, f"decode_tp={d_tp} not in legal TP set {legality.tp}"
        if cfg.decode_replicas < 1:
            return False, "decode_replicas must be >= 1"
        ep = cfg.ep if getattr(arch, "is_moe", False) else 1
        prefill_gpus = p_tp * cfg.pp * ep
        decode_gpus = d_tp * cfg.pp * ep * cfg.decode_replicas
        if prefill_gpus + decode_gpus > world:
            return False, (
                f"disaggregated pools need {prefill_gpus}+{decode_gpus} GPUs "
                f"but only {world} available"
            )
    return True, ""


# ---------------------------------------------------------------------------
# Seed plan
# ---------------------------------------------------------------------------

@dataclass
class InferenceSeedPlan:
    candidates: list[InferenceTrialConfig]
    rationale: str = ""


def _profile_from_opt(opt: OptimizationConfig) -> dict:
    """Pull the serving request profile from the optimization config."""
    inf = getattr(opt, "inference", None) or {}
    return {
        "input_len": int(inf.get("input_len", 1024)),
        "output_len": int(inf.get("output_len", 128)),
        "max_concurrency": inf.get("max_concurrency"),
        "prefix_cache_hit_rate": float(inf.get("prefix_cache_hit_rate", 0.0) or 0.0),
    }


def default_inference_trial(
    arch: ArchitectureRecord, cluster: TargetCluster, opt: OptimizationConfig
) -> "InferenceTrialConfig":
    """Profile-anchored baseline trial — the same defaults the seed sweep starts
    from (largest intra-budget TP, batch 1, bf16, request profile from opt).

    Used to fill in fields a (partial) LLM proposal leaves unspecified, so the
    agent can name just the knob it wants to change.
    """
    leg = derive_inference_legality(arch, cluster)
    profile = _profile_from_opt(opt)
    world = cluster.num_nodes * cluster.gpus_per_node
    base_tp = max((t for t in leg.tp if t <= world), default=1)
    return InferenceTrialConfig(
        tp=base_tp,
        pp=1,
        ep=1,
        cp=1,
        batch_size=1,
        input_len=profile["input_len"],
        output_len=profile["output_len"],
        max_concurrency=profile["max_concurrency"],
        weight_dtype="bf16",
        kv_cache_dtype="bf16",
    )


def inference_trial_from_dict(
    d: dict,
    arch: ArchitectureRecord,
    cluster: TargetCluster,
    opt: OptimizationConfig,
) -> "InferenceTrialConfig":
    """Overlay a (possibly partial) proposal dict onto the profile baseline."""
    base = default_inference_trial(arch, cluster, opt)
    for k in base.as_dict():
        if k in d and d[k] is not None:
            setattr(base, k, d[k])
    return base


def build_inference_seed_plan(
    arch: ArchitectureRecord,
    cluster: TargetCluster,
    opt: OptimizationConfig,
    *,
    max_candidates: int = 16,
) -> InferenceSeedPlan:
    """Deterministic serving-config sweep, ordered by expected impact.

    Order: TP (latency) → batching/concurrency (throughput) → KV quant
    (capacity) → weight quant → combined → chunked prefill → speculative →
    EP (MoE).
    """
    leg = derive_inference_legality(arch, cluster)
    profile = _profile_from_opt(opt)
    in_len = profile["input_len"]
    out_len = profile["output_len"]
    world = cluster.num_nodes * cluster.gpus_per_node
    is_moe = bool(getattr(arch, "is_moe", False))

    # Baseline TP: largest legal TP that stays intra-node (good for latency)
    # while leaving room for the replica to fit (tp ≤ world).  EP defaults to
    # 1 for the TP/batch/dtype sweeps so those stay feasible on the cluster;
    # a dedicated EP sweep explores expert parallelism for MoE.
    base_tp = max(t for t in leg.tp if t <= world)

    # A prompt admitted to the engine in one step sizes the activation working
    # set, and at agentic context lengths that term alone is larger than the
    # device: 130k tokens unchunked costs ~575 GB, so every trial is rejected on
    # memory before its throughput is ever read. No server runs that way, so the
    # seeds carry a scheduler budget rather than treating it as a later knob.
    base_token_budget = 8192 if in_len >= 16384 else 0
    # What the scheduler actually has to compute per prompt, which under prefix
    # reuse is a small tail of it: at 92% reuse a 130k prompt is a ~10k prefill.
    # A budget that covers the tail retires a prompt in one step; a smaller one
    # splits it across steps that then land in decode and inflate TPOT.
    effective_prompt = max(1, int(in_len * (1.0 - profile["prefix_cache_hit_rate"])))
    covering_budget = 1 << max(11, (effective_prompt - 1).bit_length())
    # Dense attention over a 130k prompt is quadratic and costs this model a
    # thirteen-minute first token, so a checkpoint that declares an indexer is
    # seeded with it rather than being asked to rediscover its own architecture.
    base_sparse_topk = int(getattr(arch, "index_topk", 0) or 0)

    def mk(**kw) -> InferenceTrialConfig:
        base = dict(
            tp=base_tp, pp=1, ep=1, cp=1,
            batch_size=1, input_len=in_len, output_len=out_len,
            max_concurrency=profile["max_concurrency"],
            prefix_cache_hit_rate=profile["prefix_cache_hit_rate"],
            weight_dtype="bf16", kv_cache_dtype="bf16",
            chunked_prefill_size=0,
            max_num_batched_tokens=base_token_budget,
            sparse_attention_topk=base_sparse_topk,
            speculative_num_tokens=0, speculative_acceptance_rate=0.0,
        )
        base.update(kw)
        return InferenceTrialConfig(**base)

    seen: set[str] = set()
    cands: list[InferenceTrialConfig] = []

    def add(c: InferenceTrialConfig):
        sig = c.signature()
        if sig in seen:
            return
        ok, _ = validate_inference(c, arch, cluster, leg)
        if not ok:
            return
        seen.add(sig)
        cands.append(c)

    # 1) baseline
    add(mk())
    # 2) TP sweep (intra-node latency tradeoff)
    for tp in leg.tp:
        add(mk(tp=tp))
    # 2b) attention DP — for a model whose KV is one latent every head reads,
    #     this is the difference between storing the cache once per replica and
    #     once per rank, so it is seeded high rather than left to the agent to
    #     discover. Paired with a batch that can spend the freed capacity: at
    #     batch 1 there is nothing to hold and the axis only costs GEMM shape.
    for dp in [d for d in leg.attention_dp if d > 1 and base_tp % d == 0]:
        add(mk(attention_dp=dp, batch_size=16))
        add(mk(attention_dp=dp, batch_size=64, kv_cache_dtype="fp8"))
    # 2c) 4-bit weights. Not one point on a precision sweep for these models but
    #     the only precision they fit in, so it is seeded with the batch and the
    #     attention-DP degree that a fitting configuration would want.
    if "fp4" in leg.weight_dtype:
        add(mk(batch_size=16, weight_dtype="fp4",
               linear_weight_dtype="mxfp4", kv_cache_dtype="fp8"))
        for dp in [d for d in leg.attention_dp if d > 1 and base_tp % d == 0]:
            add(mk(batch_size=32, weight_dtype="fp4", linear_weight_dtype="mxfp4",
                   kv_cache_dtype="fp8", attention_dp=dp))
    # 2c) 4-bit weights. Not one point on a precision sweep for these models but
    #     the only precision they fit in, so it is seeded with the batch and the
    #     attention-DP degree that a fitting configuration would want.
    if "fp4" in leg.weight_dtype:
        add(mk(batch_size=16, weight_dtype="fp4",
               linear_weight_dtype="mxfp4", kv_cache_dtype="fp8"))
        for dp in [d for d in leg.attention_dp if d > 1 and base_tp % d == 0]:
            add(mk(batch_size=32, weight_dtype="fp4", linear_weight_dtype="mxfp4",
                   kv_cache_dtype="fp8", attention_dp=dp))
    # 2d) The high-throughput region. The deterministic plan used to stop at
    #     batch 64 with an unchunked prompt, which at agentic context is not a
    #     configuration that runs: batch 64 unchunked costs a ~50 s first token,
    #     so the whole large-batch region read as infeasible and only the agent
    #     stage ever reached it. Chunking the prompt is what makes these legal.
    #
    #     Concurrency is seeded at tier boundaries rather than at round numbers.
    #     Attention DP makes TTFT a sawtooth in concurrency with period equal to
    #     the DP degree -- the tier is ceil(concurrency / dp) sequences per rank,
    #     TTFT jumps at each boundary and then falls across the tier -- so the
    #     best point always sits at the top of a tier, and a plan that samples
    #     round numbers lands mid-tier and reports a worse configuration than
    #     exists.
    #
    #     Both chunked and unchunked are seeded, because which one wins depends
    #     on how much of the prompt is actually new. Chunking is what makes a
    #     cold 130k prompt survivable, but under heavy prefix reuse only a small
    #     tail is ever computed -- at 92% reuse a 130k prompt is a ~10k prefill
    #     -- and then chunking it just buys more steps for no benefit.
    if "fp4" in leg.weight_dtype:
        for dp in [d for d in leg.attention_dp if d > 1 and base_tp % d == 0]:
            for bs in [b for b in leg.batch_size if b in (64, 128, 256)]:
                for mc in _tier_tops(bs, dp, n=9):
                    add(mk(batch_size=bs, attention_dp=dp, max_concurrency=mc,
                           weight_dtype="fp4", linear_weight_dtype="mxfp4",
                           kv_cache_dtype="fp8", chunked_prefill_size=2048,
                           max_num_batched_tokens=2048))
                    add(mk(batch_size=bs, attention_dp=dp, max_concurrency=mc,
                           weight_dtype="fp4", linear_weight_dtype="mxfp4",
                           kv_cache_dtype="fp8"))
                    if covering_budget != base_token_budget:
                        add(mk(batch_size=bs, attention_dp=dp,
                               max_concurrency=mc, weight_dtype="fp4",
                               linear_weight_dtype="mxfp4",
                               kv_cache_dtype="fp8",
                               max_num_batched_tokens=covering_budget))
        # Elementwise fusion and the TP-collective optimizations are cheap and
        # compound with the above; kept as a separate point so their effect is
        # attributable rather than baked into every large-batch trial.
        best_dp = max([d for d in leg.attention_dp if base_tp % d == 0] or [1])
        for bs in [b for b in leg.batch_size if b in (64, 128)]:
            add(mk(batch_size=bs, attention_dp=best_dp,
                   max_concurrency=(_tier_tops(bs, best_dp) or [None])[0],
                   weight_dtype="fp4", linear_weight_dtype="mxfp4",
                   kv_cache_dtype="fp8", chunked_prefill_size=2048,
                   max_num_batched_tokens=2048, fused_kernels=True,
                   quick_reduce=(base_tp > 1),
                   fuse_rmsnorm_allreduce=(base_tp > 1),
                   attention_backend="aiter"))
    # 3) batching / concurrency (throughput)
    for bs in [b for b in leg.batch_size if b in (4, 16, 64)]:
        add(mk(batch_size=bs))
    # 4) KV-cache quantization (capacity + bandwidth)
    add(mk(kv_cache_dtype="fp8"))
    add(mk(batch_size=16, kv_cache_dtype="fp8"))
    # 5) weight quantization (compute + memory)
    add(mk(weight_dtype="fp8"))
    # 6) combined best-guess throughput config
    add(mk(batch_size=32, weight_dtype="fp8", kv_cache_dtype="fp8"))
    # 7) chunked prefill (only meaningful for long prompts)
    if in_len >= 2048:
        add(mk(batch_size=16, chunked_prefill_size=1024))
    # 8) speculative decoding (latency)
    add(mk(speculative_num_tokens=4, speculative_acceptance_rate=0.7,
           speculative_draft_cost_factor=0.2))
    # 8b) CUDA-graph capture (per-step launch overhead / mixed-step penalty).
    add(mk(batch_size=16, cudagraph_mode="full"))
    add(mk(batch_size=16, cudagraph_mode="piecewise"))
    # 8c) KV-cache memory fraction (usable HBM → max concurrency).
    add(mk(batch_size=16, kv_cache_memory_fraction=0.9))
    # 8d) Paged-KV block size (fragmentation → KV bytes / max concurrency).
    add(mk(batch_size=16, kv_block_size=16))
    # 8e) Scheduler per-step token budget (caps prefill+decode tokens/step).
    add(mk(batch_size=16, max_num_batched_tokens=8192))
    add(mk(batch_size=16, chunked_prefill_size=1024, max_num_batched_tokens=8192))
    # 9) MoE EP sweep — pick the largest TP that still fits with this EP.
    if is_moe:
        for ep in [e for e in leg.ep if e in (1, 2, 4, 8)]:
            feasible_tp = [t for t in leg.tp if t * ep <= world]
            tp_for_ep = max(feasible_tp) if feasible_tp else 1
            add(mk(tp=tp_for_ep, ep=ep, batch_size=16))

    # 9a2) Expert parallelism and attention DP together. Neither substitutes for
    #      the other -- EP shards what each rank computes, attention DP shards
    #      which requests it holds -- and this pairing is what the measured MLA
    #      fleets run, so it is worth a seed rather than only a crossing the
    #      agent might find.
    if is_moe:
        for ep in [e for e in leg.ep if e in (1, 2, 4, 8)]:
            feasible_tp = [t for t in leg.tp if t * ep <= world]
            tp_for_ep = max(feasible_tp) if feasible_tp else 1
            if tp_for_ep > 1 and tp_for_ep in leg.attention_dp:
                add(mk(tp=tp_for_ep, ep=ep, batch_size=16,
                       attention_dp=tp_for_ep, kv_cache_dtype="fp8"))

    # 9b) MoE DeepEP — overlap the EP All-to-All behind expert compute. Only
    #     meaningful with EP>1, so pair it with the largest feasible EP.
    if is_moe:
        ep_for_deepep = max([e for e in leg.ep if e in (2, 4, 8) and e <= world] or [1])
        if ep_for_deepep > 1:
            feasible_tp = [t for t in leg.tp if t * ep_for_deepep <= world]
            tp_deepep = max(feasible_tp) if feasible_tp else 1
            add(mk(tp=tp_deepep, ep=ep_for_deepep, batch_size=16, use_turbo_deepep=True))

    # 9c) MoE expert-routing imbalance (+ redundant-expert mitigation). Only
    #     meaningful with EP>1, so pair with the largest feasible EP.
    if is_moe:
        ep_for_imb = max([e for e in leg.ep if e in (2, 4, 8) and e <= world] or [1])
        if ep_for_imb > 1:
            feasible_tp = [t for t in leg.tp if t * ep_for_imb <= world]
            tp_imb = max(feasible_tp) if feasible_tp else 1
            add(mk(tp=tp_imb, ep=ep_for_imb, batch_size=16, ep_load_balance=1.3))
            add(
                mk(
                    tp=tp_imb,
                    ep=ep_for_imb,
                    batch_size=16,
                    ep_load_balance=1.3,
                    redundant_experts=8,
                )
            )

    # 10) Feature B — custom collective ops. Force alternate algorithms for the
    #     dominant collective (TP AllReduce when TP>1, EP AllToAll for MoE).
    if base_tp > 1:
        add(mk(batch_size=16, tp_allreduce_algo="one_shot"))
        add(mk(batch_size=16, tp_allreduce_algo="hierarchical"))
    if is_moe:
        ep_for_a2a = max([e for e in leg.ep if e in (2, 4, 8) and e <= world] or [1])
        feasible_tp = [t for t in leg.tp if t * ep_for_a2a <= world]
        tp_a2a = max(feasible_tp) if feasible_tp else 1
        if ep_for_a2a > 1:
            add(mk(tp=tp_a2a, ep=ep_for_a2a, batch_size=16, ep_a2a_algo="hierarchical"))

    # 11) Feature A — prefill/decode disaggregation. Split the cluster into a
    #     latency-tuned prefill pool (higher TP) and a throughput-tuned decode
    #     pool (lower TP, more replicas), keeping the total within ``world``.
    if len(leg.tp) > 1:
        hi_tp = max(leg.tp)
        lo_tp = min(t for t in leg.tp if t > 0)
        ep = 1
        # Number of decode replicas that fit alongside one prefill pool.
        remaining = world - hi_tp
        dec_replicas = max(1, remaining // max(1, lo_tp)) if remaining > 0 else 1
        add(
            mk(
                tp=lo_tp,
                batch_size=16,
                disaggregate=True,
                prefill_tp=hi_tp,
                decode_tp=lo_tp,
                decode_replicas=dec_replicas,
            )
        )
        # Same split, naming the KV-transfer engine (NIXL link preset).
        add(
            mk(
                tp=lo_tp,
                batch_size=16,
                disaggregate=True,
                prefill_tp=hi_tp,
                decode_tp=lo_tp,
                decode_replicas=dec_replicas,
                transfer_backend="nixl",
            )
        )

    # 12) Kernel backend (ROCm attention library) — shape-dependent best pick.
    add(mk(batch_size=16, attention_backend="aiter"))
    add(mk(batch_size=16, attention_backend="ck"))
    # 13) Native sparse attention (NSA top-k) — pays off for long contexts.
    if in_len + out_len >= 4096:
        add(mk(batch_size=16, sparse_attention_topk=2048))
    # 14) Fused elementwise kernels — cut per-step launch overhead (pair with a
    #     capture mode that still has overhead to amortise).
    add(mk(batch_size=16, cudagraph_mode="piecewise", fused_kernels=True))
    # 15) Speculative decoding with an explicit draft-model cost charged.
    add(
        mk(
            batch_size=16,
            speculative_num_tokens=4,
            speculative_acceptance_rate=0.7,
            speculative_draft_cost_factor=0.2,
        )
    )
    # 16) MoE expert compute precision (mxfp4 / fp8 expert grouped-GEMM).
    if is_moe:
        ep_for_dtype = max([e for e in leg.ep if e in (2, 4, 8) and e <= world] or [1])
        feasible_tp = [t for t in leg.tp if t * ep_for_dtype <= world]
        tp_dtype = max(feasible_tp) if feasible_tp else 1
        add(mk(tp=tp_dtype, ep=ep_for_dtype, batch_size=16, moe_expert_dtype="fp8"))
        add(mk(tp=tp_dtype, ep=ep_for_dtype, batch_size=16, moe_expert_dtype="mxfp4"))
    # 17) Custom collective ops (TP>1): quick-reduce + fused RMSNorm+AllReduce.
    if base_tp > 1:
        add(mk(batch_size=16, quick_reduce=True))
        add(mk(batch_size=16, fuse_rmsnorm_allreduce=True))
    # 18) Offered-load probe — a Poisson arrival rate to expose the queueing knee.
    add(mk(batch_size=16, request_rate=8.0, arrival_model="poisson"))

    cands = cands[:max_candidates]
    return InferenceSeedPlan(
        candidates=cands,
        rationale=(
            f"inference seed sweep: TP∈{leg.tp}, batch∈{{1,4,16,32,64}}, "
            f"kv_dtype∈{leg.kv_cache_dtype}, weight_dtype∈{leg.weight_dtype}, "
            f"chunked-prefill, speculative, EP∈{leg.ep} "
            f"(profile: in={in_len}, out={out_len})"
        ),
    )


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------

# Lower-is-better metrics.
_MINIMIZE = {
    "ttft_ms", "itl_ms", "request_latency_ms", "tpot_ms", "latency_ms",
    # Capacity and step-quality objectives: less is better.
    "memory_per_gpu_gb", "kv_cache_gb", "decode_step_ms_pure",
    "mixed_step_fraction_pct", "tpot_pollution_pct",
    "decode_step_ms_mixed", "weights_gb", "activation_gb",
    "prefill_comm_ms", "decode_comm_ms", "replica_gpus",
}

# Friendly aliases the user may put in the YAML `objective:` field.
_OBJECTIVE_ALIASES = {
    "min_ttft": "ttft_ms",
    "ttft": "ttft_ms",
    "min_latency": "request_latency_ms",
    "latency": "request_latency_ms",
    "min_itl": "itl_ms",
    "itl": "itl_ms",
    "tpot": "itl_ms",
    "max_throughput": "decode_throughput_tps_per_gpu",
    "throughput": "decode_throughput_tps_per_gpu",
    "tokens_per_s_per_gpu": "decode_throughput_tps_per_gpu",
    "max_concurrency": "max_concurrent_sequences",
    "sustainable_concurrency": "max_sustainable_concurrency",
    "max_sustainable_concurrency": "max_sustainable_concurrency",
    # Total tokens per GPU, prompt plus generation. This is what InferenceX
    # ranks on and it is not the same ordering as decode-only throughput: at a
    # 144:1 prompt-to-generation ratio the prompt decides the winner.
    "max_total_throughput": "total_throughput_tps_per_gpu",
    "total_throughput": "total_throughput_tps_per_gpu",
    "tput_per_gpu": "total_throughput_tps_per_gpu",
    "total_throughput_per_gpu": "total_throughput_tps_per_gpu",
    "max_total_throughput_fleet": "total_throughput_tps",
    # Generation side alone.
    "max_output_throughput": "decode_throughput_tps_per_gpu",
    "output_throughput": "decode_throughput_tps_per_gpu",
    "output_tput_per_gpu": "decode_throughput_tps_per_gpu",
    "max_output_throughput_fleet": "decode_throughput_tps",
    # Prompt side alone -- the metric a prefill-heavy agentic fleet lives on.
    "max_input_throughput": "prefill_throughput_tps_per_gpu",
    "input_throughput": "prefill_throughput_tps_per_gpu",
    "input_tput_per_gpu": "prefill_throughput_tps_per_gpu",
    "max_input_throughput_fleet": "prefill_throughput_tps",
    # What a single user feels, as opposed to what the fleet delivers.
    "max_interactivity": "interactivity_tok_s_per_user",
    "interactivity": "interactivity_tok_s_per_user",
    "intvty": "interactivity_tok_s_per_user",
    "max_per_user_throughput": "per_request_decode_tps",
    "per_user_throughput": "per_request_decode_tps",
    # Latency, by the name each audience uses for it.
    "min_e2el": "request_latency_ms",
    "e2el": "request_latency_ms",
    "min_tpot": "itl_ms",
    "min_decode_step": "decode_step_ms_pure",
    # Capacity and efficiency.
    "min_memory": "memory_per_gpu_gb",
    "min_kv": "kv_cache_gb",
    "min_gpus": "replica_gpus",
    "min_weights": "weights_gb",
    "min_activation": "activation_gb",
    "min_prefill_comm": "prefill_comm_ms",
    "min_decode_comm": "decode_comm_ms",
    # Scheduler quality: how much of the decode budget prefill chunks eat.
    "min_mixed_step_fraction": "mixed_step_fraction_pct",
    "min_tpot_pollution": "tpot_pollution_pct",
}

# The catalogue the study sweeps, mapped to the InferenceX metric each answers.
# Energy is deliberately absent: InferenceX reports avg_power_w and
# joules_per_{output,total}_token, and the projector models no power at all, so
# there is nothing to optimize against and pretending otherwise would invent it.
INFERENCEX_OBJECTIVES = {
    "total_throughput_tps_per_gpu":  "tput_per_gpu (headline ranking)",
    "total_throughput_tps":          "tput_per_gpu x GPUs (fleet total)",
    "decode_throughput_tps_per_gpu": "output_tput_per_gpu",
    "decode_throughput_tps":         "output tokens/s (fleet)",
    "prefill_throughput_tps_per_gpu":"input_tput_per_gpu",
    "prefill_throughput_tps":        "input tokens/s (fleet)",
    "interactivity_tok_s_per_user":  "mean_intvty",
    "per_request_decode_tps":        "per-user generation rate",
    "ttft_ms":                       "mean_ttft",
    "itl_ms":                        "mean_tpot / mean_itl",
    "request_latency_ms":            "mean_e2el",
    "decode_step_ms_pure":           "uncontended step time",
    "max_concurrent_sequences":      "kv_cache_pool_tokens (as sequences)",
    "max_sustainable_concurrency":   "concurrency the pool sustains",
    "memory_per_gpu_gb":             "HBM footprint",
    "kv_cache_gb":                   "KV footprint",
    "mixed_step_fraction_pct":       "prefill interference in decode",
    "tpot_pollution_pct":            "TPOT inflation from interference",
    "decode_step_ms_mixed":          "step time when a prefill chunk lands",
    "weights_gb":                    "resident weight footprint",
    "activation_gb":                 "activation working set",
    "prefill_comm_ms":               "collective time in prefill",
    "decode_comm_ms":                "collective time in decode",
    "replica_gpus":                  "GPUs a replica costs",
}

# Deliberately absent, because the serving projection reports none of them and
# an objective that scores None reads as a failed search rather than a missing
# model: energy (avg_power_w, joules_per_{output,total}_token) is not modelled
# at all, and MFU / TFLOP-per-second / iteration time are training-mode figures
# the serving path never emits.
UNSUPPORTED_OBJECTIVES = {
    "avg_power_w", "joules_per_output_token", "joules_per_total_token",
    "mfu", "tflops_per_s_per_gpu", "iteration_ms",
}

DEFAULT_INFERENCE_OBJECTIVE = "decode_throughput_tps_per_gpu"


def resolve_objective(objective: str | None) -> str:
    if not objective:
        return DEFAULT_INFERENCE_OBJECTIVE
    key = str(objective).strip()
    return _OBJECTIVE_ALIASES.get(key, key)


def objective_is_minimize(objective: str) -> bool:
    return resolve_objective(objective) in _MINIMIZE


def score_result(result: dict, objective: str) -> float | None:
    """Signed score where *higher is always better* (negate minimize metrics)."""
    obj = resolve_objective(objective)
    val = result.get(obj)
    if val is None:
        return None
    return -float(val) if obj in _MINIMIZE else float(val)


# ---------------------------------------------------------------------------
# Metric parsing (matches inference_projection launcher output)
# ---------------------------------------------------------------------------

_FLOAT = r"([\-+]?\d+(?:\.\d+)?)"

_RE_TTFT = re.compile(rf"TTFT[^:]*:\s*{_FLOAT}\s*ms", re.IGNORECASE)
_RE_ITL = re.compile(rf"ITL\s*/\s*TPOT[^:]*:\s*{_FLOAT}\s*ms", re.IGNORECASE)
_RE_REQ_LAT = re.compile(rf"End-to-end request latency:\s*{_FLOAT}\s*ms", re.IGNORECASE)
_RE_DEC_TPS = re.compile(rf"Aggregate decode throughput:\s*{_FLOAT}\s*tok/s", re.IGNORECASE)
_RE_DEC_TPS_GPU = re.compile(rf"Decode throughput / GPU:\s*{_FLOAT}", re.IGNORECASE)
_RE_PREFILL_TPS = re.compile(rf"Prefill throughput:\s*{_FLOAT}\s*tok/s", re.IGNORECASE)
_RE_TOTAL_MEM = re.compile(rf"Projected Total Memory:\s*{_FLOAT}\s*GB", re.IGNORECASE)
_RE_KV = re.compile(rf"KV cache[^:]*:\s*{_FLOAT}\s*GB", re.IGNORECASE)
_RE_MAXCONC = re.compile(r"Max concurrent sequences:\s*(\d+)", re.IGNORECASE)
_RE_SUSTAINABLE = re.compile(r"Max sustainable concurrency:\s*(\d+)", re.IGNORECASE)
_RE_CONC_USED = re.compile(r"Concurrency used:\s*(\d+)", re.IGNORECASE)
# InferenceX ranks on total tokens per GPU -- prompt plus generation -- which at
# agentic context is dominated by the prompt and is a different ordering from
# decode-only throughput. It was the one headline metric never parsed.
_RE_TOTAL_TPS_GPU = re.compile(rf"Total throughput / GPU:\s*{_FLOAT}", re.IGNORECASE)
_RE_REPLICA_GPUS = re.compile(r"Replica GPUs[^:]*:\s*(\d+)", re.IGNORECASE)
_RE_INTERACTIVITY = re.compile(rf"Interactivity \(per user\):\s*{_FLOAT}", re.IGNORECASE)
_RE_PER_REQ_TPS = re.compile(rf"Per-request decode throughput:\s*{_FLOAT}", re.IGNORECASE)
_RE_STEP_PURE = re.compile(rf"Decode step latency \(pure\):\s*{_FLOAT}\s*ms", re.IGNORECASE)
_RE_MIXED_FRAC = re.compile(rf"Mixed-step fraction:\s*{_FLOAT}\s*%", re.IGNORECASE)
_RE_POLLUTION = re.compile(rf"TPOT pollution:\s*{_FLOAT}\s*%", re.IGNORECASE)
_RE_STEP_MIXED = re.compile(rf"Decode step latency \(pure\):[^|]*\|\s*mixed:\s*{_FLOAT}\s*ms", re.IGNORECASE)
_RE_WEIGHTS = re.compile(rf"Weights \([^)]*\):\s*{_FLOAT}\s*GB", re.IGNORECASE)
_RE_ACTIVATION = re.compile(rf"Activation working set:\s*{_FLOAT}\s*GB", re.IGNORECASE)
_RE_PREFILL_COMM = re.compile(rf"prefill:\s*TP-AR.*?total\s+{_FLOAT}", re.IGNORECASE)
_RE_DECODE_COMM = re.compile(rf"decode:\s*TP-AR.*?total\s+{_FLOAT}", re.IGNORECASE)


def _f(m) -> float | None:
    return float(m.group(1)) if m else None


def parse_inference_metrics(stdout: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if (m := _RE_TTFT.search(stdout)):
        out["ttft_ms"] = _f(m)
    if (m := _RE_ITL.search(stdout)):
        out["itl_ms"] = _f(m)
    if (m := _RE_REQ_LAT.search(stdout)):
        out["request_latency_ms"] = _f(m)
    if (m := _RE_DEC_TPS.search(stdout)):
        out["decode_throughput_tps"] = _f(m)
    if (m := _RE_DEC_TPS_GPU.search(stdout)):
        out["decode_throughput_tps_per_gpu"] = _f(m)
    if (m := _RE_PREFILL_TPS.search(stdout)):
        out["prefill_throughput_tps"] = _f(m)
    if (m := _RE_TOTAL_MEM.search(stdout)):
        out["memory_per_gpu_gb"] = _f(m)
    if (m := _RE_KV.search(stdout)):
        out["kv_cache_gb"] = _f(m)
    if (m := _RE_MAXCONC.search(stdout)):
        out["max_concurrent_sequences"] = int(m.group(1))
    if (m := _RE_SUSTAINABLE.search(stdout)):
        out["max_sustainable_concurrency"] = int(m.group(1))
    if (m := _RE_CONC_USED.search(stdout)):
        out["concurrency_used"] = int(m.group(1))
    if (m := _RE_TOTAL_TPS_GPU.search(stdout)):
        out["total_throughput_tps_per_gpu"] = _f(m)
    if (m := _RE_REPLICA_GPUS.search(stdout)):
        out["replica_gpus"] = int(m.group(1))
    if (m := _RE_INTERACTIVITY.search(stdout)):
        out["interactivity_tok_s_per_user"] = _f(m)
    if (m := _RE_PER_REQ_TPS.search(stdout)):
        out["per_request_decode_tps"] = _f(m)
    if (m := _RE_STEP_PURE.search(stdout)):
        out["decode_step_ms_pure"] = _f(m)
    if (m := _RE_MIXED_FRAC.search(stdout)):
        out["mixed_step_fraction_pct"] = _f(m)
    if (m := _RE_POLLUTION.search(stdout)):
        out["tpot_pollution_pct"] = _f(m)
    if (m := _RE_STEP_MIXED.search(stdout)):
        out["decode_step_ms_mixed"] = _f(m)
    if (m := _RE_WEIGHTS.search(stdout)):
        out["weights_gb"] = _f(m)
    if (m := _RE_ACTIVATION.search(stdout)):
        out["activation_gb"] = _f(m)
    if (m := _RE_PREFILL_COMM.search(stdout)):
        out["prefill_comm_ms"] = _f(m)
    if (m := _RE_DECODE_COMM.search(stdout)):
        out["decode_comm_ms"] = _f(m)
    # Per-GPU and fleet forms the projector prints only one side of. Ranking on
    # a fleet total rewards spending more GPUs, so both are kept and named.
    gpus = out.get("replica_gpus") or 0
    if gpus:
        if out.get("prefill_throughput_tps") is not None:
            out["prefill_throughput_tps_per_gpu"] = out["prefill_throughput_tps"] / gpus
        if out.get("total_throughput_tps_per_gpu") is not None:
            out["total_throughput_tps"] = out["total_throughput_tps_per_gpu"] * gpus
    return out
