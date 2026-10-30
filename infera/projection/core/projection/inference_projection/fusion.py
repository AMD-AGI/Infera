"""Per-pattern kernel fusion: what each fusion removes from a step, and its rank.

A fusion removes kernels and the intermediate tensors that ran between them.
Each pattern is priced as the removed kernels' fixed cost plus the bytes no
longer written and re-read,
at the bandwidth the projector's own element-wise terms stream at. Only traffic
the unfused step is already charged for is removed, so a saving never exceeds
what the baseline counts:

  * hidden-wide norm / residual passes  -- ``_estimate_layernorm_residual_time_ms``
  * activation casts                    -- ``QuantCastProfiler``
  * the MoE activation                  -- ``MoEMLPProfiler`` step 4

RoPE and routing traffic are not charged separately, so those patterns save
kernels only. Tokens are per rank (``batch * q_len / tp / cp``), the convention
the cast profiler uses; under EP == TP or EP == 1 the MoE terms reduce to the
same per-rank count the MoE profiler charges.

In benchmark mode the saving comes off the anchor's measured step, which is
right only for a fusion the anchor's engine does not already run -- which is
what offering the knob asserts.
"""

from __future__ import annotations

from dataclasses import dataclass

from infera.projection.core.projection.module_profilers.moe_mlp import _ACTIVATION_BW_FRACTION
from infera.projection.core.projection.module_profilers.quantization import _WRITE_BYTES

_BF16 = 2.0
_FALLBACK_HBM_BW_GBPS = 5300.0

# A fused kernel rarely reaches its roofline, so the saving is reported at three
# efficiencies of it; a fusion worth under 2% of the step is not worth proposing.
CONFIDENCE_BANDS = {"low": 0.75, "mid": 0.875, "high": 1.0}
MIN_SHARE = 0.02


@dataclass(frozen=True)
class FusionPattern:
    name: str
    replaces: str
    layers: str  # "all" | "moe"
    needs_quant: bool = False


PATTERNS: dict[str, FusionPattern] = {
    p.name: p
    for p in (
        FusionPattern(
            "add_rmsnorm",
            "residual add + following RMSNorm, twice per layer",
            "all",
        ),
        FusionPattern(
            "norm_quant",
            "RMSNorm + activation quant feeding the attention and MLP GEMMs",
            "all",
            needs_quant=True,
        ),
        FusionPattern(
            "act_quant",
            "SiLU*mul + quant feeding the down projection",
            "all",
            needs_quant=True,
        ),
        FusionPattern(
            "rope_kvstore",
            "RoPE into the KV-cache store",
            "all",
        ),
        FusionPattern(
            "moe_routing",
            "router top-k + permute into one routing kernel",
            "moe",
        ),
        FusionPattern(
            "moe_expert",
            "expert gate/up -> activation (-> quant) -> down in one kernel",
            "moe",
        ),
    )
}


def applicable(name: str, *, is_moe: bool, quantized: bool) -> bool:
    p = PATTERNS[name]
    return (p.layers != "moe" or is_moe) and (quantized or not p.needs_quant)


def validate(fusions) -> list[str]:
    names = [str(f).strip() for f in (fusions or ()) if str(f).strip()]
    bad = [f for f in names if f not in PATTERNS]
    if bad:
        raise ValueError(f"unknown fusion(s) {bad}; known: {sorted(PATTERNS)}")
    return names


@dataclass
class FusionSaving:
    name: str
    kernels: float
    bytes: float
    ms: float


def savings(
    model_config,
    mp_config,
    *,
    fusions,
    batch: int,
    q_len: int,
    n_dense: int,
    n_moe: int,
    quant_dtype: str | None,
    hbm_bandwidth_gbps: float | None,
    occupancy_us: float,
) -> list[FusionSaving]:
    """Per-step saving of each selected fusion, in selection order."""
    names = validate(fusions)
    if not names:
        return []
    tp = max(1, int(mp_config.tensor_model_parallel_size or 1))
    cp = max(1, int(getattr(mp_config, "context_model_parallel_size", 1) or 1))
    t = max(1, batch * q_len // tp // cp)
    h = int(model_config.hidden_size)
    f = int(model_config.ffn_hidden_size or 0)
    moe_i = int(getattr(model_config, "moe_ffn_hidden_size", 0) or 0) or f
    k = max(1, int(getattr(model_config, "moe_router_topk", 1) or 1))
    swiglu = bool(getattr(model_config, "swiglu", True))
    quant = quant_dtype is not None
    w = _WRITE_BYTES.get(str(quant_dtype).lower(), 1.0) if quant else 0.0
    bw = float(hbm_bandwidth_gbps or _FALLBACK_HBM_BW_GBPS) * _ACTIVATION_BW_FRACTION
    layers = n_dense + n_moe
    chosen = set(names)

    out = []
    for name in names:
        kern, byt = 0.0, 0.0
        if name == "add_rmsnorm":
            # 6 passes (add 3, norm 3) become 4 per instance.
            kern = 2 * layers
            byt = layers * 2 * 2 * t * h * _BF16
        elif name == "norm_quant" and quant:
            # The cast pass goes, and the norm writes packed rather than bf16.
            # A MoE layer's MLP-input cast ran over the routed copies.
            attn = t * h * (_BF16 + w) + t * h * (_BF16 - w)
            dense_mlp = attn
            moe_mlp = k * t * h * (_BF16 + w) + t * h * (_BF16 - w)
            kern = 2 * layers
            byt = layers * attn + n_dense * dense_mlp + n_moe * moe_mlp
        elif name == "act_quant" and quant:
            kern = n_dense + (0 if "moe_expert" in chosen else n_moe)
            byt = n_dense * t * f * (_BF16 + w)
            if "moe_expert" not in chosen:
                byt += n_moe * k * t * moe_i * (_BF16 + w)
        elif name == "rope_kvstore":
            kern = layers
        elif name == "moe_routing":
            kern = 2 * n_moe
        elif name == "moe_expert":
            act = (3 if swiglu else 2) * k * t * moe_i * _BF16
            cast = k * t * moe_i * (_BF16 + w) if quant else 0.0
            kern = n_moe * (2 if quant else 1)
            byt = n_moe * (act + cast)
        ms = kern * occupancy_us / 1000.0 + byt / (bw * 1e6)
        out.append(FusionSaving(name, kern, byt, ms))
    return out


def opportunities(step_ms: float, rows: list[FusionSaving]) -> list[dict]:
    """Ranked report: biggest saving first, with bands."""
    rep = []
    # The step times come out of numpy, whose scalars json cannot encode.
    step_ms = float(step_ms)
    for s in sorted(rows, key=lambda s: -s.ms):
        ms = float(s.ms)
        share = ms / step_ms if step_ms > 0 else 0.0
        p = PATTERNS[s.name]
        rep.append(
            {
                "fusion": s.name,
                "replaces": p.replaces,
                "kernels_removed_per_step": round(float(s.kernels), 1),
                "bytes_removed_per_step": int(s.bytes),
                "saving_ms": {b: round(ms * f, 4) for b, f in CONFIDENCE_BANDS.items()},
                "share_of_step": round(share, 4),
                "worth_it": bool(share >= MIN_SHARE),
            }
        )
    return rep
