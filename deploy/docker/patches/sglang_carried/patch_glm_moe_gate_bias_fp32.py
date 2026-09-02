#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""GLM MoE gate: keep `e_score_correction_bias` in fp32 end to end.

WHAT: on ROCm with aiter, `MoEGate.__init__` allocates the MoE
`e_score_correction_bias` parameter as **bf16** whenever a quant_config is
present, and `biased_grouped_topk_gpu` casts it down again at the aiter call.
GLM's bias is a narrow band at a large offset, so bf16 cannot represent it: the
router ends up choosing among hundreds of experts using a bias with single-digit
distinct levels. This patch keeps the parameter fp32 for GLM MoE models and, at
the aiter boundary, promotes the gating logits to fp32 instead of demoting the
bias.

WHY: measured on this host, reading the checkpoints' own safetensors --

    checkpoint              tensor                       range          fp32  bf16
    GLM-5.3-MXFP4    L10 mlp.gate.e_score_correction_bias 6.817 - 7.063   238     8
    GLM-5.3          L10 (byte-identical values)          6.817 - 7.063   238     8
    GLM-5.3-Flash-MXFP4 L10 [288 experts]                 6.179 - 6.564   281    10
    GLM-5.3-Flash-MXFP4 L11 [288 experts]                 6.772 - 7.167   282    11

  bf16's ULP at 7.0 is 0.03125 and the whole spread is ~0.25, so 238 distinct
  fp32 biases collapse into 8 bf16 bins (Flash: 282 -> 11). `noaux_tc` selects
  experts by `sigmoid(logits) + correction_bias`, so this is a routing
  perturbation. Upstream (sgl-project/sglang#37133) measured **98.50 % of tokens
  select a different top-8 expert set** on GLM-5.2, and states plainly that
  accuracy benchmarks cannot resolve it (GSM8K 0.941 -> 0.947, GPQA-D 0.8333 ->
  0.8182, both inside the error bars). Those are UPSTREAM's numbers on GLM-5.2 /
  MI355X. We have measured the dtype collapse; we have NOT measured a quality
  delta, and nothing here should be read as claiming one.

  This is a correctness fix and it COSTS a little throughput: upstream reports a
  flat +4.5 us per gating call, <= 1.2 % of a decode step. Do not expect tok/s.

  The trigger is live in both our images, verified link by link:
    1. `SGLANG_USE_AITER=1` is baked into the image environment, so
       `_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip` is True.
    2. `"quark"` is in the downcast tuple and we launch `--quantization quark`.
    3. `DeepseekV2MoE` forwards the model-level quant_config into `MoEGate`
       unchanged, so `quant_config is not None` holds.
    4. On HIP the flashinfer and `_is_cuda` arms of `biased_grouped_topk_gpu`
       are skipped, so we land in the `elif _use_aiter:` arm that casts down.

HOW: two edits that MUST land together (see "BOTH OR NEITHER" below).

  A. `models/deepseek_v2.py` -- add `_moe_gate_bias_wants_fp32(config)` and use
     it to skip the bf16 downcast in `MoEGate.__init__`.
  B. `layers/moe/topk.py` -- in `biased_grouped_topk_gpu`'s aiter arm, when the
     bias arrives fp32, pass it through and promote `gating_output` to fp32;
     otherwise keep today's downcast byte-for-byte.

BOTH OR NEITHER -- why edit B is not optional polish. aiter's kernel launcher
is:

    VLLM_DISPATCH_FLOATING_TYPES_rmTorch(gating_output.dtype(), ..., [&] {
        ... reinterpret_cast<scalar_t*>(gating_output.data_ptr()),
            reinterpret_cast<scalar_t*>(correction_bias.data_ptr()), ...

  (`csrc/kernels/topk_softmax_kernels_group.cu`, read in our own image.) It
  dispatches on the GATING dtype only and then **reinterpret_casts the bias to
  that same scalar_t with no check of the bias tensor's own dtype**. Handing it
  an fp32 bias alongside a bf16 gating tensor is not an error and not a cast --
  it reads fp32 bytes as bf16 and returns silent garbage. So applying A without
  B is strictly WORSE than the bug it fixes. That is the concrete reason this
  script is all-or-nothing rather than two independent edits.

  The converse direction is safe: `AITER_DTYPE_fp32` is in the dispatch list, so
  promoting the gating tensor is a supported instantiation.

WHY THE GATE IS NOT UPSTREAM'S. #37133 gates on `any("GlmMoeDsa" in arch for
arch in config.architectures)`. That predicate is False for the Flash family and
would leave it broken:

  * `Glm5NextForConditionalGeneration.__init__` does `self.config =
    config.text_config` and passes the TEXT config down, so the object `MoEGate`
    receives is `text_config` -- and GLM-5.3-Flash-MXFP4's `text_config` has
    **`architectures: None`**. There is no arch string to match; widening the
    string list does not help.
  * Flash imports `DeepseekV2MoE as Glm5NextMoE`, i.e. the very same `MoEGate`,
    and has the same collapse (282 -> 11 above).

  So the predicate here is a union of three tests, cheapest first:

    1. `moe_router_dtype == "float32"` -- the model author's own declaration.
       Present on GLM-5.3, GLM-5.3-MXFP4, GLM-5.3-Flash, GLM-5.3-Flash-MXFP4 and
       GLM-5.2-FP8-fixed. sglang v0.5.18 ignores the field entirely (zero
       references under `python/sglang/srt/`), which is the actual root cause.
    2. `model_type` -- `glm_moe_dsa` (5.1/5.2/5.3 big) or `glm5_next*` (Flash,
       whose text config reports `glm5_next_text`). Needed because GLM-5.2-MXFP4
       and GLM-5.1-FP8 predate the `moe_router_dtype` field. `model_type`
       survives the NextN rewrite in `configs/model_config.py`, so draft heads
       are covered without a substring hack.
    3. upstream's arch-string test, kept so this stays a superset of #37133 and
       is a no-op once that lands.

  BLAST RADIUS, measured rather than assumed: of 38 checkpoints on this host,
  exactly 5 carry `moe_router_dtype: "float32"` and all 5 are GLM 5.2/5.3. Every
  DeepSeek-V3/V4/R1, Kimi-K3, Qwen2.5/3/3.5, MiniMax-M3, Hy3, MiMo, Llama and
  Mistral config has it unset and a non-GLM `model_type`, so all of them take
  the unchanged branch and stay byte-identical.

CONTEXT
  upstream       sgl-project/sglang#37133 "[GLM-5.2] Keep GlmMoeDsa MoE
                 e_score_correction_bias in fp32" (xiaobochen-amd), OPEN against
                 main, not merged. Its CI is red only at `pr-gate` /
                 `*-finish` aggregators -- the missing-`run-ci`-label failure --
                 so no AMD job has actually run, and the AMD runners are
                 `linux-mi300-*` (gfx942), not our gfx950. Treat it as unvalidated
                 on our hardware.
  bases          The `MoEGate` dtype block is textually IDENTICAL on v0.5.18 and
                 c821c425, so edit A has one anchor. `topk.py` drifted -- v0.5.18
                 has `correction_bias.to(dtype=gating_output.dtype)` inline where
                 c821c425 hoisted a `bias` local and a `scaling` local -- so edit
                 B carries one alternative per base and requires that EXACTLY ONE
                 of them match.
  why a script   Upstream's diff FAILS `git apply --check` on v0.5.18 (the
                 `topk.py` hunk) and its arch predicate is wrong for Flash. Both
                 bases are pinned and neither can move: c821c425 is a
                 merged-and-frozen ref whose upstream branch was reverted
                 wholesale on 2026-09-01, so there is no newer ref to bump to.
                 Anchoring on source text serves both bases from one file.
  NOT covered    `fused_topk()` has a second `aiter_biased_grouped_topk` call
                 with the same downcast. It is the ungrouped path and
                 `topk_method == "noaux_tc"` never reaches it, so it is left
                 alone deliberately -- same as upstream.
  precedent      `router_dtype` is an established transformers config field
                 (switch_transformers, nllb_moe) defaulting to `"float32"` and
                 honoured as `getattr(torch, config.router_dtype)`. GLM's
                 `moe_router_dtype` is the same idea under a namespaced name.

Self-locating and idempotent. All edits or none, across BOTH files: an anchor
that is missing, ambiguous, or matches in more than one variant writes NOTHING
and exits 1 -- because a half-applied fix here is not a degraded fix, it is a
silently mis-typed kernel argument.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_TAG = "[glm-moe-gate-bias-fp32]"

# --------------------------------------------------------------------------
# Edit A -- models/deepseek_v2.py
# --------------------------------------------------------------------------

_A_REL = "models/deepseek_v2.py"

# The helper. Inserted immediately above MoEGate, which is where the only caller
# is; `class MoEGate(nn.Module):` occurs exactly once on both bases.
_A_HELPER_ANCHOR = "class MoEGate(nn.Module):\n"

_A_HELPER = '''def _moe_gate_bias_wants_fp32(config) -> bool:
    """True when this model's MoE gate bias must stay fp32.

    GLM stores `e_score_correction_bias` as a narrow band at a large offset
    (~6.2-7.2), where bf16's ULP of 0.03125 collapses hundreds of distinct
    biases into single digits and reorders `noaux_tc` top-k routing.

    Three tests, cheapest first, because no single one covers every GLM
    checkpoint we serve:
      1. `moe_router_dtype` -- the model author's own declaration. Carried by
         GLM-5.3 / -MXFP4 / -Flash / -Flash-MXFP4 and GLM-5.2-FP8.
      2. `model_type` -- GLM-5.1/5.2 predate that field, and the Flash family
         hands MoEGate its `text_config`, whose `architectures` is None, so the
         arch test below cannot see it. `model_type` is always present and
         survives the NextN rewrite in configs/model_config.py.
      3. architectures -- upstream sgl-project/sglang#37133's predicate, kept so
         this stays a superset of it and becomes a no-op once it lands.
    """
    router_dtype = getattr(config, "moe_router_dtype", None)
    if isinstance(router_dtype, str) and router_dtype.lower() in ("float32", "fp32"):
        return True
    model_type = getattr(config, "model_type", None) or ""
    if model_type.startswith("glm_moe_dsa") or model_type.startswith("glm5_next"):
        return True
    return any("GlmMoeDsa" in arch for arch in (getattr(config, "architectures", None) or []))


'''

# The downcast itself. Identical text on v0.5.18 and c821c425.
_A_GATE_ANCHOR = """        if config.topk_method == "noaux_tc" and not is_hash_moe:
            correction_bias_dtype = torch.float32
            if quant_config is not None:
                if _use_aiter and quant_config.get_name() in (
                    "fp8",
                    "compressed_tensors",
                    "quark",
                ):
                    correction_bias_dtype = torch.bfloat16
"""

_A_GATE_PATCHED = """        if config.topk_method == "noaux_tc" and not is_hash_moe:
            correction_bias_dtype = torch.float32
            # GLM53_BIASFP32: GLM's bias sits in a ~0.25-wide band around +7,
            # where bf16 steps by 0.03125 -- 238 distinct biases become 8 (Flash:
            # 282 -> 11), reordering top-k. HF stores it fp32; keep it fp32.
            # The matching promote lives at the aiter boundary in
            # layers/moe/topk.py and is NOT optional: aiter reinterpret_casts the
            # bias to the GATING tensor's dtype without checking it.
            if quant_config is not None and not _moe_gate_bias_wants_fp32(config):
                if _use_aiter and quant_config.get_name() in (
                    "fp8",
                    "compressed_tensors",
                    "quark",
                ):
                    correction_bias_dtype = torch.bfloat16
"""

# --------------------------------------------------------------------------
# Edit B -- layers/moe/topk.py, biased_grouped_topk_gpu's aiter arm
# --------------------------------------------------------------------------

_B_REL = "layers/moe/topk.py"

_B_PREAMBLE = """        # GLM53_BIASFP32: do NOT re-downcast an fp32 bias here. aiter dispatches
        # on gating_output.dtype and then reinterpret_casts the bias pointer to
        # that same scalar_t with no check of its own dtype, so an fp32 bias
        # under a bf16 gating tensor is read as garbage rather than rejected.
        # Promote the gating logits instead. Gated on the BIAS dtype, so a bias
        # that already arrives bf16 is byte-identical to before.
        if correction_bias.dtype == torch.float32:
            _glm_gating = gating_output.to(torch.float32)
            _glm_bias = correction_bias
        else:
            _glm_gating = gating_output
            _glm_bias = {fallback}
"""

# One variant per base. EXACTLY ONE must match; two matches means the file is
# not what either base says it is.
_B_VARIANTS: list[tuple[str, str, str]] = [
    (
        "v0.5.18 (inline downcast)",
        """        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        aiter_biased_grouped_topk(
            gating_output,
            correction_bias.to(dtype=gating_output.dtype),
            topk_weights,
            topk_ids,
            num_expert_group,
""",
        """        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
"""
        + _B_PREAMBLE.format(fallback="correction_bias.to(dtype=gating_output.dtype)")
        + """        aiter_biased_grouped_topk(
            _glm_gating,
            _glm_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
""",
    ),
    (
        "c821c425 (hoisted `bias` local)",
        """        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
        aiter_biased_grouped_topk(
            gating_output,
            bias,
            topk_weights,
            topk_ids,
            num_expert_group,
""",
        """        topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
"""
        + _B_PREAMBLE.format(fallback="bias")
        + """        aiter_biased_grouped_topk(
            _glm_gating,
            _glm_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
""",
    ),
]

# Present iff each edit landed. Used for idempotency and for the build-time
# bytecode check, the same way patch 01 greps for `_p1v2_trim`.
_A_MARKER = "not _moe_gate_bias_wants_fp32(config)"
_B_MARKER = "_glm_gating = gating_output.to(torch.float32)"


def _srt_dir() -> Path | None:
    """Locate `sglang/srt`, preferring the interpreter's own sglang."""
    spec = importlib.util.find_spec("sglang")
    if spec and spec.origin:
        d = Path(spec.origin).parent / "srt"
        if d.is_dir():
            return d
    root = os.environ.get("SGLANG_DIR", "/sgl-workspace/sglang")
    d = Path(root) / "python" / "sglang" / "srt"
    return d if d.is_dir() else None


def _fail(msg: str) -> int:
    print(f"{_TAG} {msg}", file=sys.stderr)
    print(f"{_TAG} sglang drifted — re-anchor the patch, nothing written", file=sys.stderr)
    return 1


def _plan_a(path: Path) -> str | int:
    """Return the new text for deepseek_v2.py, or an exit code."""
    src = path.read_text()
    if _A_MARKER in src:
        return src

    n = src.count(_A_GATE_ANCHOR)
    if n != 1:
        return _fail(
            f"{_A_REL}: MoEGate correction-bias dtype block matched {n} times (want 1)"
        )
    n = src.count(_A_HELPER_ANCHOR)
    if n != 1:
        return _fail(f"{_A_REL}: 'class MoEGate(nn.Module):' matched {n} times (want 1)")

    out = src.replace(_A_HELPER_ANCHOR, _A_HELPER + _A_HELPER_ANCHOR, 1)
    return out.replace(_A_GATE_ANCHOR, _A_GATE_PATCHED, 1)


def _plan_b(path: Path) -> str | int:
    """Return the new text for topk.py, or an exit code."""
    src = path.read_text()
    if _B_MARKER in src:
        return src

    hits = [(name, old, new) for name, old, new in _B_VARIANTS if src.count(old) == 1]
    ambiguous = [name for name, old, _ in _B_VARIANTS if src.count(old) > 1]
    if ambiguous:
        return _fail(f"{_B_REL}: anchor is ambiguous for variant(s) {ambiguous}")
    if len(hits) != 1:
        names = [name for name, _, _ in _B_VARIANTS]
        return _fail(
            f"{_B_REL}: {len(hits)} of {len(names)} aiter-call variants matched "
            f"(want exactly 1); tried {names}"
        )

    name, old, new = hits[0]
    print(f"{_TAG} {_B_REL}: matched variant {name}")
    return src.replace(old, new, 1)


def main() -> int:
    srt = _srt_dir()
    if srt is None:
        print(f"{_TAG} sglang not importable — skipping")
        return 0

    paths = {_A_REL: srt / _A_REL, _B_REL: srt / _B_REL}
    for rel, p in paths.items():
        if not p.is_file():
            return _fail(f"{p} is missing — sglang layout changed")

    # Plan both files before writing either: edit A without edit B hands aiter a
    # mis-typed bias pointer, which is worse than the defect.
    planned = {}
    for rel, p in paths.items():
        result = _plan_a(p) if rel == _A_REL else _plan_b(p)
        if isinstance(result, int):
            return result
        planned[rel] = result

    changed = [rel for rel, text in planned.items() if text != paths[rel].read_text()]
    if not changed:
        print(f"{_TAG} already present — skipping")
        return 0

    for rel in changed:
        paths[rel].write_text(planned[rel])
        print(f"{_TAG} patched {paths[rel]}")
    print(f"{_TAG} GLM MoE gate bias now stays fp32 through the aiter router")
    return 0


if __name__ == "__main__":
    sys.exit(main())
