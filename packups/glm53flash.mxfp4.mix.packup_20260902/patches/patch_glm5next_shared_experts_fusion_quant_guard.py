#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""glm5_next -- refuse shared-expert fusion when the shared expert is excluded
from quantization but the routed experts are not.

WHAT: `Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason`
accepts a `quant_config` and never reads it. Its five branches cover the config
(`n_shared_experts`), the device (CUDA or gfx95 AITER), the SM level, the EP
size and the DeepEP backend -- and say nothing about quantization. On a
mixed-precision checkpoint whose routed experts are MXFP4 and whose shared
experts are BF16, the gate answers "fuse", the loader remaps
`mlp.shared_experts` into routed slot `n_routed_experts`, and a BF16 tensor is
copied into an MXFP4-packed buffer. This patch adds the guard its two sibling
model families already have.

WHY: without it GLM-5.3-Flash-MXFP4 (OneNexus/GLM-5.3-Flash-MXFP4, Quark, 288
routed experts + 1 shared) cannot load at all on 8xMI355X (gfx950). Instrumented
at the failing copy:

    layer_id=10  wname='model.layers.10.mlp.experts.w2_weight'  shard=w2
    eid=288      param=(289, 4096, 256)   loaded=(4096, 2048)
    qm=QuarkFusedMoEMethod

There are only 288 routed experts (0..287), so `eid=288` IS the shared expert,
and `param` dim0 = 289 is the routed buffer grown by one to hold it. Dim 1 tells
the rest: `moe_intermediate_size` 2048, TP4-sharded to 512, MXFP4-packed two
values per byte to 256 -- against a BF16 shared-expert `down_proj` of
`[4096, 2048]` that TP4-shards to 512. Surfaces as

    RuntimeError: The size of tensor a (256) must match the size of tensor b
    (512) at non-singleton dimension 1

from `_load_w2` -> `expert_data.copy_(loaded_weight)`, while loading shard 4 of
120 (tqdm reports "3/120 Completed"; layer 10's shared experts live in
`model-00004-of-00120.safetensors`).

The checkpoint states its intent correctly and always did: `config.json`'s
`quantization_config.exclude` lists `model.layers.N.mlp.shared_experts.{gate,up,
down}_proj` for every MoE layer. `QuarkConfig.can_fuse_shared_expert()`
(`layers/quantization/quark/quark.py:1001`) reads exactly that and returns
False. The right answer was computed and never asked for.

HOW: mirror `deepseek_v2.py:3069` verbatim -- open the gate with
`quant_blocks_shared_experts_fusion(quant_config)`
(`models/deepseek_common/utils.py:155`), which duck-types
`quant_config.can_fuse_shared_expert()` and is what `deepseek_v4.py:3289` uses
too. Two edits: add the symbol to the existing
`deepseek_common.utils` import block, and insert the check as the FIRST
statement of the gate.

  FIRST, not merely somewhere. deepseek_v2 places it ahead of its
  `enforce_shared_experts_fusion` early-return on purpose, commented "Need to
  disable if quant precision mismatch, even if --enforce-shared-experts-fusion
  is specified", and the shared test module pins that
  (`test_mixed_precision_quant_vetoes_even_when_enforced`). glm5_next has no
  enforce early-return today; putting the check first means it still holds if
  one is added later.

CONTEXT
  origin       PR #36607 commit `bd1cc98b` "[AMD] Enable GLM shared-expert
               fusion on gfx95" (andy.luo@amd.com, 2026-08-27). Its entire
               production change is nine lines swapping
               `if not _is_cuda` for `if not (_is_cuda or _use_aiter_gfx95)`.
               Before it, ROCm always returned a disable reason, so this bug was
               unreachable on AMD; that commit opened the door and added no
               quantization guard. Its 56 accompanying test lines
               (`TestGlm5NextGate`) cover backend / SM / EP / DeepEP and pass
               `quant_config=None` in every case, so the quant dimension is
               untested for this model. Sibling PR #37057 ("... on gfx942",
               CLOSED) extends the same gate and repeats the omission.
  upstream     UNFIXED at PR #36607's head. `refs/pull/36607/head` is exactly
               c821c425 with zero commits beyond it; the only downstream
               movement is `c767511e` (2026-09-01, on PR #36507), which reverts
               #36607 WHOLESALE -- 22 files, removing the gfx95 enablement
               rather than guarding it. So there is no upstream fix to take, and
               a rebase onto post-revert #36507 would drop ROCm support
               entirely. `quant_blocks_shared_experts_fusion` appears in
               upstream in three places only -- its definition plus
               deepseek_v2 and deepseek_v4 -- and never in glm5_next.
  same class   Issue #37268 is this bug on NVIDIA: GLM-5.3-NVFP4 on H100 logs
               "Shared experts fusion optimization enabled." then dies in
               `_load_w13` with 3072 vs 6144, same 2x ratio, same accepted
               workaround. PR #37325 fixes it in exactly this shape (add a gate
               branch, keep the enforce override, extend the same test module).
               PR #25261 does the same for GLM5 AutoRound INT4 -- note that one
               produced SILENT WRONG OUTPUT rather than a crash, because INT4
               shapes happen to line up. MXFP4's 2:1 packing is why we got a
               loud failure instead of a quiet one.
  runtime      `--disable-shared-experts-fusion` is the exact runtime
               equivalent: `layers/moe/utils.py:459` short-circuits on it before
               ever calling the gate. Verified on the vendor image
               (lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260822) + c821c425
               + bare launch_server + the unmodified model dir at TP4: server
               up, both memory pools present, 8 AITER mHC lines, `17*23` -> 391.
               The flag and this patch are interchangeable in effect; the patch
               additionally makes the engine right for anyone who forgets it.
  cost         PR #37057 measures fusion at +21.18% output tok/s (333.572 ->
               404.226) on 8xMI300X / TP8 / FP8. That is gfx942 / FP8, NOT our
               gfx950 / mixed-MXFP4 configuration, and is quoted here only so
               nobody assumes the guard is free. For THIS checkpoint it is free:
               fusion does not load at all, so the choice is not fast vs slow
               but runs vs does not run.
  not fixed    The MTP/NextN draft layer 45 also has BF16 routed experts, and
               `model.layers.45.mlp.experts` is absent from `exclude` (only
               deeper per-expert entries are present, and those can never match
               a FusedMoE prefix). Independent of fusion; only reachable with
               speculative decoding on. Out of scope here.

Self-locating and idempotent. All edits or none: an anchor that is missing or no
longer unique writes NOTHING and fails (exit 1), because a half-applied fix
crashes in the same place the unpatched tree does.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_TAG = "[glm5next-fusion-quant-guard]"

_REL = "models/glm5_next.py"

# The helper this patch calls must already exist, or the added import is an
# ImportError at model load rather than a clean build failure here.
_HELPER_REL = "models/deepseek_common/utils.py"
_HELPER_DEF = "def quant_blocks_shared_experts_fusion(\n"

# Proof that the gate edit specifically landed. Distinct from the bare symbol
# name, which the import edit alone would already satisfy.
_MARKER = "if quant_blocks_shared_experts_fusion(quant_config):"

# (anchor, anchor + our edit). Each anchor must occur exactly once; the
# replacement doubles as the already-applied marker.
_EDITS: list[tuple[str, str]] = [
    # Import the guard. Appended last in the existing block: isort sorts
    # underscore-prefixed names ahead of bare ones ('_' is 0x5F, 'q' is 0x71).
    (
        """from sglang.srt.models.deepseek_common.utils import (
    _device_sm,
    _is_cuda,
    _use_aiter_gfx95,
)
""",
        """from sglang.srt.models.deepseek_common.utils import (
    _device_sm,
    _is_cuda,
    _use_aiter_gfx95,
    quant_blocks_shared_experts_fusion,
)
""",
    ),
    # The guard itself, as the gate's first statement. Anchored on the def line
    # plus its two-line comment and the `text_config` binding -- the def line
    # alone is unique, but carrying the body makes a silent reorder upstream
    # fail the match instead of landing the check in a moved position.
    (
        """    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        # Kept in lockstep with the wrapper gate below: a divergence drops the
        # shared-expert weights and runs the fused slot uninitialized.
        text_config = getattr(hf_config, "text_config", hf_config)
""",
        """    def shared_experts_fusion_disable_reason(cls, hf_config, quant_config):
        # Kept in lockstep with the wrapper gate below: a divergence drops the
        # shared-expert weights and runs the fused slot uninitialized.
        #
        # Quantization first, mirroring deepseek_v2 / deepseek_v4: a precision
        # mismatch must veto fusion even when it is explicitly enforced, because
        # the fused slot is allocated with the ROUTED experts' packing. On a
        # Quark MXFP4 checkpoint whose shared experts are excluded (BF16), the
        # loader remaps mlp.shared_experts into routed slot n_routed_experts and
        # _load_w2 copies a BF16 (4096, 2048) tensor into an MXFP4-packed
        # (4096, 256) slot. See this patch's header.
        if quant_blocks_shared_experts_fusion(quant_config):
            return (
                "Quantization keeps shared experts at a higher precision than "
                "the routed experts, so they cannot be fused into the "
                "quantized routed-expert path."
            )
        text_config = getattr(hf_config, "text_config", hf_config)
""",
    ),
]


def _srt_dir() -> Path | None:
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        return None
    d = Path(spec.origin).parent / "srt"
    return d if d.is_dir() else None


def main() -> int:
    srt = _srt_dir()
    if srt is None:
        print(f"{_TAG} sglang not importable — skipping")
        return 0

    f = srt / _REL
    if not f.is_file():
        print(f"{_TAG} {f} is missing — sglang layout changed, re-anchor the patch")
        return 1

    helper = srt / _HELPER_REL
    if not helper.is_file() or _HELPER_DEF not in helper.read_text():
        print(
            f"{_TAG} quant_blocks_shared_experts_fusion not found in {_HELPER_REL} — "
            "the guard this patch imports does not exist here, refusing to write"
        )
        return 1

    src = out = f.read_text()
    if _MARKER in src:
        print(f"{_TAG} already present — skipping")
        return 0

    for old, new in _EDITS:
        if new in out:
            continue  # this edit is already in the tree
        found = out.count(old)
        if found != 1:
            where = "absent" if found == 0 else f"{found}x ambiguous"
            print(f"{_TAG} anchor {where} in {_REL}: {old.splitlines()[0]!r}")
            print(f"{_TAG} sglang drifted — re-cut the patch, nothing written")
            return 1
        out = out.replace(old, new, 1)

    if out == src:
        print(f"{_TAG} already present — skipping")
        return 0
    if _MARKER not in out:
        print(f"{_TAG} guard edit did not land — refusing to write a partial patch")
        return 1

    f.write_text(out)
    print(f"{_TAG} patched {f}")
    print(
        f"{_TAG} mixed-precision checkpoints now refuse shared-expert fusion "
        "instead of mis-loading the shared expert"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
