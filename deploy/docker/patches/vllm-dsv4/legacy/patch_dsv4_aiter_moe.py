###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""[DEPRECATED — NEW IMAGE NO-OP] DeepSeek-V4 MXFP4 MoE aiter-dispatch fix for
vLLM's ``rocm_aiter_moe.py`` (ROCm / gfx950 / MI355X).

WHAT: (1) set GateMode.INTERLEAVE only for GPT-OSS Swiglu, so DSv4 Silu keeps
  gate_mode="" -> aiter flydsl a4w4; (2) bit-cast uint8 w1/w2_scale to
  float8_e8m0fnu before fused_moe.
WHY: stock vLLM forces INTERLEAVE for every mxfp4_w4a16 expert -> DSv4 falls to a
  cktile kernel rejecting fp4 -> gibberish / "Unsupported scales/output dtype!".
DEPS: NO-OP on the dev748 base (upstream: SWIGLUOAI/activation_interleave present)
  -> NOT applied by any Dockerfile, kept in legacy/ for older bases (see README.md).
  VERSION: verified vllm 0.23.1rc1.dev301, aiter Duyi-Wang@1cef616f (2026-06-15).
"""

import os
import sys

try:
    import vllm
except Exception:
    print("dsv4-aiter-moe: vLLM not importable — skipping")
    sys.exit(0)

f = os.path.join(
    os.path.dirname(vllm.__file__),
    "model_executor/layers/fused_moe/experts/rocm_aiter_moe.py",
)
if not os.path.exists(f):
    print(f"dsv4-aiter-moe: {f} not found — skipping")
    sys.exit(0)

src = open(f).read()
MARKER = "# DSV4-AITER-MOE-PATCH"
if MARKER in src:
    print("dsv4-aiter-moe: already patched")
    sys.exit(0)

# UPSTREAMED-CHECK: aiter >=~0.1.16 already defaults gate_mode="" and gates
# INTERLEAVE on SWIGLUOAI/activation_interleave. Applying our fork-era rewrite
# there would FORCE the wrong afp4 path -> garbage; detect and no-op.
if "SWIGLUOAI" in src or "activation_interleave" in src:
    print(
        "dsv4-aiter-moe: gate_mode/scale fix already upstream in this aiter "
        "(SWIGLUOAI/activation_interleave present) — skipping (no-op)"
    )
    sys.exit(0)

# Anchor: the fused_moe call passing w1/w2_scale straight from quant_config
# (present in v0.23.0 and nightly). Insert our block before it and rewrite the
# two scale kwargs to the bit-cast locals.
ANCHOR = "        return rocm_aiter_ops.fused_moe(\n"
if src.count(ANCHOR) != 1:
    print(
        f"dsv4-aiter-moe: anchor 'return rocm_aiter_ops.fused_moe(' found "
        f"{src.count(ANCHOR)}x (expected 1) — layout drifted, skipping"
    )
    sys.exit(0)

INSERT = (
    "        # DSV4-AITER-MOE-PATCH: DSv4 (Silu) must NOT use GateMode.INTERLEAVE\n"
    "        # (GPT-OSS/Swiglu-only); it forces q_dtype_a=bf16, skips flydsl a4w4, and\n"
    "        # falls to a cktile kernel rejecting fp4 -> 'Unsupported scales/output dtype!'.\n"
    '        _dsv4_gate_mode = ""\n'
    '        if quant_config.use_mxfp4_w4a16 and "swiglu" in str(activation_method).lower():\n'
    "            try:\n"
    "                from aiter.ops.flydsl.moe_common import GateMode\n"
    "                _dsv4_gate_mode = GateMode.INTERLEAVE.value\n"
    "            except ImportError:\n"
    "                pass\n"
    "        import torch as _dsv4_torch\n"
    "        _dsv4_w1s = quant_config.w1_scale\n"
    "        _dsv4_w2s = quant_config.w2_scale\n"
    "        # MXFP4 E8M0 scales are uint8 in vLLM params; aiter wants float8_e8m0fnu\n"
    "        # (bit-cast, not numeric convert).\n"
    "        if _dsv4_w1s is not None and _dsv4_w1s.dtype == _dsv4_torch.uint8:\n"
    "            _dsv4_w1s = _dsv4_w1s.view(_dsv4_torch.float8_e8m0fnu)\n"
    "        if _dsv4_w2s is not None and _dsv4_w2s.dtype == _dsv4_torch.uint8:\n"
    "            _dsv4_w2s = _dsv4_w2s.view(_dsv4_torch.float8_e8m0fnu)\n"
)
src = src.replace(ANCHOR, INSERT + ANCHOR, 1)

# Rewrite the two scale kwargs to use the bit-cast locals.
for old, new in [
    ("            w1_scale=quant_config.w1_scale,\n", "            w1_scale=_dsv4_w1s,\n"),
    ("            w2_scale=quant_config.w2_scale,\n", "            w2_scale=_dsv4_w2s,\n"),
]:
    if src.count(old) != 1:
        print(f"dsv4-aiter-moe: scale kwarg {old.strip()!r} found {src.count(old)}x — skipping")
        sys.exit(1)
    src = src.replace(old, new, 1)

# If this build passes gate_mode to fused_moe (nightly with aiter#3123),
# route it through our conditional local. On v0.23.0 there is no such kwarg,
# so this is a no-op and DSv4 already avoids INTERLEAVE.
GATE_KWARGS = [
    "            gate_mode=gate_mode,\n",
    "            gate_mode=GateMode.INTERLEAVE.value,\n",
    "            gate_mode=GateMode.INTERLEAVE,\n",
]
_rewrote_gate = False
for gk in GATE_KWARGS:
    if src.count(gk) == 1:
        src = src.replace(gk, "            gate_mode=_dsv4_gate_mode,\n", 1)
        _rewrote_gate = True
        break

open(f, "w").write(src)
print(f"dsv4-aiter-moe: patched {f} (gate_mode kwarg rewritten: {_rewrote_gate})")
