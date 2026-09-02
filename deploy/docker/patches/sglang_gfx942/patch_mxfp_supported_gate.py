#!/usr/bin/env python3
"""Let MXFP4 checkpoints load on gfx942: widen `mxfp_supported()` to CDNA3.

THE BLOCK
---------
`sglang/srt/utils/common.py` decides MX support by string-matching the arch:

    def mxfp_supported():
        if torch.version.hip:
            gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
            return any(gfx in gcn_arch for gfx in ["gfx95"])
        return False

and `layers/quantization/mxfp4.py :: Mxfp4Config.from_config` turns a False into
a hard stop rather than a fallback:

    if _is_hip:
        if mxfp_supported():
            return cls(...)
        else:
            platform = torch.cuda.get_device_properties(0).gcnArchName
            raise ValueError(f"Current platform {platform} not support mxfp4 computation")

openai/gpt-oss-120b is post-trained natively in MXFP4 and ships no FP8
alternative, so on MI300X this raises at load and the model never serves.

WHY WIDENING IT IS THE RIGHT CHANGE, AND WHY IT IS NARROW
---------------------------------------------------------
The gate is a capability *claim*, not a kernel selector. Upstream discussion
#13611 concludes the check "is just too strict" and that the MXFP4/FP8 path is
functional on gfx94x; PR #13929 ("a simple toggle", +1 -1) would have flipped it
but was closed as stale in Aug 2026 without merging or accuracy numbers. So the
patch is justified by the correctness run on the box, not by the PR: gpt-oss-120b
serves through both the mixed and the disagg tier on MI300X with every
correctness probe passing — see manual/wip/gfx942-e2e.md.

Critically, this patches `mxfp_supported()` ONLY. The neighbouring
`is_gfx95_supported()` looks identical and must NOT be touched: `mxfp4.py:149`
uses it as `_is_shuffle_moe_mxfp4`, which pre-shuffles MoE weights into a
gfx950 CDNA4 layout and records `is_shuffled` on the parameter for the kernel to
read. Widening that one would hand gfx942 weights in a layout its kernels do not
expect — wrong numerics rather than a clean error. Leaving it False keeps gfx942
on the unshuffled path, which is the correct one.

UPSTREAM STATUS (2026-08-17)
  discussion #13611  "MXFP4 on MI300X" — maintainers state the gate is too
                     strict and the compute path already works on gfx94x.
  PR #13929          the +1 -1 toggle. CLOSED AS STALE, not rejected on merit.
  own PR             none filed. Should be, with the accuracy evidence this run
                     produces attached — that is exactly what #13929 lacked.

Applied only by Dockerfile.sglang.gfx942 (its own patch dir), so the gfx950
image is untouched. Idempotent and self-locating.
"""

import importlib.util
import os
import sys

MARKER = "INFERA_GFX942_MXFP_GATE"

OLD = """def mxfp_supported():
    \"\"\"
    Returns whether the current platform supports MX types.
    \"\"\"
    if torch.version.hip:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx95"])
    else:
        return False"""

NEW = f'''# {MARKER}: WHY gpt-oss-120b ships MXFP4-only weights and this gate raises on
# gfx942, though the compute path works there (sglang#13611). HOW add gfx94 to the
# claim. NOTE is_gfx95_supported() below stays gfx95-only — it selects a CDNA4
# weight shuffle, not a capability. See infera patch_mxfp_supported_gate.py.

# A real literal, not a comment: `strings *.pyc` must prove this reached the
# BYTECODE, and the compiler discards comments.
{MARKER} = "applied"


def mxfp_supported():
    """
    Returns whether the current platform supports MX types.
    """
    if torch.version.hip:
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return any(gfx in gcn_arch for gfx in ["gfx95", "gfx94"])
    else:
        return False'''


def find_common_py() -> str:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("cannot locate the sglang package")
    root = list(spec.submodule_search_locations)[0]
    path = os.path.join(root, "srt", "utils", "common.py")
    if not os.path.isfile(path):
        sys.exit(f"not found: {path}")
    return path


def main() -> int:
    path = find_common_py()
    src = open(path).read()

    if MARKER in src:
        print(f"[patch] already applied: {path}")
        return 0

    if OLD not in src:
        print("[patch] ERROR: mxfp_supported() not found in the expected shape.")
        print(f"        {path}")
        print("        Refusing to guess — a near-miss here silently disables MXFP4.")
        return 1

    open(path, "w").write(src.replace(OLD, NEW, 1))
    print(f"[patch] {MARKER}: mxfp_supported() now accepts gfx94x in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
