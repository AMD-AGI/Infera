#!/usr/bin/env python3
"""Assert that the GLM-5.3-Flash (glm5_next) overlay actually landed in the image.

Run at the END of Dockerfile.sglang.glm53, after every later layer has had its
chance to clobber /sgl-workspace/sglang. Cheap, and it turns "a later layer
reinstalled sglang over the overlay" into a build failure instead of a
CrashLoopBackOff twenty minutes into a weight load.

DELIBERATELY NOT a ModelRegistry.get_supported_archs() check. registry.py
imports every model module with strict=False and logs+swallows the failures, so
on a GPU-less builder (aiter -> rocminfo) glm5_next would be reported "not
registered" for a reason that has nothing to do with this image -- a false
failure that would train people to ignore the check. Instead: verify the same
contract STATICALLY (module present, class defined, exported via EntryClass,
compiles), and only then attempt the real import, treating a GPU-absence error
as a pass.
"""

import ast
import importlib
import pathlib
import sys
import traceback

MODULE = "sglang.srt.models.glm5_next"
CLASS = "Glm5NextForConditionalGeneration"
SRC = pathlib.Path("/sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py")

# An import failure naming any of these is the builder lacking a GPU, not the
# overlay missing. Keep this list tight: anything not listed is a real failure,
# and a marker that is too broad turns this check into a rubber stamp -- which
# is worse than not having it, because the build then claims the overlay landed.
#
# Two that used to be here and are not:
#   "HIP error"  matches every runtime error HIP raises, a real broken kernel
#                included. The device-absence ones are spelled out instead.
#   "torch.cuda" appears in the message of any AttributeError raised while
#                touching a torch.cuda symbol -- including one raised BY
#                glm5_next.py against a torch this image actually ships, which
#                is precisely the defect this script exists to catch.
GPU_ABSENCE = (
    "rocminfo",
    "No HIP GPUs are available",
    "no CUDA-capable device",
    "hipErrorNoDevice",
    "no ROCm-capable device",
    "Found no NVIDIA driver",
    "Torch not compiled with CUDA enabled",
)


def _is_gpu_absence(exc):
    """True only for a device-absence failure raised *outside* the overlay.

    Two conditions, both required. The message test walks the `__cause__` /
    `__context__` chain, because aiter's rocminfo probe is routinely re-raised
    as something whose own `str()` says nothing about GPUs. The frame test is
    the sharper half: whatever the message says, if the innermost frame is in
    glm5_next.py then the overlay is what broke, and a builder without a GPU is
    not the explanation. The usual true-negative -- glm5_next.py's own
    `import aiter` -- puts glm5_next.py in the traceback but never at the
    bottom of it.
    """
    seen, cur, blobs = set(), exc, []
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        blobs.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    joined = "\n".join(blobs).lower()
    if not any(m.lower() in joined for m in GPU_ABSENCE):
        return False

    innermost = None
    cur, seen = exc, set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        frames = traceback.extract_tb(cur.__traceback__)
        if frames:
            innermost = frames[-1].filename
        cur = cur.__cause__ or cur.__context__
    if innermost and pathlib.Path(innermost).resolve() == SRC.resolve():
        return False
    return True


def fail(msg):
    sys.exit(f"[verify-glm53-overlay] FAIL: {msg}")


def main():
    if not SRC.is_file():
        fail(
            f"{SRC} is absent -- the overlay did not land, or a later layer "
            f"reinstalled sglang over it"
        )

    text = SRC.read_text()
    try:
        tree = ast.parse(text, filename=str(SRC))
    except SyntaxError as e:
        fail(f"{SRC} does not parse: {e}")

    classes = {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
    if CLASS not in classes:
        fail(
            f"{SRC} defines no `class {CLASS}` (found {len(classes)} classes) "
            f"-- wrong commit checked out?"
        )

    if "EntryClass" not in text:
        fail(f"{SRC} has no EntryClass export; sglang's loader would never reach {CLASS}")

    compile(text, str(SRC), "exec")  # bytecode-level, not just AST
    print(f"[verify-glm53-overlay] static: {CLASS} defined and exported in {SRC}")

    # Best effort. A GPU-absence error on the builder is expected and passes.
    try:
        mod = importlib.import_module(MODULE)
    except Exception as e:  # noqa: BLE001 - see GPU_ABSENCE
        blob = f"{type(e).__name__}: {e}"
        if _is_gpu_absence(e):
            print(
                f"[verify-glm53-overlay] import skipped, builder has no GPU "
                f"({blob.splitlines()[0][:120]})"
            )
            return
        fail(f"importing {MODULE} raised {blob}")

    if not hasattr(mod, CLASS):
        fail(f"{MODULE} imported but does not expose {CLASS}")
    print(f"[verify-glm53-overlay] import OK: {MODULE}.{CLASS}")


if __name__ == "__main__":
    main()
