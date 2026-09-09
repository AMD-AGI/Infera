#!/usr/bin/env python3
"""`60_write_handoff._apply_block` must take its shape from `engine.patch`.

**Shown failing first, and the four cases are the four things STEP 3 can hand
it.** No node, no torch, no container.

`apply.py:665` selects by the entry's shape and never by `apply_mode` — a bare
`if entry.get("patch")` — and `kernel_optimization.schema.json`'s
`apply/files/items` carries a `oneOf` making the two exclusive. So the entry is
a choice, and this control is what decides it is made from evidence rather than
from a default:

  1/2. **absent or empty diff -> `replacement`.** Forge reverting every candidate is
     `improved: false`, a result and not an error. The whole-file path stays.
  3. **one-file diff -> `patch`,** a bare filename, with the diff written to
     `apply/patches/<name>`. A prefixed value would resolve to
     `apply/patches/apply/patches/x.patch` (`apply.py:409`, `:666`).
  4. **many-file diff -> refusal.** `apply.py:655-663` stages exactly this
     entry's own file into `tree/<rel>` and runs `patch -p1` there, so a hunk
     for any other path dies as `No file to patch` — after the campaign hours.
     Trimming the diff to fit would deliver part of a change and call it the
     change.

Case 4 is the one worth having: forge's own `git add -u` comment says an agent
commonly lands the winning change in a sibling module the kernel imports, so a
multi-file diff is the expected case rather than a pathological one.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve()
PKG = HERE.parent.parent.parent.parent          # .../e2e-flow
STEPS = PKG / "assets" / "optimize_kernel.task" / "steps"

sys.path.insert(0, str(STEPS))
sys.path.insert(0, str(PKG / "assets" / "lib"))

_spec = importlib.util.spec_from_file_location("_wh", STEPS / "60_write_handoff.py")
_wh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_wh)

REL = "srt/layers/sampler.py"

ONE_FILE = f"""diff --git a/{REL} b/{REL}
index e83e415..2e83f4e 100644
--- a/{REL}
+++ b/{REL}
@@ -1,3 +1,4 @@
+# the campaign's edit
 import logging
 from typing import Callable
"""

TWO_FILES = ONE_FILE + """diff --git a/srt/layers/sampler_config.py b/srt/layers/sampler_config.py
index 1111111..2222222 100644
--- a/srt/layers/sampler_config.py
+++ b/srt/layers/sampler_config.py
@@ -1,1 +1,2 @@
+DEFAULT_BLOCK = 256
 X = 1
"""

#: A real 64-hex digest is required by the schema and by the cross-check; the
#: value is arbitrary here because no stock file is reachable off-node, which is
#: exactly why the fixture declares it rather than letting the body measure one.
STOCK_SHA = "a" * 64


def _pinned() -> dict:
    return {
        "operator_id": "sampler_vocab_softmax",
        "edit_target": {"repo_root_var": "@SGLANG_ROOT@", "source_file": REL},
        "integration": {
            "target_files": [REL],
            "apply_mode": "overlay_files",
            "base_sha256": {REL: STOCK_SHA},
        },
    }


def _run(diff_text: str | None) -> tuple[dict | None, str]:
    """Returns (the one file entry, ""), or (None, the refusal's message)."""
    with tempfile.TemporaryDirectory(prefix="m4-engine-patch-") as tmp:
        root = pathlib.Path(tmp)
        packup, forge = root / "packup", root / "forge"
        (packup / "results").mkdir(parents=True)
        forge.mkdir()
        kernel = packup / "results" / "optimized_kernel.py"
        kernel.write_text("def run(**kw):\n    return None\n", encoding="utf-8")
        if diff_text is not None:
            (forge / "engine.patch").write_text(diff_text, encoding="utf-8")
        # **stderr, not `str(SystemExit)`.** `_lib.die` prints the reason and
        # then raises `SystemExit(1)`, so the exception carries the exit code
        # and the sentence is on the stream. A first version of this control
        # asserted against `str(exit_)`, matched `"1"`, and reported case 4 as
        # failing while the body was refusing correctly and saying exactly the
        # right thing.
        captured = io.StringIO()
        try:
            with contextlib.redirect_stderr(captured):
                block = _wh._apply_block(_pinned(), packup, kernel, {}, forge)
        except SystemExit:
            return None, captured.getvalue()
        entry = (block.get("files") or [None])[0]
        if entry and entry.get("patch"):
            # Read back from disk: the entry naming a patch that was never
            # written is the defect rung 0 hit once already.
            written = packup / "apply" / "patches" / entry["patch"]
            entry = dict(entry, _written=written.is_file(),
                         _bytes=written.stat().st_size if written.is_file() else 0)
        return entry, ""


def main() -> int:
    failures: list[str] = []

    # ---- case 1: no diff at all (a mock, or STEP 3 never ran) --------------
    entry, refusal = _run(None)
    if refusal or not entry or entry.get("replacement") != "results/optimized_kernel.py":
        failures.append(f"1 absent engine.patch should give a replacement entry; got {entry} {refusal}")
    elif "patch" in entry:
        failures.append("1 absent engine.patch gave BOTH keys; the schema's oneOf refuses that")
    else:
        print("ok   1 absent engine.patch -> replacement")

    # ---- case 2: an empty diff (improved: false) ---------------------------
    entry, refusal = _run("")
    if refusal or not entry or entry.get("replacement") != "results/optimized_kernel.py":
        failures.append(f"2 empty engine.patch should give a replacement entry; got {entry} {refusal}")
    else:
        print("ok   2 empty engine.patch -> replacement (improved: false is a result)")

    # ---- case 3: the campaign edited the one target ------------------------
    entry, refusal = _run(ONE_FILE)
    if refusal or not entry:
        failures.append(f"3 a one-file diff should be accepted; refused with {refusal}")
    elif entry.get("patch") != "sampler_vocab_softmax.patch":
        failures.append(f"3 expected a bare patch filename; got {entry.get('patch')!r}")
    elif "replacement" in entry:
        failures.append("3 emitted BOTH patch and replacement; the schema's oneOf refuses that")
    elif not entry.get("_written"):
        failures.append("3 the entry names a patch file that was never written to apply/patches/")
    elif entry["_bytes"] != len(ONE_FILE.encode()):
        failures.append(f"3 the written patch is {entry['_bytes']} bytes, not the diff's {len(ONE_FILE.encode())}")
    else:
        print(f"ok   3 one-file diff -> patch entry, {entry['_bytes']} bytes on disk")

    # ---- case 4: forge also touched a sibling ------------------------------
    entry, refusal = _run(TWO_FILES)
    if entry is not None:
        failures.append("4 a two-file diff was ACCEPTED into a single-file entry; "
                        "apply_patch would die with 'No file to patch' after the campaign hours")
    elif "sampler_config.py" not in refusal:
        failures.append(f"4 refused, but without naming the foreign file: {refusal[:200]}")
    else:
        print("ok   4 two-file diff -> refused, naming the file that does not fit")

    for line in failures:
        print(f"FAIL {line}")
    print(f"\n{len(failures)} failed" if failures else "\n4 cases passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
