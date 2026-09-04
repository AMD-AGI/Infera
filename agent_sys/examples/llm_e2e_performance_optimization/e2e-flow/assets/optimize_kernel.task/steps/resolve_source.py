#!/usr/bin/env python3
"""Where the operator's source file actually is, on this filesystem.

One resolver, because STEP 3 needs it twice — the seed on the mock branch and
the campaign's output on the real one — and two heredocs computing one path is
two chances for them to compute it differently. That is not hypothetical here:
both heredocs previously resolved `edit_target.source_file` against the
*workset*, and the real branch's copy would have failed after the campaign
hours rather than before it.

**The file is in the engine container, not in the workset.** `edit_target`
names a path relative to `edit_target.repo_root_var`, and the workset carries
the *apparatus* (driver, cases, entrypoints); it does not carry the engine's
sources and never did.

Resolvable at all only because m1 through m4 share one container (CONTRACT §5):
the tree m5 will patch is on m4's filesystem while m4 runs. Outside it every
answer here names a file that does not exist, and the caller treats that as a
refusal rather than as an empty kernel.

**`--relative` prints the tail instead, and that is not a convenience.** After
the campaign the optimised file is not in the container's tree — STEP 3 hands
forge a *copy* as `--workspace` and forge edits and commits there
(`kernel_agents/loop/runner.py:1600`). The tail is the one part of the answer
that is the same in both trees, because the copy is taken at `SGLANG_ROOT`
itself. An earlier note here said forge edits the container's checkout because
m3's spec carries `image_repo_path: @SGLANG_ROOT@`; `image_repo_path` appears
nowhere in KernelForge's source, and the spec is the measurement contract, not
an edit target.

Prints the path on stdout and exits 0; prints why on stderr and exits 1.
Nothing is guessed — a path invented here is one m5 would apply to a real
image.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _lib as lib  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", required=True, help="STEP 1's inputs.json")
    ap.add_argument("--what", default="the source file", help="for the diagnostic")
    ap.add_argument("--relative", action="store_true",
                    help="print the path relative to its container root, not the absolute one")
    a = ap.parse_args()

    pinned = lib.load_json(Path(a.inputs))
    target = pinned.get("edit_target") or {}
    source_file = str(target.get("source_file") or "").strip()
    if not source_file:
        lib.die("the workset's edit_target declares no source_file; there is nothing to resolve")

    container_path = lib.container_path_for(source_file, str(target.get("repo_root_var") or ""))
    if not container_path:
        lib.die(
            f"cannot place {source_file!r} under any root in assets/lib/container_roots.yaml "
            f"(edit_target.repo_root_var is {target.get('repo_root_var')!r}). Refused rather than "
            f"guessed: {a.what} resolved by inference is a file m5 would patch in a real image"
        )

    if a.relative:
        # `patchkit.split_placeholder` and not a local partition: the same split
        # decides where `apply.py:637` extracts to, and two implementations of
        # one frame conversion is how this package earned its "No file to patch"
        # (CONTRACT §4.3, one authority).
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "lib"))
        import patchkit  # noqa: E402 — the path insert above is what makes it importable

        _root, relative = patchkit.split_placeholder(str(container_path))
        print(relative)
        return 0

    resolved = lib.expand_container_path(str(container_path))
    if resolved is None:
        lib.die(f"{container_path} names a root with no path in container_roots.yaml")

    print(str(resolved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
