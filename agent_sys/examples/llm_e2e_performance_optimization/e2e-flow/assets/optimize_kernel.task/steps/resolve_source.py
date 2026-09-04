#!/usr/bin/env python3
"""Where the operator's source file actually is, on this filesystem.

One resolver, because STEP 3 needs it twice — the seed on the mock branch and
the campaign's output on the real one — and two heredocs computing one path is
two chances for them to compute it differently. That is not hypothetical here:
both heredocs previously resolved `edit_target.source_file` against the
*workset*, and the real branch's copy would have failed after the campaign
hours rather than before it.

**The file is in the engine container, not in the workset.** `edit_target`
names a path relative to `edit_target.repo_root_var`, and m3 generates
`invocation_spec.json` with `image_repo_path: @SGLANG_ROOT@` and its `sources`
under that root — so forge edits the container's sglang checkout. The workset
carries the *apparatus* (driver, cases, entrypoints); it does not carry the
engine's sources and never did.

Resolvable at all only because m1 through m4 share one container (CONTRACT §5):
the tree m5 will patch is on m4's filesystem while m4 runs. Outside it every
answer here names a file that does not exist, and the caller treats that as a
refusal rather than as an empty kernel.

Prints the absolute path on stdout and exits 0; prints why on stderr and exits
1. Nothing is guessed — a path invented here is one m5 would apply to a real
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

    resolved = lib.expand_container_path(str(container_path))
    if resolved is None:
        lib.die(f"{container_path} names a root with no path in container_roots.yaml")

    print(str(resolved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
