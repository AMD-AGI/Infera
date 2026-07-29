#!/usr/bin/env python3
"""Generate the shippable Bug 2b patch from a pristine upstream copy.

Applies ONLY the fix (never the instrumentation) to /tmp/patchgen/eagle_worker_v2.py
and emits a unified diff against upstream HEAD.
"""
import difflib
import os
import sys

sys.path.insert(0, "/home/yihou/glm52_fix/bug2b/r02_fix")
import fix_uniform_draft_graph as F  # noqa: E402

PRISTINE = "/tmp/patchgen/eagle_worker_v2.py.pristine"
WORK = "/tmp/patchgen/eagle_worker_v2.py"
REL = "python/sglang/srt/speculative/eagle_worker_v2.py"
OUT = "/home/yihou/glm52_fix/bug2b/bug2b_uniform_draft_graph.patch"

src = open(PRISTINE).read()
if src.count(F.ANCHOR) != 1:
    print(f"anchor matched {src.count(F.ANCHOR)} times in pristine -- abort")
    sys.exit(2)
patched = src.replace(F.ANCHOR, F.REPLACEMENT, 1)

open(WORK, "w").write(patched)
import py_compile

py_compile.compile(WORK, doraise=True)

diff = difflib.unified_diff(
    src.splitlines(keepends=True),
    patched.splitlines(keepends=True),
    fromfile=f"a/{REL}",
    tofile=f"b/{REL}",
    n=6,
)
body = "".join(diff)
header = f"diff --git a/{REL} b/{REL}\n--- a/{REL}\n+++ b/{REL}\n"
# unified_diff already emits ---/+++ lines; strip them from body
lines = body.splitlines(keepends=True)
while lines and (lines[0].startswith("---") or lines[0].startswith("+++")):
    lines.pop(0)
open(OUT, "w").write(header + "".join(lines))
print(f"wrote {OUT} ({len(header) + len(''.join(lines))} bytes)")
print(f"hunks: {''.join(lines).count('@@ ') // 2}")
