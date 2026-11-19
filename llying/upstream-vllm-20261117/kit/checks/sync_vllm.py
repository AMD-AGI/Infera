#!/usr/bin/env python3
"""Make the installed vllm package (in a throwaway container) match a source tree's Python code: copy every file of
SRC/vllm except compiled extensions, and delete installed Python files that the tree no longer has. The image's
compiled extensions (*.so) and its generated _version.py stay, so the tree must not change ROCm compiled code relative
to the image's commit (the step checks that with git before calling this).
  sync_vllm.py SRC_TREE
"""

import os
import shutil
import sys

import vllm

src = os.path.join(sys.argv[1], "vllm")
dst = os.path.dirname(vllm.__file__)
keep = {"_version.py"}
copied = removed = 0
for root, dirs, files in os.walk(src):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    rel = os.path.relpath(root, src)
    os.makedirs(os.path.join(dst, rel), exist_ok=True)
    for f in files:
        if f.endswith(".so"):
            continue
        shutil.copy2(os.path.join(root, f), os.path.join(dst, rel, f))
        copied += 1
for root, dirs, files in os.walk(dst):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    rel = os.path.relpath(root, dst)
    for f in files:
        if f.endswith(".py") and f not in keep and not os.path.exists(os.path.join(src, rel, f)):
            os.remove(os.path.join(root, f))
            removed += 1
print(f"synced {src} -> {dst}: {copied} files copied, {removed} stale .py removed")
