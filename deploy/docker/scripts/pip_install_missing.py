###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Install the requirements of an installed distribution that are absent, and only those.

    python3 pip_install_missing.py amd-infera

A vLLM nightly base pins its own torch / transformers / httpx stack; a plain
``pip install .`` would upgrade any of those whose version misses infera's lower
bound and can leave vLLM importing a version it was never built against. This
installs a requirement only when no version of it is present, and reports, but
leaves alone, an installed version outside the specifier.
"""

import subprocess
import sys
from importlib import metadata

from packaging.requirements import Requirement

dist = sys.argv[1]
missing, mismatched = [], []
for spec in metadata.requires(dist) or []:
    req = Requirement(spec)
    if req.marker is not None and not req.marker.evaluate({"extra": ""}):
        continue
    try:
        have = metadata.version(req.name)
    except metadata.PackageNotFoundError:
        missing.append(str(req))
        continue
    if req.specifier and not req.specifier.contains(have, prereleases=True):
        mismatched.append(f"{req.name} {have} (wants {req.specifier})")

for m in mismatched:
    print(f"[{dist}] kept the base's {m}")
if missing:
    print(f"[{dist}] installing missing: {' '.join(missing)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *missing])
else:
    print(f"[{dist}] no missing requirements")
