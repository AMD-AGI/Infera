#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""The utils.py + mooncake/conn.py half of ../patch_mooncake_early_send_wait_event.py.

The rocm10.2 v0.5.21 base sits between the two existing recipes: prefill.py
already records `_early_send_wait_event` for the early-send path (and its
non-final-chunk assert was reformatted, so the parent script's prefill.py anchor
no longer matches), but mooncake/conn.py still never waits on it, unlike
v0.5.22 (sglang#41395). This runs the parent's edits for the two consumer files
only; ../v0522/patch_prefill_nonfinal_chunk_wait_event.py then adds the
non-final overlap-send event, and its consumer check passes on what this wrote.

All-or-nothing like the parent: an anchor that drifted writes nothing, exit 1.
"""

import importlib.util
import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent / "patch_mooncake_early_send_wait_event.py"


def main() -> int:
    spec = importlib.util.spec_from_file_location("mc_early_send_wait_event", _PARENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._EDITS.pop("disaggregation/prefill.py")
    return mod.main()


if __name__ == "__main__":
    sys.exit(main())
