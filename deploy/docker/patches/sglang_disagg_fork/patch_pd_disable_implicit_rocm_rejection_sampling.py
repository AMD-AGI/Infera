#!/usr/bin/env python3
"""Do not implicitly enable EAGLE rejection sampling on a PD decode worker.

ROCm's v0.5.18 speculative hook enables rejection sampling by default. A
disaggregated decode worker cannot use that path because the current PD
handoff does not carry ``EagleDraftInput.draft_probs`` from prefill. Its
startup warmup consequently puts ``None`` in ``draft_probs_list`` and crashes
at ``torch.stack``.

An operator can still request rejection sampling explicitly. Infera's GLM-5.2
PD recipe leaves it off and uses temperature=0 for real-token correctness;
the AgentX performance run uses simulated acceptance.
"""

from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path

TAG = "[pd-rocm-rejection-sampling]"
MARKER = "PD decode does not receive draft_probs from prefill"
OLD = """    if (
        is_hip()
        and not server_args.speculative_use_rejection_sampling
"""
NEW = f"""    if (
        is_hip()
        # {MARKER}; do not implicitly select the incompatible path.
        and server_args.disaggregation_mode != "decode"
        and not server_args.speculative_use_rejection_sampling
"""


def main() -> int:
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        raise SystemExit(f"{TAG} cannot locate sglang")
    path = Path(spec.origin).parent / "srt" / "arg_groups" / "speculative_hook.py"
    source = path.read_text()
    if MARKER in source:
        print(f"{TAG} already present — skipping")
        return 0
    if source.count(OLD) != 1:
        raise SystemExit(f"{TAG} expected one ROCm default-enable anchor in {path}")
    path.write_text(source.replace(OLD, NEW, 1))
    py_compile.compile(str(path), doraise=True)
    print(f"{TAG} implicit rejection sampling is disabled for PD decode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
