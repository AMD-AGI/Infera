#!/usr/bin/env python3
"""Instrument the PD+MTP draft-extend divergence (Bug 2).

Adds a rank-tagged log line immediately before the conditional
`init_forward_metadata` call in base_spec_worker.prepare_for_draft_extend.

Hypothesis under test: `is_idle()` and/or `can_cuda_graph` evaluate
DIFFERENTLY per DP rank, so some ranks take the eager
`init_forward_metadata` path (which does a .item() GPU->CPU sync) while
others skip straight into the next collective -> ragged collective ->
deadlock. py-spy showed a 3-way split: DP0,3,5,7 in broadcast; DP2,4,6 in
all_gather_into_tensor; DP1 inside init_forward_metadata.

Idempotent: re-running is a no-op. Restore with --revert.
"""
import argparse
import os
import re
import shutil
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/speculative/base_spec_worker.py"
BACKUP_SUFFIX = ".bug2_orig"
MARKER = "GLM52_BUG2_DIVERGENCE_PROBE"

# The exact anchor: the conditional whose per-rank divergence we suspect.
ANCHOR = """        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run_graph(
            forward_batch
        )
        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
"""

PROBE = '''        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run_graph(
            forward_batch
        )
        # ''' + MARKER + '''  (debug only; remove before shipping)
        try:
            import logging as _lg, os as _os, torch as _t
            _dp = getattr(draft_model_runner, "dp_rank", None)
            if _dp is None:
                _dp = getattr(getattr(draft_model_runner, "server_args", None), "dp_rank", None)
            if _dp is None:
                _dp = _os.environ.get("SGLANG_DP_RANK", "?")
            _tp = getattr(draft_model_runner, "tp_rank", "?")
            _idle = bool(batch.forward_mode.is_idle())
            _ccg = bool(can_cuda_graph)
            _taken = (not _idle) and (not _ccg)
            _gnt = getattr(forward_batch, "global_num_tokens_cpu", None)
            _crd = getattr(forward_batch, "can_run_dp_cuda_graph", None)
            _sl = getattr(forward_batch, "seq_lens", None)
            _sln = int(_sl.numel()) if _sl is not None else -1
            _lg.getLogger(__name__).warning(
                "[''' + MARKER + '''] dp=%s tp=%s mode=%s idle=%s can_cuda_graph=%s "
                "TAKES_EAGER_PATH=%s bs=%s seq_lens_numel=%s global_num_tokens_cpu=%s "
                "can_run_dp_cuda_graph=%s",
                _dp, _tp, str(batch.forward_mode), _idle, _ccg, _taken, bs, _sln, _gnt, _crd,
            )
        except Exception as _e:  # never let the probe break the run
            import logging as _lg
            _lg.getLogger(__name__).warning("[''' + MARKER + '''] probe failed: %r", _e)
        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(TARGET):
        sys.exit(f"FAIL: target not found: {TARGET}")

    backup = TARGET + BACKUP_SUFFIX
    src = open(TARGET).read()

    if args.revert:
        if not os.path.exists(backup):
            sys.exit(f"FAIL: no backup at {backup}")
        shutil.copyfile(backup, TARGET)
        print(f"OK: reverted {TARGET} from {backup}")
        return

    if MARKER in src:
        print("OK: probe already present (idempotent no-op)")
        return

    n = src.count(ANCHOR)
    if n != 1:
        sys.exit(f"FAIL: expected exactly 1 anchor match, found {n}. Source drifted.")

    if not os.path.exists(backup):
        shutil.copyfile(TARGET, backup)
        print(f"OK: backup -> {backup}")

    out = src.replace(ANCHOR, PROBE, 1)
    open(TARGET, "w").write(out)

    # verify it parses
    import py_compile
    try:
        py_compile.compile(TARGET, doraise=True)
    except Exception as e:
        shutil.copyfile(backup, TARGET)
        sys.exit(f"FAIL: probe broke syntax, reverted. {e}")

    print(f"OK: probe installed in {TARGET}")
    print(f"OK: grep for [{MARKER}] in the decode log")


if __name__ == "__main__":
    main()
