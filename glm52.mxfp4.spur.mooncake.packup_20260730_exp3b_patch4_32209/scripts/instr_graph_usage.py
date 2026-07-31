#!/usr/bin/env python3
"""Measure whether the draft CUDA graph is ACTUALLY REPLAYED, per rank.

Why this exists
---------------
Charter criterion 5 for this bug is "the graph path is provably taken (marker
count > 0)" -- NOT "no hang". Variant B already gets a hang-free run by forcing
the draft path eager. So a green conc=32 proves nothing on its own: an arm that
silently never uses the draft graph looks exactly like an arm that uses it and
works.

This matters specifically for the #32209-style patch 4 (`patch4_32209_style.py`)
because of one deliberate divergence in it:

    if getattr(draft_input, "future_indices", None) is not None:
        return True          # require eager

Upstream reads `future_dsa_topk_indices_available` there; our baseline has no
such attribute, so the port falls back to REQUIRING EAGER whenever overlap
scheduling has left the inputs unresolved. If `future_indices` is set on most
iterations, that fallback silently converts patch 4 into Variant B -- correct,
hang-free, and useless as a fix. The only way to tell is to count.

What it records
---------------
At the ONE site that decides it -- `EAGLEDraftCudaGraphRunner.can_run_graph` --
per call:

  * the group-wide gate `can_run_dp_draft_cuda_graph` (the all-gathered vote)
  * the local `can_run_dp_cuda_graph` (pre-existing gate, for contrast)
  * whether bs was supported at all
  * the FINAL answer -- i.e. was the graph replayed this iteration

and, at the rank-local predicate site, why a rank asked for eager:

  * `future_indices is not None`  (the ported-divergence fallback)
  * `dsa_topk_indices is None`    (the real guard term 4)

Summary lines are emitted every `EVERY` calls and once at process exit, so the
answer survives a crash and does not require the run to end cleanly.

Read the result with:  common/analyze_graph_usage.sh <decode.log>

Idempotent. Anchors asserted unique. Invalidates the .pyc (a stale .pyc
silently reverts a patch -- CLAUDE.md).
"""
import os
import sys

ROOT = "/sgl-workspace/sglang/python/sglang/srt"
MARK = "GLM52_GUSE"
EVERY = 200

RUNNER = os.path.join(ROOT, "speculative/eagle_draft_cuda_graph_runner.py")
WORKER = os.path.join(ROOT, "speculative/eagle_worker_v2.py")

# ---------------------------------------------------------------- counters --
PREAMBLE = '''
# GLM52_GUSE -- draft-graph usage counter; see common/instr_graph_usage.py
import atexit as _guse_atexit
import logging as _guse_logging

_guse_log = _guse_logging.getLogger("glm52.guse")
_GUSE = {{
    "calls": 0,
    "graph": 0,          # can_run_graph returned True -> graph replayed
    "no_bs": 0,          # refused on batch size / padding
    "no_dp": 0,          # refused by can_run_dp_cuda_graph (pre-existing gate)
    "no_draft_vote": 0,  # refused by can_run_dp_draft_cuda_graph (patch 4)
}}
_GUSE_EVERY = {every}


def _guse_rank():
    try:
        import torch.distributed as _d

        return _d.get_rank() if _d.is_initialized() else -1
    except Exception:
        return -1


def _guse_emit(tag):
    c = _GUSE["calls"] or 1
    _guse_log.info(
        "GLM52_GUSE %s rank=%s calls=%d graph=%d (%.1f%%) "
        "refused_bs=%d refused_dp=%d refused_draftvote=%d",
        tag,
        _guse_rank(),
        _GUSE["calls"],
        _GUSE["graph"],
        100.0 * _GUSE["graph"] / c,
        _GUSE["no_bs"],
        _GUSE["no_dp"],
        _GUSE["no_draft_vote"],
    )
    for h in list(_guse_log.handlers) + list(_guse_logging.getLogger().handlers):
        try:
            h.flush()
        except Exception:
            pass


def _guse_record(is_bs_supported, dp_ok, draft_vote_ok, final):
    _GUSE["calls"] += 1
    if final:
        _GUSE["graph"] += 1
    elif not is_bs_supported:
        _GUSE["no_bs"] += 1
    elif not dp_ok:
        _GUSE["no_dp"] += 1
    elif not draft_vote_ok:
        _GUSE["no_draft_vote"] += 1
    if _GUSE["calls"] % _GUSE_EVERY == 0:
        _guse_emit("periodic")


_guse_atexit.register(lambda: _guse_emit("final"))

'''

# The instrumented tail of can_run_graph. Anchored on the patched form that
# patch4_32209_style.py produces, so this script REQUIRES that patch to be
# applied first -- which is the point: it measures that patch.
RUNNER_ANCHOR = """            is_bs_supported = (
                is_bs_supported
                and forward_batch.can_run_dp_cuda_graph
                and forward_batch.can_run_dp_draft_cuda_graph
            )"""

RUNNER_REPL = """            # GLM52_GUSE: capture each gate separately BEFORE folding them
            # together, so a refusal can be attributed to the right cause.
            _guse_bs = bool(is_bs_supported)
            _guse_dp = bool(forward_batch.can_run_dp_cuda_graph)
            _guse_dv = bool(forward_batch.can_run_dp_draft_cuda_graph)
            is_bs_supported = _guse_bs and _guse_dp and _guse_dv
            _guse_record(_guse_bs, _guse_dp, _guse_dv, bool(is_bs_supported))"""

# ------------------------------------------------------- why-eager counter --
WORKER_ANCHOR = """        if getattr(draft_input, "future_indices", None) is not None:
            return not draft_input.future_dsa_topk_indices_available
        return getattr(draft_input, "dsa_topk_indices", None) is None"""

WORKER_REPL = """        # GLM52_GUSE: attribute each eager request to its cause.
        #   future_seed_missing -- overlap path, flag says the seed will NOT be
        #                          there after resolve (the real term 4, one
        #                          iteration early)
        #   future_seed_ok      -- overlap path, graph allowed
        #   seed_none           -- non-overlap path, term 4 fired
        #   graph_ok            -- non-overlap path, graph allowed
        # If eager dominates while the two *_ok buckets stay at zero, the port
        # has degenerated into Variant B and the draft graph is never used.
        if getattr(draft_input, "future_indices", None) is not None:
            _guse_eager = not draft_input.future_dsa_topk_indices_available
            _GUSE_WHY["future_seed_missing" if _guse_eager else "future_seed_ok"] += 1
            if sum(_GUSE_WHY.values()) % 200 == 0:
                _guse_why_emit()
            return _guse_eager
        _guse_seed_missing = getattr(draft_input, "dsa_topk_indices", None) is None
        _GUSE_WHY["seed_none" if _guse_seed_missing else "graph_ok"] += 1
        if sum(_GUSE_WHY.values()) % 200 == 0:
            _guse_why_emit()
        return _guse_seed_missing"""

WORKER_PREAMBLE = '''
# GLM52_GUSE -- why-eager counter; see common/instr_graph_usage.py
import atexit as _guse_w_atexit
import logging as _guse_w_logging

_guse_w_log = _guse_w_logging.getLogger("glm52.guse.why")
_GUSE_WHY = {
    "future_seed_missing": 0,
    "future_seed_ok": 0,
    "seed_none": 0,
    "graph_ok": 0,
}


def _guse_why_emit():
    t = sum(_GUSE_WHY.values()) or 1
    try:
        import torch.distributed as _d

        rank = _d.get_rank() if _d.is_initialized() else -1
    except Exception:
        rank = -1
    _guse_w_log.info(
        "GLM52_GUSE_WHY rank=%s total=%d future_seed_missing=%d (%.1f%%) "
        "future_seed_ok=%d (%.1f%%) seed_none=%d (%.1f%%) graph_ok=%d (%.1f%%)",
        rank,
        t,
        _GUSE_WHY["future_seed_missing"],
        100.0 * _GUSE_WHY["future_seed_missing"] / t,
        _GUSE_WHY["future_seed_ok"],
        100.0 * _GUSE_WHY["future_seed_ok"] / t,
        _GUSE_WHY["seed_none"],
        100.0 * _GUSE_WHY["seed_none"] / t,
        _GUSE_WHY["graph_ok"],
        100.0 * _GUSE_WHY["graph_ok"] / t,
    )
    for h in list(_guse_w_log.handlers) + list(_guse_w_logging.getLogger().handlers):
        try:
            h.flush()
        except Exception:
            pass


_guse_w_atexit.register(_guse_why_emit)

'''


def die(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def insert_preamble(src, text):
    """Put the counter block after the import block, before the first def/class."""
    lines = src.split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith(("def ", "class ", "@")):
            return "\n".join(lines[:i]) + text + "\n".join(lines[i:])
    die("no top-level def/class found to anchor the preamble before")


def patch_file(path, preamble, anchor, repl):
    src = open(path).read()
    if MARK in src:
        print(f"  already instrumented: {os.path.relpath(path, ROOT)}")
        return False
    n = src.count(anchor)
    if n != 1:
        die(
            f"{os.path.relpath(path, ROOT)}: anchor matched {n}x, want 1.\n"
            "Is patch4_32209_style.py applied? This instrumentation anchors on "
            "the form that patch produces.\n---\n" + anchor[:300]
        )
    src = src.replace(anchor, repl, 1)
    src = insert_preamble(src, preamble)
    open(path, "w").write(src)
    os.utime(path, None)
    d, base = os.path.split(path)
    pc = os.path.join(d, "__pycache__")
    if os.path.isdir(pc):
        stem = base[:-3] + "."
        for f in os.listdir(pc):
            if f.startswith(stem):
                os.remove(os.path.join(pc, f))
    print(f"  instrumented {os.path.relpath(path, ROOT)} ({src.count(MARK)} markers)")
    return True


def main():
    patch_file(RUNNER, PREAMBLE.format(every=EVERY), RUNNER_ANCHOR, RUNNER_REPL)
    patch_file(WORKER, WORKER_PREAMBLE, WORKER_ANCHOR, WORKER_REPL)
    print("ok")


if __name__ == "__main__":
    main()
