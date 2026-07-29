#!/usr/bin/env python3
"""R1 probe: log every input to the draft graph/eager decision, per rank per iteration.

Decides between:
  H1  guard term 4 (dsa_topk_indices is None) is rank-divergent  -> fix term 4
  H2  can_run_graph itself is rank-divergent (all-gather stale/skipped) -> worse bug
  H3  paths agree but dp_padding_mode differs (MAX_LEN captured vs SUM_LEN eager)
      -> all_gather vs all_reduce mismatch

Design notes (each one is a lesson already paid for):

* The probe logs the decision BEFORE it is acted on, so a rank that then blocks
  inside the graph still leaves its record.
* A replayed CUDA graph executes no Python. So we must NOT try to count anything
  inside the graph -- we only record what the host decided, which is visible on
  both arms.
* dp_padding_mode is read off the forward_batch, which is set by
  prepare_mlp_sync_batch on the eager path; for the graph path we record what the
  runner will use (its captured constant), not what the batch says.
* Output goes to stderr with a fixed prefix so `strings <log> | grep` works even
  though server logs contain binary bytes.

Usage:  python3 probe_guard.py [--revert]
"""
import os
import re
import shutil
import sys

SGL = "/sgl-workspace/sglang/python/sglang/srt"
TARGET = f"{SGL}/speculative/eagle_worker_v2.py"
MARKER = "GLM52_R1_PROBE"

PROBE = '''
        # ---- GLM52_R1_PROBE begin ----
        try:
            import os as _os, sys as _sys
            if _os.environ.get("GLM52_R1_PROBE", "0") == "1":
                from sglang.srt.layers.dp_attention import (
                    get_attention_dp_rank as _gdr,
                )
                _r = _gdr()
                _it = getattr(self, "_glm52_it", 0) + 1
                self._glm52_it = _it
                _fm = forward_batch.forward_mode
                # the four guard terms, each evaluated separately
                _t1 = bool(can_cuda_graph)                       # from prepare_for_draft
                _t2 = not _fm.is_idle()
                _t3 = bool(self.seed_dsa_topk_from_draft_extend)
                _t4 = draft_input.dsa_topk_indices is None
                _final = _t1 and not (_t1 and _t2 and _t3 and _t4)
                # H2 inputs: is can_run_graph's own input rank-uniform?
                _gnt = getattr(forward_batch, "global_num_tokens_cpu", None)
                _cdg = getattr(forward_batch, "can_run_dp_cuda_graph", None)
                # H3 input: which padding mode / collective the eager path will use
                _pm = getattr(forward_batch, "dp_padding_mode", None)
                _pm = _pm.name if _pm is not None else None
                _bs = getattr(forward_batch, "batch_size", None)
                _nreq = len(batch.reqs) if getattr(batch, "reqs", None) is not None else -1
                _sys.stderr.write(
                    "GLM52_R1 dp=%s it=%s mode=%s bs=%s nreq=%s "
                    "t1_cangraph=%s t2_notidle=%s t3_seed=%s t4_topknone=%s final=%s "
                    "gnt=%s can_dp_cg=%s padmode=%s\\n"
                    % (_r, _it, _fm.name, _bs, _nreq,
                       _t1, _t2, _t3, _t4, _final, _gnt, _cdg, _pm)
                )
                _sys.stderr.flush()
        except Exception as _e:
            import sys as _sys
            _sys.stderr.write("GLM52_R1_ERR %r\\n" % (_e,))
            _sys.stderr.flush()
        # ---- GLM52_R1_PROBE end ----
'''

ANCHOR = """        if (
            can_cuda_graph
            and not forward_batch.forward_mode.is_idle()
            and self.seed_dsa_topk_from_draft_extend
            and draft_input.dsa_topk_indices is None
        ):
            can_cuda_graph = False
"""


def purge_pyc(path):
    """A .pyc compiled from the unpatched source silently wins if mtimes match.

    This exact failure invalidated a full experiment once (WARMUP_MATRIX run 5).
    """
    os.utime(path, None)
    d, f = os.path.split(path)
    pyc_dir = os.path.join(d, "__pycache__")
    if os.path.isdir(pyc_dir):
        stem = f[:-3]
        for name in os.listdir(pyc_dir):
            if name.startswith(stem + "."):
                os.remove(os.path.join(pyc_dir, name))
                print(f"  purged {name}")


def main():
    revert = "--revert" in sys.argv
    bak = TARGET + ".r1bak"

    if revert:
        if not os.path.exists(bak):
            print("no backup, nothing to revert")
            return 1
        shutil.copyfile(bak, TARGET)  # copyfile, NOT copy2 -- do not restore mtime
        os.remove(bak)
        purge_pyc(TARGET)
        print("reverted")
        return 0

    src = open(TARGET).read()
    if MARKER in src:
        print("already instrumented")
        return 0
    if ANCHOR not in src:
        print("ANCHOR NOT FOUND -- source drifted, refusing to patch blindly")
        return 2

    if not os.path.exists(bak):
        shutil.copyfile(TARGET, bak)

    # probe goes BEFORE the guard so it records the decision inputs even if the
    # rank never returns from what follows
    new = src.replace(ANCHOR, PROBE + ANCHOR, 1)
    open(TARGET, "w").write(new)
    purge_pyc(TARGET)

    import py_compile

    py_compile.compile(TARGET, doraise=True)
    print(f"instrumented {TARGET}")
    print(f"  marker count in source: {open(TARGET).read().count(MARKER)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
