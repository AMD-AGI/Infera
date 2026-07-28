#!/usr/bin/env python3
"""Probe 2: pinpoint WHERE inside dsa_backend.init_forward_metadata the decode
leg stalls, and whether ranks agree on getting there.

Context: probe 1's hypothesis (per-rank `can_cuda_graph` divergence) is REFUTED
by the logs -- the draft-extend CUDA graph is never captured on HIP
("Capture draft extend CUDA graph begin" appears 0 times, only "draft decode"
does), so `cuda_graph_runner_for_draft_extend is None` and `can_cuda_graph` is
falsy on EVERY rank. All ranks take the eager `init_forward_metadata` path.

So the divergence must be INSIDE init_forward_metadata, or the ranks are not
even all arriving at draft-extend. This probe logs entry/exit plus the
individual sync points, rank-tagged, so the hung rank's last line identifies
the exact blocking statement.

Idempotent. Revert with --revert.
"""
import argparse
import os
import shutil
import sys

TARGET = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"
BACKUP_SUFFIX = ".bug2_orig"
MARKER = "GLM52_BUG2_DSA_STALL"

ANCHOR = '''    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device
'''

PROBE = '''    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        batch_size = forward_batch.batch_size
        device = forward_batch.seq_lens.device
        # ''' + MARKER + '''  (debug only)
        try:
            import logging as _lg, os as _os
            _r = _os.environ.get("SGLANG_DP_RANK", _os.environ.get("RANK", "?"))
            _log = _lg.getLogger(__name__)
            _slc = forward_batch.seq_lens_cpu
            _log.warning(
                "[''' + MARKER + '''] ENTER r=%s mode=%s bs=%s seq_lens_numel=%s "
                "seq_lens_cpu_is_none=%s gnt_cpu=%s",
                _r, str(forward_batch.forward_mode), batch_size,
                int(forward_batch.seq_lens.numel()),
                _slc is None,
                getattr(forward_batch, "global_num_tokens_cpu", None),
            )
        except Exception:
            pass
'''

# second anchor: right after the max_seqlen_k computation, to see if the .item() returned
ANCHOR2 = '''        # [b, max_seqlen_k]
        page_table = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]
'''

PROBE2 = '''        try:
            import logging as _lg, os as _os
            _lg.getLogger(__name__).warning(
                "[''' + MARKER + '''] GOT_MAXSEQLEN r=%s max_seqlen_k=%s",
                _os.environ.get("SGLANG_DP_RANK", _os.environ.get("RANK", "?")),
                max_seqlen_k,
            )
        except Exception:
            pass
        # [b, max_seqlen_k]
        page_table = self.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices, :max_seqlen_k
        ]
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
        print(f"OK: reverted {TARGET}")
        return

    if MARKER in src:
        print("OK: probe already present (no-op)")
        return

    for name, a in (("ANCHOR", ANCHOR), ("ANCHOR2", ANCHOR2)):
        n = src.count(a)
        if n != 1:
            sys.exit(f"FAIL: {name} matched {n} times, expected 1. Source drifted.")

    if not os.path.exists(backup):
        shutil.copyfile(TARGET, backup)
        print(f"OK: backup -> {backup}")

    out = src.replace(ANCHOR, PROBE, 1).replace(ANCHOR2, PROBE2, 1)
    open(TARGET, "w").write(out)

    import py_compile
    try:
        py_compile.compile(TARGET, doraise=True)
    except Exception as e:
        shutil.copyfile(backup, TARGET)
        sys.exit(f"FAIL: broke syntax, reverted. {e}")

    print(f"OK: probe installed in {TARGET}")


if __name__ == "__main__":
    main()
