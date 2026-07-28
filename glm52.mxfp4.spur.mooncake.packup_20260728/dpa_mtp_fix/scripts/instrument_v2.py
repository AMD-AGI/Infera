#!/usr/bin/env python3
"""Probe v2: find the REAL divergence, now that the uniform-branch fix is in.

Status: fix_bug2.py makes every DP rank enter `init_forward_metadata`
(verified: the stuck rank's frame is base_spec_worker.py:182, the patched line).
The deadlock persists, and the stuck rank is now blocked at
`dsa_backend.py:752` -- which is the **else** arm:

    if forward_batch.seq_lens_cpu is not None:
        max_seqlen_k = ...seq_lens_cpu.max()...     # host read, cheap
    else:
        max_seqlen_k = ...seq_lens.max().item()...  # <-- line 752, GPU->CPU SYNC

So the surviving divergence is `seq_lens_cpu is None` differing per rank
(`gpu_only = batch.seq_lens_cpu is None`, base_spec_worker.py:112). A rank on
the else-arm issues a blocking device sync while its peers, on the host-read
arm, sail into the next collective.

This probe logs, per rank and per call:
  - seq_lens_cpu present or None      (which arm of the max_seqlen_k branch)
  - seq_lens numel, forward mode, global_num_tokens_cpu
so we can check whether `seq_lens_cpu is None` is rank-uniform.

Install on top of fix_bug2.py (anchors target the PATCHED text).
Idempotent. --revert restores.
"""
import argparse
import os
import shutil
import sys

DSA = "/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py"
BACKUP_SUFFIX = ".probe_v2_orig"
MARKER = "GLM52_BUG2_ARM_PROBE"

# anchor on the post-fix text
ANCHOR = """        cache_seqlens_int32 = (forward_batch.seq_lens + draft_token_num).to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
        if forward_batch.seq_lens_cpu is not None:
"""

REPLACEMENT = '''        cache_seqlens_int32 = (forward_batch.seq_lens + draft_token_num).to(torch.int32)
        cu_seqlens_k = compute_cu_seqlens(cache_seqlens_int32)
        # ''' + MARKER + '''  (debug only)
        try:
            import logging as _lg, os as _os
            _lg.getLogger(__name__).warning(
                "[''' + MARKER + '''] r=%s mode=%s seq_lens_cpu_is_None=%s "
                "numel=%s gnt=%s -> arm=%s",
                _os.environ.get("SGLANG_DP_RANK", _os.environ.get("RANK", "?")),
                str(forward_batch.forward_mode),
                forward_batch.seq_lens_cpu is None,
                int(forward_batch.seq_lens.numel()),
                getattr(forward_batch, "global_num_tokens_cpu", None),
                "HOST" if forward_batch.seq_lens_cpu is not None else "GPU_SYNC",
            )
        except Exception:
            pass
        if forward_batch.seq_lens_cpu is not None:
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(DSA):
        sys.exit(f"FAIL: {DSA} not found")

    backup = DSA + BACKUP_SUFFIX
    src = open(DSA).read()

    if args.revert:
        if not os.path.exists(backup):
            sys.exit("FAIL: no probe_v2 backup")
        shutil.copyfile(backup, DSA)
        print(f"OK: reverted {DSA}")
        return

    if MARKER in src:
        print("OK: probe v2 already present (no-op)")
        return

    n = src.count(ANCHOR)
    if n != 1:
        sys.exit(f"FAIL: anchor matched {n} times, expected 1.")

    if not os.path.exists(backup):
        shutil.copyfile(DSA, backup)
        print(f"OK: backup -> {backup}")

    open(DSA, "w").write(src.replace(ANCHOR, REPLACEMENT, 1))

    import py_compile
    try:
        py_compile.compile(DSA, doraise=True)
    except Exception as e:
        shutil.copyfile(backup, DSA)
        sys.exit(f"FAIL: broke syntax, reverted. {e}")
    print(f"OK: probe v2 installed in {DSA}")


if __name__ == "__main__":
    main()
