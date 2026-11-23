#!/usr/bin/env python3
"""CPU check of the sync-skip overlay: the fast path must give exactly the synced clamped lengths, or decline.
  test_syncskip.py OVERLAY_FILE        (runs in the vLLM image: numpy, torch)
Exits non-zero on any mismatch, if the overlay does not compile, or if the fast path never fires on long contexts.
"""
import logging
import os
import py_compile
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_syncskip_overlay as m  # noqa: E402

overlay = sys.argv[1]
py_compile.compile(overlay, doraise=True)
src = open(overlay).read()
assert m.NEW in src and "_saturated_clamped_lens" in src, "overlay does not contain the patch"
assert m.NEW_META_SYNC in src and "_META_SYNC = " in src, "overlay does not contain the metadata-sync switch"

ns = {"np": np, "os": os, "logger": logging.getLogger("t")}
exec(m.HELPER, ns)
fast = ns["_saturated_clamped_lens"]
topk, slack = 2048, ns["_SYNC_SKIP_SLACK"]
rng = random.Random(1)
fired = checked = 0
for it in range(20000):
    n_real = rng.randint(1, 64)
    n_pad = rng.choice([0, 0, rng.randint(1, 16)])
    q = rng.choice([1, 4])
    long_ctx = rng.random() < 0.7
    seq = [rng.randint(topk + 64, 1_000_000) if long_ctx else rng.randint(q, 6000) for _ in range(n_real)] + [0] * n_pad
    seg = [q] * n_real + [0] * n_pad
    # Upper bound: exact, or optimistic by up to 3 drafts + a draft-step index of up to 2.
    ub = [s + rng.randint(0, 5) if s else 0 for s in seq]
    seq_np = np.array(seq, dtype=np.int32)
    seg_np = np.array(seg, dtype=np.int32)
    ref_s = np.minimum(seq_np, topk).astype(np.int32)
    ref_c = np.minimum(seq_np - seg_np, topk).astype(np.int32)
    got = fast(torch.tensor(ub, dtype=torch.int32), len(seq), seg_np, topk, slack)
    if got is None:
        continue
    fired += 1
    s, c = got
    if not (np.array_equal(s, ref_s) and np.array_equal(c, ref_c) and s.dtype == ref_s.dtype and c.dtype == ref_c.dtype):
        print("MISMATCH", it, seq[:4], ub[:4], seg[:4], s[:4], ref_s[:4], c[:4], ref_c[:4])
        sys.exit(1)
    if s.tobytes() != ref_s.tobytes() or c.tobytes() != ref_c.tobytes():
        print("KEY BYTES DIFFER", it)
        sys.exit(1)
    checked += 1
print(f"ok: fast path fired on {fired} of 20000 random batches, all {checked} equal to the synced values")
sys.exit(0 if fired > 5000 else 2)
