#!/usr/bin/env python3
"""Cross-rank check of the decode graph variant audit (debug runner, SGLANG_DEBUG_GRAPH_VARIANT_AUDIT=1).

Each DP rank logs 256-replay blocks: bits of the variant it replayed (new) and of the one the per-rank
rule of the image would have picked (old). Ranks replay in lockstep, so block k / bit i is the same step
on every rank. Reports replays where the ranks disagree (new must be 0; old shows how often the image
mismatched), and the summed invariant counters of the last block: viol (dense with a sequence longer than
index_topk), bound_viol (a sequence longer than the agreed bound), nobound (replays without a bound).

usage: audit_stats.py DECODE.log
"""
import collections
import re
import sys

LINE = re.compile(r"DP(\d+) TP\d+\].*\[variant-audit\] block=(\d+) n=(\d+) dense=(\d+) old_dense=(\d+) viol=(\d+) "
                  r"bound_viol=(\d+) nobound=(\d+) new=([0-9a-f]+) old=([0-9a-f]+)")
blocks = collections.defaultdict(dict)
last = {}
for line in open(sys.argv[1], errors="replace"):
    m = LINE.search(line)
    if not m:
        continue
    dp, blk = int(m.group(1)), int(m.group(2))
    blocks[blk][dp] = (int(m.group(9), 16), int(m.group(10), 16))
    last[dp] = tuple(int(m.group(i)) for i in range(3, 9))
ranks = sorted(last)
if not ranks:
    sys.exit("no [variant-audit] lines")
common = [b for b in sorted(blocks) if len(blocks[b]) == len(ranks)]
new_mis = old_mis = 0
for b in common:
    for i in range(256):
        nb = {(blocks[b][r][0] >> i) & 1 for r in ranks}
        ob = {(blocks[b][r][1] >> i) & 1 for r in ranks}
        new_mis += len(nb) > 1
        old_mis += len(ob) > 1
n = len(common) * 256
print(f"ranks={ranks} aligned_blocks={len(common)} replays={n} new_mismatch={new_mis} old_mismatch={old_mis} "
      f"({old_mis / max(1, n):.1%} of replays)")
for r in ranks:
    nn, dense, old_dense, viol, bviol, nob = last[r]
    print(f"  DP{r}: n={nn} dense={dense} ({dense / max(1, nn):.1%}) old_dense={old_dense} viol={viol} "
          f"bound_viol={bviol} nobound={nob}")
