#!/usr/bin/env bash
# memlog-ec2: memlog-ec + the same empty_cache() rule on EVERY rank, every 8 decode-loop iterations
# (memlog-ec only ran it at the TP0-only Decode-batch log point; S2k's TP1-3 stayed at 287.9 GiB and aborted).
set -euo pipefail
S=/shared_nfs/yihou_glm52/dcp_pd_20261021/src
SRC=$S/sglang-ab7a898b77-memlog-ec.yihou; DST=$S/sglang-ab7a898b77-memlog-ec2.yihou
[[ -e $DST ]] && { echo "exists: $DST"; exit 1; }
cp -a "$SRC" "$DST"
F=$DST/python/sglang/srt/disaggregation/decode.py
python3 - "$F" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
anchor = ("            self.process_decode_queue()\n\n"
          "            # Get the next batch to run\n"
          "            plan = self.get_next_disagg_decode_batch_to_run(\n")
assert s.count(anchor) == 1, s.count(anchor)
add = ("            self.process_decode_queue()\n"
       "            # yihou gmu: release cached allocator blocks on every rank when device free < 6 GiB.\n"
       "            self._yihou_it = getattr(self, \"_yihou_it\", 0) + 1\n"
       "            if self._yihou_it % 8 == 0 and torch.cuda.mem_get_info()[0] < 6 * 2**30:\n"
       "                torch.cuda.empty_cache()\n\n"
       "            # Get the next batch to run\n"
       "            plan = self.get_next_disagg_decode_batch_to_run(\n")
open(p, 'w').write(s.replace(anchor, add)); print("patched", p)
PY
python3 -c "import ast; ast.parse(open('$F').read())" && echo syntax_ok
