set -euo pipefail
F=/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-ab7a898b77-memlog-ec2.yihou/python/sglang/srt/disaggregation/decode.py
python3 - "$F" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
anchor = ("            self.process_decode_queue()\n\n"
          "            # Get the next batch to run\n"
          "            plan = self.get_next_disagg_decode_batch_to_run(\n")
assert "yihou gmu" not in s and s.count(anchor) == 2
add = ("            self.process_decode_queue()\n"
       "            # yihou gmu: release cached allocator blocks on every rank when device free < 6 GiB.\n"
       "            self._yihou_it = getattr(self, \"_yihou_it\", 0) + 1\n"
       "            if self._yihou_it % 8 == 0 and torch.cuda.mem_get_info()[0] < 6 * 2**30:\n"
       "                torch.cuda.empty_cache()\n\n"
       "            # Get the next batch to run\n"
       "            plan = self.get_next_disagg_decode_batch_to_run(\n")
open(p, 'w').write(s.replace(anchor, add)); print("patched both loops", p)
PY
python3 -c "import ast; ast.parse(open('$F').read())" && echo syntax_ok; grep -c "yihou gmu" $F
