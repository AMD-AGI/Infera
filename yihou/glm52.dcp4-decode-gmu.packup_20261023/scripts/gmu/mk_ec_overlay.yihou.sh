#!/usr/bin/env bash
# Copy of the memlog overlay that also calls torch.cuda.empty_cache() at the Decode-batch log point when device free < 6 GiB.
set -euo pipefail
S=/shared_nfs/yihou_glm52/dcp_pd_20261021/src
SRC=$S/sglang-ab7a898b77-memlog.yihou; DST=$S/sglang-ab7a898b77-memlog-ec.yihou
[[ -e $DST ]] && { echo "exists: $DST"; exit 1; }
cp -a "$SRC" "$DST"
F=$DST/python/sglang/srt/managers/scheduler_components/metrics_reporter.py
python3 - "$F" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
anchor = '        _free, _tot = _t.cuda.mem_get_info()\n'
assert s.count(anchor) == 1
add = anchor + ('        if _free < 6 * 2**30:\n'
                '            _t.cuda.empty_cache()\n'
                '            msg += "empty_cache, "\n'
                '            _free, _tot = _t.cuda.mem_get_info()\n')
open(p, 'w').write(s.replace(anchor, add)); print("patched", p)
PY
python3 -c "import ast; ast.parse(open('$F').read())" && echo syntax_ok
