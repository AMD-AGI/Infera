#!/usr/bin/env bash
# Build a yihou overlay copy of sglang ab7a898b77 whose "Decode batch" log line also prints device/torch memory.
set -euo pipefail
S=/shared_nfs/yihou_glm52/dcp_pd_20261021/src
SRC=$S/sglang-ab7a898b77; DST=$S/sglang-ab7a898b77-memlog.yihou
[[ -e $DST ]] && { echo "exists: $DST"; exit 1; }
cp -a "$SRC" "$DST"
F=$DST/python/sglang/srt/managers/scheduler_components/metrics_reporter.py
python3 - "$F" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
anchor = '        msg = f"Decode batch{iter_msg}, #running-req: {num_running_reqs}, {token_usage_msg}"\n'
assert s.count(anchor) == 1
add = anchor + (
    '        # yihou gmu diagnostic: device free vs torch allocator state, GiB.\n'
    '        import torch as _t\n'
    '        _free, _tot = _t.cuda.mem_get_info()\n'
    '        msg += (f"mem: dev_free {_free / 2**30:.2f} alloc {_t.cuda.memory_allocated() / 2**30:.2f} "\n'
    '                f"rsv {_t.cuda.memory_reserved() / 2**30:.2f} peak_alloc {_t.cuda.max_memory_allocated() / 2**30:.2f} "\n'
    '                f"peak_rsv {_t.cuda.max_memory_reserved() / 2**30:.2f}, ")\n')
s = s.replace(anchor, add); open(p, 'w').write(s)
print("patched", p)
PY
python3 -c "import ast,sys; ast.parse(open('$F').read())" && echo syntax_ok
find "$DST" -name '__pycache__' -path '*scheduler_components*' -maxdepth 8 | head -2
