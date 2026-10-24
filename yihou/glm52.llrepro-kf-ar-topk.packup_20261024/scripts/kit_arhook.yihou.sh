#!/usr/bin/env bash
# kit_arhook (task llrepro 2026-10-24, ON 135): DECODE_CUSTOM_AR=1 drops --disable-custom-all-reduce from the DECODE args in
# our copy of llying's kit (user 10-24: custom AR on, performance first, correctness later). Empty = unchanged.
set -euo pipefail
K=/shared_nfs/yihou_glm52/llrepro_20261023/kit/sglang_2p1d_glm5.2
grep -q DECODE_CUSTOM_AR $K/engine/common.sh || { cp -p $K/engine/common.sh /shared_nfs/yihou_glm52/llrepro_20261023/common.sh.before-arhook; cp -p $K/engine/worker.sh /shared_nfs/yihou_glm52/llrepro_20261023/worker.sh.before-arhook; python3 -I - $K <<'PY'
import sys
k = sys.argv[1]
p = k + "/engine/common.sh"; s = open(p).read()
a = "    DECODE_ENV_EXTRA DECODE_MOUNT_EXTRA)"
assert s.count(a) == 1
s = s.replace(a, "    DECODE_ENV_EXTRA DECODE_MOUNT_EXTRA DECODE_CUSTOM_AR)")
open(p, "w").write(s)
p = k + "/engine/worker.sh"; s = open(p).read()
a = 'args+=("${extra[@]}")\n'
assert s.count(a) == 1
s = s.replace(a, a + '''# yihou llrepro: DECODE_CUSTOM_AR=1 removes --disable-custom-all-reduce on the decode (custom all-reduce on).
if [[ "$role" == decode && "${DECODE_CUSTOM_AR:-0}" == 1 ]]; then
    _a=(); for x in "${args[@]}"; do [[ "$x" == --disable-custom-all-reduce ]] || _a+=("$x"); done; args=("${_a[@]}")
fi
''')
open(p, "w").write(s)
PY
}
bash -n $K/engine/common.sh && bash -n $K/engine/worker.sh && grep -n "DECODE_CUSTOM_AR" $K/engine/common.sh $K/engine/worker.sh
