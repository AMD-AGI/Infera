#!/usr/bin/env bash
# kit_kfhook (task llrepro 2026-10-23, ON 135): add two decode-only hooks to our copy of llying's kit (empty = no change):
# DECODE_ENV_EXTRA ("K=V K=V") and DECODE_MOUNT_EXTRA ("src:dst:ro"), forwarded to the decode node via KIT_VARS.
set -euo pipefail
K=/shared_nfs/yihou_glm52/llrepro_20261023/kit/sglang_2p1d_glm5.2
grep -q DECODE_ENV_EXTRA $K/engine/common.sh || { cp -p $K/engine/common.sh /shared_nfs/yihou_glm52/llrepro_20261023/common.sh.before-kfhook; cp -p $K/engine/worker.sh /shared_nfs/yihou_glm52/llrepro_20261023/worker.sh.before-kfhook; python3 -I - $K <<'PY'
import sys
k = sys.argv[1]
p = k + "/engine/common.sh"; s = open(p).read()
a = "    KV_EVENTS_PORT KV_SNAPSHOT_PORT ETCD_IMAGE CTR)\n"
assert s.count(a) == 1
s = s.replace(a, "    KV_EVENTS_PORT KV_SNAPSHOT_PORT ETCD_IMAGE CTR\n    DECODE_ENV_EXTRA DECODE_MOUNT_EXTRA)   # yihou llrepro: decode-only hooks (KF kernels)\n")
open(p, "w").write(s)
p = k + "/engine/worker.sh"; s = open(p).read()
a = 'for kv in "${env[@]}"; do env_args+=(-e "$kv"); done\n'
assert s.count(a) == 1
s = s.replace(a, a + '# yihou llrepro: decode-only extra env (DECODE_ENV_EXTRA, space separated K=V).\n[[ "$role" != decode ]] || for kv in ${DECODE_ENV_EXTRA:-}; do env_args+=(-e "$kv"); done\n')
a = '[[ "$role" != prefill ]] || mounts+=(-v "$KIT/patches/deepseek_v2.py:/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py:ro")\n'
assert s.count(a) == 1
s = s.replace(a, a + '# yihou llrepro: decode-only extra mount (DECODE_MOUNT_EXTRA = src:dst:ro).\n[[ "$role" != decode || -z "${DECODE_MOUNT_EXTRA:-}" ]] || mounts+=(-v "$DECODE_MOUNT_EXTRA")\n')
open(p, "w").write(s)
PY
}
bash -n $K/engine/common.sh && bash -n $K/engine/worker.sh && grep -n "yihou llrepro" $K/engine/common.sh $K/engine/worker.sh
