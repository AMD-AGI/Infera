#!/usr/bin/env bash
# PR #42553 GPU validation on ONE GPU: (1) PR unit test, (2) 150 GB GTT page_first MLA smoke.
# usage: run_pr42553.yihou.sh <node> <job|-> <gpu> [pool_GB=150] [image]
set -uo pipefail
node=${1:?node}; job=${2-}; gpu=${3:?gpu}; GBS=${4:-150}; IMG=${5:-infera-sglang:v0521-yihou-1009}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010; P=$D/diag/pr42553
MNT=$D/harness/mounts-v0521/src/pool_host_common.yihou-pr42553.py
TGT=/sgl-workspace/sglang/python/sglang/srt/mem_cache/pool_host/common.py
ts=$(date -u +%Y%m%d-%H%M%S); out=$P/run.$ts.yihou; mkdir -p "$out"
if [[ -z "$job" || "$job" == "-" ]]; then SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes); R=/tmp/yihou_v0521dbg/pr42553/$ts
else export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS="$node:$job"; SSH=(ssh); R=/mnt/m2m_nobackup/yihou_v0521dbg/pr42553/$ts; fi
BASE="--rm --device /dev/kfd --device /dev/dri --group-add video --network none --ipc host --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -e HIP_VISIBLE_DEVICES=$gpu -e PYTHONUNBUFFERED=1 -v $D:$D:ro -v $MNT:$TGT:ro -v $R:$R"
echo "[pr42553] $node gpu=$gpu image=$IMG out=$out (node-local $R)"
"${SSH[@]}" "$node" "mkdir -p $R && chmod 777 $R && \
  docker run $BASE --name yihou-pr42553-ut-$ts --entrypoint bash $IMG -c 'cd /sgl-workspace/sglang && (git apply --include=\"test/*\" $P/pr42553-8c27ef48.yihou.diff || patch -p1 --forward -i $P/pr42553-8c27ef48.yihou.diff) && grep -c _hip_non_userptr test/registered/unit/mem_cache/test_hicache_host_register.py && python3 -m pytest -q -p no:cacheprovider test/registered/unit/mem_cache/test_hicache_host_register.py' > $R/unittest.log 2>&1; echo UT_EXIT=\$? >> $R/unittest.log; \
  t0=\$(date +%s.%N); docker run $BASE --name yihou-pr42553-smoke-$ts -e HSA_USERPTR_FOR_PAGED_MEM=0 --entrypoint python3 $IMG $P/gtt_smoke.yihou.py $GBS > $R/smoke.log 2>&1; echo SMOKE_EXIT=\$? >> $R/smoke.log; \
  echo \"SMOKE_WALL_INCL_EXIT_S=\$(awk -v a=\$t0 -v b=\$(date +%s.%N) 'BEGIN{printf \"%.1f\", b-a}')\" >> $R/smoke.log"
"${SSH[@]}" "$node" "tar -C $R -cf - ." | tar -C "$out" -xf -
tail -4 "$out/unittest.log"; grep -aE "^(HSA|patched|dims|ALLOC|D2H|H2D|FREE|RESULT|SMOKE)|MISMATCH|Error|error" "$out/smoke.log"
echo "[pr42553] logs: $out"
