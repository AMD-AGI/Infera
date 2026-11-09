#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 / KERNEL-PLAN section 7, K2 step 1 (between the PP8 scores 64c and 64d, on the batch host):
# aiter's existing TP8 MoE prologue variants (sorting backend / dispatch policy, fused stage-1 quant + scale sort),
# bench/run_prologue.sh. Variant 0 (production) first on GPU 0 (saves the reference outputs), then 1-6 on GPUs 1-6.
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
K=/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit
cleanup() { docker rm -f llying-kopt-bench >/dev/null 2>&1; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
echo "$(date -u +%FT%TZ) start on $(hostname)"
docker rm -f llying-kopt-bench >/dev/null 2>&1
docker run -d --name llying-kopt-bench --network host --ipc host --shm-size 32g --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 -v /perf_apps:/perf_apps -v $K/overlay:/yihou_overlay:ro \
  -v llying-kopt-cache:/root/.cache -v llying-kopt-aiter-jit:/sgl-workspace/aiter/aiter/jit \
  --entrypoint sleep infera-sglang:v0522-fdbafb60 infinity >/dev/null || exit 1
run() { docker exec -e HIP_VISIBLE_DEVICES=$1 -e HIP_FORCE_DEV_KERNARG=1 llying-kopt-bench bash -lc "bash $B/run_prologue.sh $1"; }
run 0
for i in 1 2 3 4 5 6; do run $i & done
wait
echo "$(date -u +%FT%TZ) done"
cat $B/out/prologue_[0-6].log > $B/out/prologue.log
cat $B/out/prologue.log
