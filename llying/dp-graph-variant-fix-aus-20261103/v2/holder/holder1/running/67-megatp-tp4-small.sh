#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14: the TP4 score (65b) shows no step gain at busiest-rank bs 16-24 and +2-4 ms at
# bs 4-12, where MegaMoeTP was never measured. aiter's op test (4 GPUs) at the small global token counts of TP4 decode
# (bs 1-16 x 4 verify rows x 4 ranks), random routing, bf16 / fp8 comm. Output: bench/out/megatp_tp4_small.log
O=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench/out/megatp_tp4_small.log
cleanup() { docker rm -f llying-kopt-bench >/dev/null 2>&1; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
echo "$(date -u +%FT%TZ) start on $(hostname)"
docker rm -f llying-kopt-bench >/dev/null 2>&1
docker run -d --name llying-kopt-bench --network host --ipc host --shm-size 32g --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 -v /perf_apps:/perf_apps \
  -v llying-kopt-cache:/root/.cache -v llying-kopt-aiter-jit:/sgl-workspace/aiter/aiter/jit \
  --entrypoint sleep infera-sglang:v0522-fdbafb60 infinity >/dev/null || exit 1
docker exec -e HSA_ENABLE_IPC_MODE_LEGACY=1 -e HIP_VISIBLE_DEVICES=0,1,2,3 llying-kopt-bench bash -lc \
  "cd /sgl-workspace/aiter && timeout 1800 python3 -m torch.distributed.run --standalone --nproc_per_node=4 \
     op_tests/multigpu_tests/test_mega_moe_tp.py --models glm5 --modes ag_rs -t 16 32 64 96 128 192 256 \
     -r random --seeds 0" 2>&1 | grep -v "^\[aiter\]" | grep -E "^\||passed|FAIL|Error" > $O
echo "$(date -u +%FT%TZ) done"
cat $O
