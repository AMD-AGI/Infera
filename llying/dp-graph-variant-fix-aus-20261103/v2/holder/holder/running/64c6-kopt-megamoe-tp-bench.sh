#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 / KERNEL-PLAN section 7 (between the PP8 scores 64c and 64d, on the batch host, all 8 GPUs):
# aiter's fused TP MoE layer (MegaMoeTP: all-gather + prologue + GEMM1 + act + GEMM2 + reduce-scatter in one persistent
# kernel) on the GLM-5.2 shape (257 experts incl. the fused shared one, top-9, inter 2048 / 8), ag_rs, at the TP8
# decode global token counts. aiter's own op test (accuracy vs torch + CUDA-graph us of the slowest rank, fp8 and bf16
# comm). Production per MoE layer at M ~640 (PP8 trace): all-gather 38.4 + MoE ~175 + reduce-scatter 32.2 us.
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
K=/perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit
O=$B/out/megamoe_tp.log
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
docker exec -e HSA_ENABLE_IPC_MODE_LEGACY=1 llying-kopt-bench bash -lc \
  "cd /sgl-workspace/aiter && timeout 3000 python3 -m torch.distributed.run --standalone --nproc_per_node=8 \
     op_tests/multigpu_tests/test_mega_moe_tp.py --models glm5 --modes ag_rs -t 512 576 640 704 768 832 \
     -r random hot --seeds 0" > $O 2>&1
rc=$?
echo "$(date -u +%FT%TZ) done rc=$rc"
grep -v "^\[aiter\]" $O | tail -80
