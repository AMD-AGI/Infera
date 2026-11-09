#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 14 (after the PP8 baseline score 64d, on the batch host, all 8 GPUs): MegaMoeTP vs the
# production TP8 MoE path on real GLM-5.2 weights (layers 3 and 40, sglang's weight shuffle, real router top-k),
# bench/megatp_realw_ab.py: SNR against production (also checks the weight layout) and CUDA-graph us.
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
O=$B/out/megatp_realw_ab.log
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
: > $O
for args in "--layer 3 --comm bf16" "--layer 40 --comm bf16" "--layer 3 --comm fp8"; do
  echo "== $args" >> $O
  docker exec -e HSA_ENABLE_IPC_MODE_LEGACY=1 llying-kopt-bench bash -lc \
    "cd $B && timeout 1800 python3 -m torch.distributed.run --standalone --nproc_per_node=8 megatp_realw_ab.py \
       --tokens 576,640,704,768,832 $args" 2>&1 | grep -v "^\[aiter\]" | grep -E "RESULT|layer |Error|error|failed|Traceback" >> $O
done
echo "$(date -u +%FT%TZ) done"
cat $O
