#!/usr/bin/env bash
# kopt 73b3 (no engine): rerun of 73b2 after fixing the import-hook recursion with KOPT_MEGATP (register through its hook) and
# the fallback check (within rounding: split-K GEMMs need not be bit-reproducible). QB off and on. 1 GPU.
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
O=$B/out/kopt73b3.log
C=llying-kopt-bench
cleanup() { docker rm -f $C >/dev/null 2>&1; kill "$g" 2>/dev/null; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
echo "$(date -u +%FT%TZ) start on $(hostname)" > $O
srun --jobid="$SLURM_JOB_ID" --overlap -N1 -n1 -w "$(hostname -s)" --export=NONE \
    /bin/bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/guard_containers.sh llying-kopt- \
    >> /perf_apps/liyingli/vfix-logs/opt/guard.log 2>&1 &
g=$!
for i in $(seq 30); do
    rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ {if ($NF + 0 > 50) b = 1} END {exit b}' && break
    echo "$(date -u +%T) waiting for VRAM" >> $O; sleep 60
done
docker rm -f $C >/dev/null 2>&1
docker run -d --name $C --network host --ipc host --shm-size 32g --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 -v /perf_apps:/perf_apps -v /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/overlay-kopt:/yihou_overlay:ro -v /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/overlay-gemmmerge/sitecustomize.py:/yihou_overlay/sitecustomize.py:ro -v /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/overlay-gemmmerge:/kopt_overlay_extra:ro \
  -v /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/aiter_csv/yihou_glm52_dpa_bf16_tuned_gemm.csv:/sgl-workspace/aiter/aiter/configs/model_configs/yihou_glm52_dpa_bf16_tuned_gemm.csv:ro -v /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/aiter_csv/yihou_glm52_ptpc_bf16_tuned_gemm.csv:/sgl-workspace/aiter/aiter/configs/model_configs/yihou_glm52_ptpc_bf16_tuned_gemm.csv:ro -v llying-kopt-cache:/root/.cache -v llying-kopt-aiter-jit:/sgl-workspace/aiter/aiter/jit \
  --entrypoint sleep infera-sglang:v0522-fdbafb60 infinity >/dev/null || exit 1
docker exec -e HIP_VISIBLE_DEVICES=0 -e PYTHONPATH=/yihou_overlay -e SGLANG_USE_AITER=1 -e KOPT_GEMM_MERGE=1 -e KOPT_MEGATP=1 $C bash -lc \
  "python3 -c \"import sglang.srt.models.deepseek_v2 as m; print('sitecustomize hooks:', m.DeepseekV2AttentionMLA.prepare_qkv_latent.__module__, m.DeepseekV2DecoderLayer.forward.__module__)\"" \
  2>&1 | grep -v "^\[aiter\]" | tail -5 >> $O
for qb in 0 1; do
  echo "== KOPT_GEMM_MERGE_QB=$qb" >> $O
  docker exec -e HIP_VISIBLE_DEVICES=0 -e SGLANG_USE_AITER=1 -e KOPT_GEMM_MERGE=1 -e KOPT_GEMM_MERGE_QB=$qb $C bash -lc \
    "cd $B && timeout 600 python3 gemm_merge_patch_test.py" 2>&1 | grep -v "^\[aiter\]" | tail -40 >> $O
done
echo "$(date -u +%FT%TZ) done" >> $O
tail -30 $O
