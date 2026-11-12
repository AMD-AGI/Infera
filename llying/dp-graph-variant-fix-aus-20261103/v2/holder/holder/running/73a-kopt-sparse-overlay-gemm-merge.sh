#!/usr/bin/env bash
# kopt 73a (microbench only): (1) the sparse MLA overlay variant file bit-compared and timed vs production; (2) which sparse
# kernels draft-like calls run and whether they have the 2 GiB cliff; (3) same-input GEMMs separate vs merged, production tables then tuned.
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
O=$B/out/kopt73a.log
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
  --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 -v /perf_apps:/perf_apps -v /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/overlay-kopt:/yihou_overlay:ro \
  -v /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/aiter_csv/yihou_glm52_dpa_bf16_tuned_gemm.csv:/sgl-workspace/aiter/aiter/configs/model_configs/yihou_glm52_dpa_bf16_tuned_gemm.csv:ro -v /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/aiter_csv/yihou_glm52_ptpc_bf16_tuned_gemm.csv:/sgl-workspace/aiter/aiter/configs/model_configs/yihou_glm52_ptpc_bf16_tuned_gemm.csv:ro -v llying-kopt-cache:/root/.cache -v llying-kopt-aiter-jit:/sgl-workspace/aiter/aiter/jit \
  --entrypoint sleep infera-sglang:v0522-fdbafb60 infinity >/dev/null || exit 1
T=/perf_apps/liyingli/bench_agentx/kopt-20261109/tune
cp /perf_apps/liyingli/bench_agentx/opt-aus-20261103/pkg/kit/overlay-variants/kf_sparse_mla_split2g.py $B/kf_sparse_mla_overlayv.py
docker exec -e HIP_VISIBLE_DEVICES=0 -e HIP_FORCE_DEV_KERNARG=1 -e DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 $C bash -lc \
  "cd $B && python3 make_kf_sparse_split2g.py /yihou_overlay/yihou_kf/kf_sparse_mla.py kf_sparse_mla_split2g.py && timeout 900 python3 sparse_split2g.py overlayv" \
  2>&1 | grep -v "^\[aiter\]" > $B/out/sparse_overlayv.log &
p0=$!
docker exec -e HIP_VISIBLE_DEVICES=1 -e HIP_FORCE_DEV_KERNARG=1 -e DEBUG_CLR_GRAPH_PACKET_CAPTURE=0 $C bash -lc \
  "cd $B && timeout 900 python3 sparse_draft_probe.py" 2>&1 | grep -v "^\[aiter\]" > $B/out/sparse_draft_probe.log &
p1=$!
docker exec -e HIP_VISIBLE_DEVICES=2 $C bash -lc "cd $B && timeout 900 python3 gemm_merge_bench.py" 2>&1 | grep -v "^\[aiter\]" > $B/out/gemm_merge_prod.log &
p2=$!
wait $p0 $p1 $p2
cat $B/out/sparse_overlayv.log $B/out/sparse_draft_probe.log $B/out/gemm_merge_prod.log >> $O
docker exec $C bash -lc "cd $B && python3 gemm_merge_bench.py --untuned $T/untuned_merge.csv" >> $O 2>&1
rm -f $T/tuned_merge.csv $T/profile_merge.csv
docker exec $C bash -lc "cd /sgl-workspace/aiter && timeout 1800 python3 csrc/gemm_a16w16/gemm_tuner.py -i $T/untuned_merge.csv -o $T/tuned_merge.csv -o2 $T/profile_merge.csv --mp 8" \
  > $T/tune_merge.log 2>&1
echo "$(date -u +%T) tuner rc=$?" >> $O
docker exec -e HIP_VISIBLE_DEVICES=0 -e AITER_CONFIG_GEMM_BF16=$T/tuned_merge.csv $C bash -lc \
  "cd $B && timeout 900 python3 gemm_merge_bench.py" 2>&1 | grep -v "^\[aiter\]" > $B/out/gemm_merge_tuned.log
cat $B/out/gemm_merge_tuned.log >> $O
echo "$(date -u +%FT%TZ) done" >> $O
tail -30 $O
