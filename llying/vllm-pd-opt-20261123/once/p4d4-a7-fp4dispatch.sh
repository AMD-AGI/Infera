# a7: FP4 MoE dispatch on the decode (scripts/make_fp4dispatch_overlay.py, VPD_FP4_DISPATCH=1): each DP rank quantizes
# its own rows to MXFP4 before the MoE all-gather (a quarter of the bytes), instead of aiter quantizing the gathered
# BF16 rows. Changes what the MoE kernels receive, so: GPU unit check, gate (real acceptance), then a 3600 s score.
# Base config from $R/once/a7-extra.env, read before the gate: A7_KV_W (KV-aware LB weight, 0 = vLLM's own LB),
# A7_GEMM (1 = a4's table), A7_D_ARGS (extra decode args), A7_D_ENV (extra decode env). -w v3-186. ONCE=p4d4-a7-fp4dispatch.
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/a7
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
mkdir -p "$O"
cp "$0" "$O/once.sh" 2>/dev/null

gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
free=$(timeout 300 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" --entrypoint python3 "$IMAGE" \
    -c 'import torch; print(" ".join(str(int(torch.cuda.mem_get_info(i)[0] / 2**30)) for i in range(torch.cuda.device_count())))' 2>/dev/null | tail -1)
echo "$(date -u +%T) $(hostname -s) GPU free GiB: ${free:-unknown}" | tee "$O/gpu-free-at-start.txt"
for x in $free; do
    if (( x < 250 )); then echo "$(date -u +%T) a7: NOT STARTED, a GPU has only $x GiB free" | tee -a logs/summary.out; exit 0; fi
done

ov=$R/runs/a7-overlays
rm -rf "$ov"; mkdir -p "$ov" && chmod 777 "$ov"
MLA=$SP/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
CC=$SP/vllm/v1/engine/core_client.py
NDP=$SP/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py
RAM=$SP/vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py
docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES=0 -v "$B:$B:ro" -v "$ov:$ov" --entrypoint bash "$IMAGE" -c \
    "md5sum $MLA $CC $NDP $RAM && python3 $B/scripts/make_syncskip_overlay.py $MLA $ov/rocm_aiter_mla_sparse.py && python3 $B/scripts/test_syncskip.py $ov/rocm_aiter_mla_sparse.py \
     && python3 $B/scripts/make_dplb_kv_overlay.py $CC $ov/core_client.py && VPD_DP_LB_KV_WEIGHT=14 python3 $B/scripts/test_dplb_kv.py $ov/core_client.py \
     && python3 $B/scripts/make_fp4dispatch_overlay.py $NDP $ov/naive_dp_ep.py $RAM $ov/rocm_aiter_moe.py && python3 $B/scripts/test_fp4dispatch.py $ov/naive_dp_ep.py $ov/rocm_aiter_moe.py" \
    > "$O/overlay-build-test.txt" 2>&1 || { echo "$(date -u +%T) a7: overlay build / test failed"; exit 0; }

OV_PIN=$R/overlay-cpupin/vllm/v1/kv_offload/cpu/gpu_worker.py:$SP/vllm/v1/kv_offload/cpu/gpu_worker.py:ro
export D_MODE=v3comm P_GEM=1 P_UTIL=0.88 D_UTIL=0.95 RENDER_TIMEOUT=600 P_RENDER=4 D_RENDER=4 P_MOUNT_X="$OV_PIN"
export D_PREFIX_CACHE=1

A7_KV_W=0 A7_GEMM=1 A7_D_ARGS="" A7_D_ENV=""
[ -r "$R/once/a7-extra.env" ] && . "$R/once/a7-extra.env" && cp "$R/once/a7-extra.env" "$O/"
OV="$ov/rocm_aiter_mla_sparse.py:$MLA:ro $ov/naive_dp_ep.py:$NDP:ro $ov/rocm_aiter_moe.py:$RAM:ro"
denv="VPD_FP4_DISPATCH=1 INFERA_VLLM_DP_ENGINE_LB=1"
if [[ "$A7_KV_W" != 0 ]]; then OV+=" $ov/core_client.py:$CC:ro"; denv+=" VPD_DP_LB_KV_WEIGHT=$A7_KV_W"; fi
gemm=$R/runs/a4-gemmtune/glm52_decode_bf16_tuned_gemm.csv
if [[ "$A7_GEMM" == 1 && -s "$gemm" ]]; then
    cp "$gemm" "$ov/"; OV+=" $ov/glm52_decode_bf16_tuned_gemm.csv:$SP/aiter/configs/model_configs/glm52_bf16_tuned_gemm.csv:ro"
fi
denv+=" $A7_D_ENV"
echo "$(date -u +%T) a7: KV weight $A7_KV_W; gemm table $([[ "$OV" == *glm52_bf16_tuned_gemm* ]] && echo on || echo off); decode args [$A7_D_ARGS]; decode env [$denv]" | tee "$O/config.txt"

# 1) Gate, real acceptance.
gtag=a7-gate-fp4dispatch
CPU_BYTES=200000000000 D_MOUNT_X="$OV" D_ENV_X="$denv" D_ARGS_X="$A7_D_ARGS" P4D4_STEP=gate_p4d4.sh \
    bash scripts/snap.sh p4d4_run.sh "$gtag" > "$O/gate-run.log" 2>&1
line=$(grep -a "gate p4d4-$gtag:" logs/summary.out | tail -1)
echo "$line" > "$O/gate.txt"
cp results/gate-p4d4-$gtag/{smoke.out,probe-stats.out,longctx.out,metrics-delta.json,probe-review.txt} "$O/" 2>/dev/null
zcat -f "$R/runs/p4d4-$gtag"/vpd-decode*.log* 2>/dev/null | grep -a -E "Error|error|illegal|fault" | grep -a -v -E "GET /metrics|errors=0" | head -20 > "$O/gate-decode-errors.txt"
ok=$(python3 - "$line" <<'EOF'
import re, sys
s = sys.argv[1]
def g(p, d=-1):
    m = re.search(p, s)
    return float(m.group(1)) if m else d
smoke = re.search(r"smoke (\d+)/(\d+)", s)
good = (smoke is not None and smoke.group(1) == smoke.group(2) and g(r"correct=(\d+)/200") >= 190
        and g(r"garbled=(\d+)", 99) == 0 and g(r"bad_start=(\d+)", 99) <= 1 and g(r'"accept_len": ([0-9.]+)') >= 3.0)
print("yes" if good else "no")
EOF
)
echo "$(date -u +%T) a7 gate clean: $ok" | tee -a "$O/gate.txt"
[[ "$ok" == yes ]] || exit 0
for i in $(seq 60); do
    shm=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo); (( shm < 100 )) && break; sleep 10
done

# 2) Score.
stag=a7-fp4dispatch-gemm-dplb-dprefix-syncskip-3600
run=p4d4-$stag
D=$R/runs/$run
mkdir -p "$D/steplog" "$D/instr" && chmod 777 "$D/steplog"
cp "$B/instr/vpd_steplog.py" "$B/instr/vpd_steplog.pth" "$D/instr/"
DM="$OV $D/instr/vpd_steplog.py:$SP/vpd_steplog.py:ro $D/instr/vpd_steplog.pth:$SP/vpd_steplog.pth:ro $D/steplog:/steplog"
( bash scripts/memguard.sh "$run" 120 ) &
CPU_BYTES=1900000000000 DURATION=3600 USAGE_LOG=1 TRACE=1 TRACE_AT=1500 \
    D_ENV_X="VPD_STEPLOG_DIR=/steplog $denv" D_MOUNT_X="$DM" D_ARGS_X="$A7_D_ARGS" \
    bash scripts/snap.sh p4d4_run.sh "$stag"
echo $? > "logs/once-$JOB.rc"
bash scripts/finish_round.sh "logs/once-$JOB.rc" "$run" a7-fp4dispatch
python3 "$B/scripts/steplog_summary.py" "$D/steplog" "$D/agentx-c104/benchmark.log" 300 > "$O/steplog_summary.txt" 2>&1
python3 "$B/scripts/dp_imbalance.py" "$D/steplog" > "$O/dp_imbalance.txt" 2>&1
find "$D/trace" -name "dp*.json*" -type f | sort | while read -r f; do
    python3 "$B/scripts/trace_gaps.py" "$f" --step-kernel _combine_sampled_and_draft_tokens_kernel --top 12
    echo
done > "$O/trace-gaps.txt" 2>&1
f1=$(find "$D/trace" -name "dp1_*.json*" -type f | head -1)
[ -n "$f1" ] && python3 scripts/trace_cat.py "$f1" --top 40 > "$O/trace-cat-dp1.txt" 2>&1
[ -n "$f1" ] && python3 "$B/scripts/kernel_dist.py" "$f1" 'allgather_vecIDF16b' 'allgather_vec<float' 'reduce_scatter_split_first_dim' 'dynamic_per_group_scaled_quant' 'mxfp4_moe_sort' > "$O/kernel-dist-dp1.txt" 2>&1
cp "$B/../vllm-p4d4-crusoe-20261117/results/a7-fp4dispatch/"{summary.txt,window_tput.txt,samples_summary.txt,decode_pressure.txt,dp_balance.txt,health.txt,score.txt,ttft_breakdown.txt} "$O/" 2>/dev/null
gzip -k -f "$D"/steplog/steps-*.csv 2>/dev/null
echo "$(date -u +%m%d-%T) a7 analyses in $O" >> logs/summary.out
wait
