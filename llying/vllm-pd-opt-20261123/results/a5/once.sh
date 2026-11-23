# a5: KV-aware decode DP balancing (scripts/make_dplb_kv_overlay.py) on top of the best config so far: a3 (sync skip +
# decode prefix caching + INFERA_VLLM_DP_ENGINE_LB=1) [+ a4's BF16 GEMM table]. vLLM's balancer scores waiting + running;
# the overlay adds VPD_DP_LB_KV_WEIGHT * kv_cache_usage (a3 step fit: a full rank's KV ~ 15.6 requests of step time).
# Routing only, so no gate (like a3). once/a5-extra.env is read right before the score: A5_KV_W (default 14),
# A5_GEMM (1 = mount a4's table if it exists), A5_D_ENV_X (extra decode env). -w v3-186. ONCE=p4d4-a5-dplbkv.
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/a5
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
mkdir -p "$O"
cp "$0" "$O/once.sh" 2>/dev/null

gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
free=$(timeout 300 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" --entrypoint python3 "$IMAGE" \
    -c 'import torch; print(" ".join(str(int(torch.cuda.mem_get_info(i)[0] / 2**30)) for i in range(torch.cuda.device_count())))' 2>/dev/null | tail -1)
echo "$(date -u +%T) $(hostname -s) GPU free GiB: ${free:-unknown}" | tee "$O/gpu-free-at-start.txt"
for x in $free; do
    if (( x < 250 )); then echo "$(date -u +%T) a5: NOT STARTED, a GPU has only $x GiB free" | tee -a logs/summary.out; exit 0; fi
done

ov=$R/runs/a5-overlays
rm -rf "$ov"; mkdir -p "$ov" && chmod 777 "$ov"
MLA=$SP/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
CC=$SP/vllm/v1/engine/core_client.py
docker run --rm --network none -v "$B:$B:ro" -v "$ov:$ov" --entrypoint bash "$IMAGE" -c \
    "md5sum $MLA $CC && python3 $B/scripts/make_syncskip_overlay.py $MLA $ov/rocm_aiter_mla_sparse.py && python3 $B/scripts/test_syncskip.py $ov/rocm_aiter_mla_sparse.py \
     && python3 $B/scripts/make_dplb_kv_overlay.py $CC $ov/core_client.py && VPD_DP_LB_KV_WEIGHT=14 python3 $B/scripts/test_dplb_kv.py $ov/core_client.py" \
    > "$O/overlay-build-test.txt" 2>&1 || { echo "$(date -u +%T) a5: overlay build / test failed"; exit 0; }
OV="$ov/rocm_aiter_mla_sparse.py:$MLA:ro $ov/core_client.py:$CC:ro"

OV_PIN=$R/overlay-cpupin/vllm/v1/kv_offload/cpu/gpu_worker.py:$SP/vllm/v1/kv_offload/cpu/gpu_worker.py:ro
export D_MODE=v3comm P_GEM=1 P_UTIL=0.88 D_UTIL=0.95 RENDER_TIMEOUT=600 P_RENDER=4 D_RENDER=4 P_MOUNT_X="$OV_PIN"
export D_PREFIX_CACHE=1
for i in $(seq 60); do
    shm=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo); (( shm < 100 )) && break; sleep 10
done

A5_KV_W=14 A5_GEMM=1 A5_D_ENV_X=""
[ -r "$R/once/a5-extra.env" ] && . "$R/once/a5-extra.env"
gemm=$R/runs/a4-gemmtune/glm52_decode_bf16_tuned_gemm.csv
if [[ "$A5_GEMM" == 1 && -s "$gemm" ]]; then
    cp "$gemm" "$ov/"; OV+=" $ov/glm52_decode_bf16_tuned_gemm.csv:$SP/aiter/configs/model_configs/glm52_bf16_tuned_gemm.csv:ro"
fi
echo "$(date -u +%T) a5: KV weight $A5_KV_W, gemm table $([[ "$OV" == *glm52_bf16_tuned_gemm* ]] && echo on || echo off), extra decode env [$A5_D_ENV_X]" | tee "$O/config.txt"

stag=a5-dplbkv${A5_KV_W}-dprefix-syncskip-3600
run=p4d4-$stag
D=$R/runs/$run
mkdir -p "$D/steplog" "$D/instr" && chmod 777 "$D/steplog"
cp "$B/instr/vpd_steplog.py" "$B/instr/vpd_steplog.pth" "$D/instr/"
DM="$OV $D/instr/vpd_steplog.py:$SP/vpd_steplog.py:ro $D/instr/vpd_steplog.pth:$SP/vpd_steplog.pth:ro $D/steplog:/steplog"
( bash scripts/memguard.sh "$run" 120 ) &
CPU_BYTES=1900000000000 DURATION=3600 USAGE_LOG=1 TRACE=1 TRACE_AT=1500 \
    D_ENV_X="VPD_STEPLOG_DIR=/steplog INFERA_VLLM_DP_ENGINE_LB=1 VPD_DP_LB_KV_WEIGHT=$A5_KV_W $A5_D_ENV_X" D_MOUNT_X="$DM" \
    bash scripts/snap.sh p4d4_run.sh "$stag"
echo $? > "logs/once-$JOB.rc"
bash scripts/finish_round.sh "logs/once-$JOB.rc" "$run" a5-dplbkv
zcat -f "$D"/vpd-decode*.log* 2>/dev/null | grep -a "\[vpd\] DP LB KV weight" | head -3 > "$O/lb-kv-active.txt"
zcat -f "$D"/vpd-decode*.log* 2>/dev/null | grep -a -c "not found tuned config" > "$O/score-untuned-lines.txt"
python3 "$B/scripts/steplog_summary.py" "$D/steplog" "$D/agentx-c104/benchmark.log" 300 > "$O/steplog_summary.txt" 2>&1
find "$D/trace" -name "dp*.json*" -type f | sort | while read -r f; do
    python3 "$B/scripts/trace_gaps.py" "$f" --step-kernel _combine_sampled_and_draft_tokens_kernel --top 12
    echo
done > "$O/trace-gaps.txt" 2>&1
f1=$(find "$D/trace" -name "dp1_*.json*" -type f | head -1)
[ -n "$f1" ] && python3 scripts/trace_cat.py "$f1" --top 40 > "$O/trace-cat-dp1.txt" 2>&1
cp "$B/../vllm-p4d4-crusoe-20261117/results/a5-dplbkv/"{summary.txt,window_tput.txt,samples_summary.txt,decode_pressure.txt,dp_balance.txt,health.txt,score.txt,ttft_breakdown.txt} "$O/" 2>/dev/null
gzip -k -f "$D"/steplog/steps-*.csv 2>/dev/null
echo "$(date -u +%m%d-%T) a5 analyses in $O" >> logs/summary.out
wait
