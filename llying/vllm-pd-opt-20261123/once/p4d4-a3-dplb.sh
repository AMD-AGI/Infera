# Phase 1 of ../PLAN: a2 (m10 config + sync-skip overlay + decode prefix caching) + vLLM-internal DP balancing on the
# decode (INFERA_VLLM_DP_ENGINE_LB=1, U28: the router stops choosing the decode rank). Once the decode steps are GPU-bound
# (a1), the MoE all-gather waits ~3 ms per step for the slowest rank, which is what rank balance affects. Routing only, no
# numerical change, so no gate (a2's gate covers the rest). 3600 s score with decode steplog + one trace. -w v3-186 to
# compare with a2 / a1 / a0 on the same node. Run by run_once.sbatch (ONCE=p4d4-a3-dplb).
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/a3
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
mkdir -p "$O"
cp "$0" "$O/once.sh" 2>/dev/null

gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
free=$(timeout 300 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" --entrypoint python3 "$IMAGE" \
    -c 'import torch; print(" ".join(str(int(torch.cuda.mem_get_info(i)[0] / 2**30)) for i in range(torch.cuda.device_count())))' 2>/dev/null | tail -1)
echo "$(date -u +%T) $(hostname -s) GPU free GiB: ${free:-unknown}" | tee "$O/gpu-free-at-start.txt"
for x in $free; do
    if (( x < 250 )); then echo "$(date -u +%T) a3: NOT STARTED, a GPU has only $x GiB free" | tee -a logs/summary.out; exit 0; fi
done

ss=$R/runs/a3-syncskip
rm -rf "$ss"; mkdir -p "$ss" && chmod 777 "$ss"
MLA=$SP/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
docker run --rm --network none -v "$B:$B:ro" -v "$ss:$ss" --entrypoint bash "$IMAGE" -c \
    "md5sum $MLA && python3 $B/scripts/make_syncskip_overlay.py $MLA $ss/rocm_aiter_mla_sparse.py && python3 $B/scripts/test_syncskip.py $ss/rocm_aiter_mla_sparse.py" \
    > "$O/syncskip-build-test.txt" 2>&1 || { echo "$(date -u +%T) a3: overlay build / test failed"; exit 0; }
OV_SS="$ss/rocm_aiter_mla_sparse.py:$MLA:ro"
OV_PIN=$R/overlay-cpupin/vllm/v1/kv_offload/cpu/gpu_worker.py:$SP/vllm/v1/kv_offload/cpu/gpu_worker.py:ro
export D_MODE=v3comm P_GEM=1 P_UTIL=0.88 D_UTIL=0.95 RENDER_TIMEOUT=600 P_RENDER=4 D_RENDER=4 P_MOUNT_X="$OV_PIN"
export D_PREFIX_CACHE=1

stag=a3-dplb-dprefix-syncskip-3600
run=p4d4-$stag
D=$R/runs/$run
mkdir -p "$D/steplog" "$D/instr" && chmod 777 "$D/steplog"
cp "$B/instr/vpd_steplog.py" "$B/instr/vpd_steplog.pth" "$D/instr/"
DM="$OV_SS $D/instr/vpd_steplog.py:$SP/vpd_steplog.py:ro $D/instr/vpd_steplog.pth:$SP/vpd_steplog.pth:ro $D/steplog:/steplog"
( bash scripts/memguard.sh "$run" 120 ) &
CPU_BYTES=1900000000000 DURATION=3600 USAGE_LOG=1 TRACE=1 TRACE_AT=1500 \
    D_ENV_X='VPD_STEPLOG_DIR=/steplog INFERA_VLLM_DP_ENGINE_LB=1' D_MOUNT_X="$DM" \
    bash scripts/snap.sh p4d4_run.sh "$stag"
echo $? > "logs/once-$JOB.rc"
bash scripts/finish_round.sh "logs/once-$JOB.rc" "$run" a3-dplb
python3 "$B/scripts/steplog_summary.py" "$D/steplog" "$D/agentx-c104/benchmark.log" 300 > "$O/steplog_summary.txt" 2>&1
find "$D/trace" -name "dp*.json*" -type f | sort | while read -r f; do
    python3 "$B/scripts/trace_gaps.py" "$f" --step-kernel _combine_sampled_and_draft_tokens_kernel --top 12
    echo
done > "$O/trace-gaps.txt" 2>&1
cp "$B/../vllm-p4d4-crusoe-20261117/results/a3-dplb/"{summary.txt,window_tput.txt,samples_summary.txt,decode_pressure.txt,dp_balance.txt,health.txt,score.txt,ttft_breakdown.txt} "$O/" 2>/dev/null
gzip -k -f "$D"/steplog/steps-*.csv 2>/dev/null
echo "$(date -u +%m%d-%T) a3 analyses in $O" >> logs/summary.out
wait
