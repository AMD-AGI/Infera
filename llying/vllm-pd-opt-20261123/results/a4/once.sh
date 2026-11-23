# Phase 2 (decode kernels) of ../PLAN: aiter BF16 GEMM table for the GLM-5.2 decode. The vLLM image's aiter has BF16
# tables for DeepSeek-R1 / V3 / V4 (hidden 7168) but none for GLM-5.2 (hidden 6144): a2's decode log lists 256 untuned
# decode shapes (M <= 512) of o_proj, lm_head, fused_qkv_a, indexer wq_b, the dense-MLP gate_up and a 4096 x 6144 GEMM.
# 1) tune them with the image's own tuner (8 GPUs); 2) gate with real acceptance; 3) 3600 s score with the a2 config +
# the table (+ once/a4-extra.env, sourced right before the score, e.g. the decode DP balancing if a3 shows a gain).
# -w v3-186 to compare with a2. Run by run_once.sbatch (ONCE=p4d4-a4-gemmtune).
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/a4
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
mkdir -p "$O"
cp "$0" "$O/once.sh" 2>/dev/null

gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
free=$(timeout 300 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" --entrypoint python3 "$IMAGE" \
    -c 'import torch; print(" ".join(str(int(torch.cuda.mem_get_info(i)[0] / 2**30)) for i in range(torch.cuda.device_count())))' 2>/dev/null | tail -1)
echo "$(date -u +%T) $(hostname -s) GPU free GiB: ${free:-unknown}" | tee "$O/gpu-free-at-start.txt"
for x in $free; do
    if (( x < 250 )); then echo "$(date -u +%T) a4: NOT STARTED, a GPU has only $x GiB free" | tee -a logs/summary.out; exit 0; fi
done

ss=$R/runs/a4-syncskip
rm -rf "$ss"; mkdir -p "$ss" && chmod 777 "$ss"
MLA=$SP/vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
docker run --rm --network none -v "$B:$B:ro" -v "$ss:$ss" --entrypoint bash "$IMAGE" -c \
    "md5sum $MLA && python3 $B/scripts/make_syncskip_overlay.py $MLA $ss/rocm_aiter_mla_sparse.py && python3 $B/scripts/test_syncskip.py $ss/rocm_aiter_mla_sparse.py" \
    > "$O/syncskip-build-test.txt" 2>&1 || { echo "$(date -u +%T) a4: overlay build / test failed"; exit 0; }
OV_SS="$ss/rocm_aiter_mla_sparse.py:$MLA:ro"

# 1) Tune.
tw=$R/runs/a4-gemmtune
rm -rf "$tw"
echo "$(date -u +%T) tuning starts" | tee "$O/tune-progress.txt"
bash "$B/scripts/tune_decode_gemm.sh" "$B/gemm-tune/glm52_decode_shapes_a2.txt" "$tw" 0,1,2,3,4,5,6,7 4500
trc=$?
cp "$tw"/{glm52_decode_bf16_tuned_gemm.csv,tune-summary.txt,help.txt,make-untuned.log,untuned.csv} "$O/" 2>/dev/null
tail -60 "$tw/tune.log" > "$O/tune-log-tail.txt" 2>/dev/null
echo "$(date -u +%T) tuning done rc=$trc" | tee -a "$O/tune-progress.txt"
(( trc == 0 )) || { echo "$(date -u +%T) a4: tuning produced no table" | tee -a logs/summary.out; exit 0; }
OV_GEMM="$tw/glm52_decode_bf16_tuned_gemm.csv:$SP/aiter/configs/model_configs/glm52_bf16_tuned_gemm.csv:ro"

OV_PIN=$R/overlay-cpupin/vllm/v1/kv_offload/cpu/gpu_worker.py:$SP/vllm/v1/kv_offload/cpu/gpu_worker.py:ro
export D_MODE=v3comm P_GEM=1 P_UTIL=0.88 D_UTIL=0.95 RENDER_TIMEOUT=600 P_RENDER=4 D_RENDER=4 P_MOUNT_X="$OV_PIN"
export D_PREFIX_CACHE=1

# 2) Gate, real acceptance.
gtag=a4-gate-gemmtune
CPU_BYTES=200000000000 D_MOUNT_X="$OV_SS $OV_GEMM" P4D4_STEP=gate_p4d4.sh bash scripts/snap.sh p4d4_run.sh "$gtag" > "$O/gate-run.log" 2>&1
line=$(grep -a "gate p4d4-$gtag:" logs/summary.out | tail -1)
echo "$line" > "$O/gate.txt"
cp results/gate-p4d4-$gtag/{smoke.out,probe-stats.out,longctx.out,metrics-delta.json,probe-review.txt} "$O/" 2>/dev/null
zcat -f "$R/runs/p4d4-$gtag"/vpd-decode*.log* 2>/dev/null | grep -a -c "not found tuned config" > "$O/gate-untuned-lines.txt"
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
echo "$(date -u +%T) a4 gate clean: $ok" | tee -a "$O/gate.txt"
[[ "$ok" == yes ]] || exit 0
for i in $(seq 60); do
    shm=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo); (( shm < 100 )) && break; sleep 10
done

# 3) Score. once/a4-extra.env may set A4_D_ENV_X (extra decode env), read here, not at job start.
A4_D_ENV_X=""
[ -r "$R/once/a4-extra.env" ] && . "$R/once/a4-extra.env"
echo "$(date -u +%T) score extra decode env: [$A4_D_ENV_X]" | tee -a "$O/gate.txt"
stag=a4-gemmtune-dprefix-syncskip-3600
run=p4d4-$stag
D=$R/runs/$run
mkdir -p "$D/steplog" "$D/instr" && chmod 777 "$D/steplog"
cp "$B/instr/vpd_steplog.py" "$B/instr/vpd_steplog.pth" "$D/instr/"
DM="$OV_SS $OV_GEMM $D/instr/vpd_steplog.py:$SP/vpd_steplog.py:ro $D/instr/vpd_steplog.pth:$SP/vpd_steplog.pth:ro $D/steplog:/steplog"
( bash scripts/memguard.sh "$run" 120 ) &
CPU_BYTES=1900000000000 DURATION=3600 USAGE_LOG=1 TRACE=1 TRACE_AT=1500 \
    D_ENV_X="VPD_STEPLOG_DIR=/steplog $A4_D_ENV_X" D_MOUNT_X="$DM" \
    bash scripts/snap.sh p4d4_run.sh "$stag"
echo $? > "logs/once-$JOB.rc"
bash scripts/finish_round.sh "logs/once-$JOB.rc" "$run" a4-gemmtune
python3 "$B/scripts/steplog_summary.py" "$D/steplog" "$D/agentx-c104/benchmark.log" 300 > "$O/steplog_summary.txt" 2>&1
zcat -f "$D"/vpd-decode*.log* 2>/dev/null | grep -a -c "not found tuned config" > "$O/score-untuned-lines.txt"
find "$D/trace" -name "dp*.json*" -type f | sort | while read -r f; do
    python3 "$B/scripts/trace_gaps.py" "$f" --step-kernel _combine_sampled_and_draft_tokens_kernel --top 12
    echo
done > "$O/trace-gaps.txt" 2>&1
f1=$(find "$D/trace" -name "dp1_*.json*" -type f | head -1)
[ -n "$f1" ] && python3 scripts/trace_cat.py "$f1" --top 40 > "$O/trace-cat-dp1.txt" 2>&1
cp "$B/../vllm-p4d4-crusoe-20261117/results/a4-gemmtune/"{summary.txt,window_tput.txt,samples_summary.txt,decode_pressure.txt,dp_balance.txt,health.txt,score.txt,ttft_breakdown.txt} "$O/" 2>/dev/null
gzip -k -f "$D"/steplog/steps-*.csv 2>/dev/null
echo "$(date -u +%m%d-%T) a4 analyses in $O" >> logs/summary.out
wait
