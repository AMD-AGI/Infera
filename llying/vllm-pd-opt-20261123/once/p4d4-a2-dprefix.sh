# Phase 1 (H2) of ../PLAN: decode prefix caching on the P4D4 (pd1_up D_PREFIX_CACHE=1 -> --enable-prefix-caching on the
# decode), so a request preempted on the decode recomputes only the blocks that were reused meanwhile (b0 round 2:
# recompute ~1% of the baseline, +20% generated in the harness). The Mooncake consumer pulls only the blocks the decode
# does not hold (mooncake_connector.py:1188-1196). 1) correctness gate with real acceptance (CPU tier 200 GB);
# 2) only if clean, a 3600 s score with the m10 config + decode prefix caching + decode steplog + one decode trace.
# Submitted with -w v3-186 to compare with a0 (same node, same config, no prefix caching: 45,337).
# Run by run_once.sbatch (ONCE=p4d4-a2-dprefix).
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/a2
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
mkdir -p "$O"
cp "$0" "$O/once.sh" 2>/dev/null

gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
free=$(timeout 300 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" --entrypoint python3 "$IMAGE" \
    -c 'import torch; print(" ".join(str(int(torch.cuda.mem_get_info(i)[0] / 2**30)) for i in range(torch.cuda.device_count())))' 2>/dev/null | tail -1)
echo "$(date -u +%T) $(hostname -s) GPU free GiB: ${free:-unknown}" | tee "$O/gpu-free-at-start.txt"
for x in $free; do
    if (( x < 250 )); then echo "$(date -u +%T) a2: NOT STARTED, a GPU has only $x GiB free" | tee -a logs/summary.out; exit 0; fi
done

OV_PIN=$R/overlay-cpupin/vllm/v1/kv_offload/cpu/gpu_worker.py:$SP/vllm/v1/kv_offload/cpu/gpu_worker.py:ro
export D_MODE=v3comm P_GEM=1 P_UTIL=0.88 D_UTIL=0.95 RENDER_TIMEOUT=600 P_RENDER=4 D_RENDER=4 P_MOUNT_X="$OV_PIN"
export D_PREFIX_CACHE=1

# 1) Gate, real acceptance.
gtag=a2-gate-dprefix
CPU_BYTES=200000000000 P4D4_STEP=gate_p4d4.sh bash scripts/snap.sh p4d4_run.sh "$gtag" > "$O/gate-run.log" 2>&1
line=$(grep -a "gate p4d4-$gtag:" logs/summary.out | tail -1)
echo "$line" > "$O/gate.txt"
cp results/gate-p4d4-$gtag/{smoke.out,probe-stats.out,longctx.out,metrics-delta.json,probe-review.txt} "$O/" 2>/dev/null
grep -a -c -- "--enable-prefix-caching" "$R/runs/p4d4-$gtag/engine-config.env" > "$O/gate-dprefix-flag-lines.txt" 2>&1
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
echo "$(date -u +%T) a2 gate clean: $ok" | tee -a "$O/gate.txt"
[[ "$ok" == yes ]] || exit 0

for i in $(seq 60); do
    shm=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo); (( shm < 100 )) && break; sleep 10
done
echo "$(date -u +%T) Shmem before the score: ${shm} GB" >> "$O/gate.txt"

# 2) Score: the m10 config + decode prefix caching, decode steplog, one trace.
stag=a2-dprefix-m10cfg-3600
run=p4d4-$stag
D=$R/runs/$run
mkdir -p "$D/steplog" "$D/instr" && chmod 777 "$D/steplog"
cp "$B/instr/vpd_steplog.py" "$B/instr/vpd_steplog.pth" "$D/instr/"
DM="$D/instr/vpd_steplog.py:$SP/vpd_steplog.py:ro $D/instr/vpd_steplog.pth:$SP/vpd_steplog.pth:ro $D/steplog:/steplog"
( bash scripts/memguard.sh "$run" 120 ) &
CPU_BYTES=1900000000000 DURATION=3600 USAGE_LOG=1 TRACE=1 TRACE_AT=1500 D_ENV_X='VPD_STEPLOG_DIR=/steplog' D_MOUNT_X="$DM" \
    bash scripts/snap.sh p4d4_run.sh "$stag"
echo $? > "logs/once-$JOB.rc"
bash scripts/finish_round.sh "logs/once-$JOB.rc" "$run" a2-dprefix
python3 "$B/scripts/steplog_summary.py" "$D/steplog" "$D/agentx-c104/benchmark.log" 300 > "$O/steplog_summary.txt" 2>&1
find "$D/trace" -name "dp*.json*" -type f | sort | while read -r f; do
    python3 "$B/scripts/trace_gaps.py" "$f" --step-kernel _combine_sampled_and_draft_tokens_kernel --top 12
    echo
done > "$O/trace-gaps.txt" 2>&1
cp "$B/../vllm-p4d4-crusoe-20261117/results/a2-dprefix/"{summary.txt,window_tput.txt,samples_summary.txt,decode_pressure.txt,dp_balance.txt,health.txt,score.txt,ttft_breakdown.txt} "$O/" 2>/dev/null
gzip -k -f "$D"/steplog/steps-*.csv 2>/dev/null
echo "$(date -u +%m%d-%T) a2 analyses in $O" >> logs/summary.out
wait
