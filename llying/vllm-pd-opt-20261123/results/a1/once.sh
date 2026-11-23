# Phase 1 (H3) of ../PLAN: the sync-skip overlay (overlay-syncskip, built from the image's file and checked by
# test_syncskip.py) on the P4D4 decode. 1) correctness gate with real MTP acceptance (gate.sh: smoke, GSM8K 200,
# 16k x 64, acceptance), CPU tier 200 GB so its pinned memory is gone quickly; 2) only if the gate is clean, a 3600 s
# score with the m10 config + the overlay + decode steplog + one decode trace. Submitted with -w v3-186 to compare with
# a0 (same node, same config, no overlay: 45,337). Run by run_once.sbatch (ONCE=p4d4-a1-syncskip).
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/a1
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
V=$SP/vllm
mkdir -p "$O"
cp "$0" "$O/once.sh" 2>/dev/null

gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
free=$(timeout 300 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" --entrypoint python3 "$IMAGE" \
    -c 'import torch; print(" ".join(str(int(torch.cuda.mem_get_info(i)[0] / 2**30)) for i in range(torch.cuda.device_count())))' 2>/dev/null | tail -1)
echo "$(date -u +%T) $(hostname -s) GPU free GiB: ${free:-unknown}" | tee "$O/gpu-free-at-start.txt"
for x in $free; do
    if (( x < 250 )); then echo "$(date -u +%T) a1: NOT STARTED, a GPU has only $x GiB free" | tee -a logs/summary.out; exit 0; fi
done

# The overlay, from this image's own file.
ss=$R/runs/a1-syncskip
rm -rf "$ss"; mkdir -p "$ss" && chmod 777 "$ss"
MLA=$V/v1/attention/backends/mla/rocm_aiter_mla_sparse.py
docker run --rm --network none -v "$B:$B:ro" -v "$ss:$ss" --entrypoint bash "$IMAGE" -c \
    "md5sum $MLA && python3 $B/scripts/make_syncskip_overlay.py $MLA $ss/rocm_aiter_mla_sparse.py && python3 $B/scripts/test_syncskip.py $ss/rocm_aiter_mla_sparse.py" \
    > "$O/syncskip-build-test.txt" 2>&1 || { echo "$(date -u +%T) a1: overlay build / test failed"; exit 0; }
OV_SS="$ss/rocm_aiter_mla_sparse.py:$MLA:ro"
OV_PIN=$R/overlay-cpupin/vllm/v1/kv_offload/cpu/gpu_worker.py:$SP/vllm/v1/kv_offload/cpu/gpu_worker.py:ro
export D_MODE=v3comm P_GEM=1 P_UTIL=0.88 D_UTIL=0.95 RENDER_TIMEOUT=600 P_RENDER=4 D_RENDER=4 P_MOUNT_X="$OV_PIN"

# 1) Gate, real acceptance.
gtag=a1-gate-syncskip
CPU_BYTES=200000000000 D_MOUNT_X="$OV_SS" P4D4_STEP=gate_p4d4.sh bash scripts/snap.sh p4d4_run.sh "$gtag" > "$O/gate-run.log" 2>&1
line=$(grep -a "gate p4d4-$gtag:" logs/summary.out | tail -1)
echo "$line" > "$O/gate.txt"
cp results/gate-p4d4-$gtag/{smoke.out,probe-stats.out,longctx.out,metrics-delta.json,probe-review.txt} "$O/" 2>/dev/null
zgrep -a -h "seq_lens sync" "$R/runs/p4d4-$gtag"/vpd-decode*.log* 2>/dev/null | tail -2 > "$O/gate-syncskip-counter.txt"
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
echo "$(date -u +%T) a1 gate clean: $ok" | tee -a "$O/gate.txt"
[[ "$ok" == yes ]] || exit 0

# Wait for the gate's pinned CPU tier to be released (p4d4_run refuses to deploy above 200 GB of Shmem).
for i in $(seq 60); do
    shm=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo); (( shm < 100 )) && break; sleep 10
done
echo "$(date -u +%T) Shmem before the score: ${shm} GB" >> "$O/gate.txt"

# 2) Score: the m10 config + the overlay, decode steplog, one trace.
stag=a1-syncskip-m10cfg-3600
run=p4d4-$stag
D=$R/runs/$run
mkdir -p "$D/steplog" "$D/instr" && chmod 777 "$D/steplog"
cp "$B/instr/vpd_steplog.py" "$B/instr/vpd_steplog.pth" "$D/instr/"
DM="$OV_SS $D/instr/vpd_steplog.py:$SP/vpd_steplog.py:ro $D/instr/vpd_steplog.pth:$SP/vpd_steplog.pth:ro $D/steplog:/steplog"
( bash scripts/memguard.sh "$run" 120 ) &
CPU_BYTES=1900000000000 DURATION=3600 USAGE_LOG=1 TRACE=1 TRACE_AT=1500 D_ENV_X='VPD_STEPLOG_DIR=/steplog' D_MOUNT_X="$DM" \
    bash scripts/snap.sh p4d4_run.sh "$stag"
echo $? > "logs/once-$JOB.rc"
bash scripts/finish_round.sh "logs/once-$JOB.rc" "$run" a1-syncskip
python3 "$B/scripts/steplog_summary.py" "$D/steplog" "$D/agentx-c104/benchmark.log" 300 > "$O/steplog_summary.txt" 2>&1
find "$D/trace" -name "dp*.json*" -type f | sort | while read -r f; do
    python3 "$B/scripts/trace_gaps.py" "$f" --step-kernel _combine_sampled_and_draft_tokens_kernel --top 12
    echo
done > "$O/trace-gaps.txt" 2>&1
zgrep -a -h "seq_lens sync" "$D"/vpd-decode*.log* 2>/dev/null | tail -4 > "$O/syncskip-counter.txt"
cp "$B/../vllm-p4d4-crusoe-20261117/results/a1-syncskip/"{summary.txt,window_tput.txt,samples_summary.txt,decode_pressure.txt,dp_balance.txt,health.txt,score.txt} "$O/" 2>/dev/null
gzip -k -f "$D"/steplog/steps-*.csv 2>/dev/null
echo "$(date -u +%m%d-%T) a1 analyses in $O" >> logs/summary.out
wait
