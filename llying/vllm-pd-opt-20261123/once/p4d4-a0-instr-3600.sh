# Phase 0a of ../PLAN (vllm-pd-opt-plan-20261123): the m10 config (best 3600 s, 48,572 tok/s/GPU) with decode
# instrumentation, to split the ~6 ms per step between production decode (~49.8 ms) and pure decode (~43-44 ms):
#   - vpd_steplog in every decode process: per step cadence, batch make-up, CUDA graph mode, memory every 500 steps
#   - decode torch traces (3 s) at 1500 s and 2700 s into the measurement
#   - py-spy record (non-blocking, 20 s) of every decode Python process and a thread CPU snapshot at 1200 s
# Run by run_once.sbatch (ONCE=p4d4-a0-instr-3600); standard analyses land in
# ../vllm-p4d4-crusoe-20261117/results/a0-instr/, phase-0 analyses in llying/vllm-pd-opt-20261123/results/a0/.
tag=a0-instr-m10cfg-3600
run=p4d4-$tag
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
D=$R/runs/$run
O=$B/results/a0
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
mkdir -p "$D/steplog" "$D/instr" "$O" && chmod 777 "$D/steplog"
cp "$B/instr/vpd_steplog.py" "$B/instr/vpd_steplog.pth" "$D/instr/"
cp "$0" "$O/once.sh" 2>/dev/null

# Another job's leftovers can hold the GPUs (v3-168 on 11-20): stop if any GPU has less than 250 GiB free.
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
free=$(timeout 300 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" --entrypoint python3 "$IMAGE" \
    -c 'import torch; print(" ".join(str(int(torch.cuda.mem_get_info(i)[0] / 2**30)) for i in range(torch.cuda.device_count())))' 2>/dev/null | tail -1)
echo "$(date -u +%T) GPU free GiB: ${free:-unknown}" | tee "$O/gpu-free-at-start.txt"
for x in $free; do
    if (( x < 250 )); then echo "$(date -u +%T) $run: NOT STARTED, a GPU has only $x GiB free" | tee -a logs/summary.out; exit 0; fi
done

# Decode-side probes keyed to the AIPerf measurement start.
measure_wait() {
    while ! grep -qa "Phase profiling (profiling) started" "$D/agentx-c104/benchmark.log" 2>/dev/null; do
        [ -e "$D/samples.jsonl.stop" ] && return 1; sleep 30
    done
}
( measure_wait || exit 0; sleep 1200
  # shellcheck disable=SC2016
  docker exec vpd-decode bash -c 'pip install -q py-spy > /dev/null 2>&1
      for d in /proc/[0-9]*; do
          pid=${d#/proc/}
          case "$(readlink "$d/exe" 2>/dev/null)" in *python*) ;; *) continue ;; esac
          name=$(tr -c "A-Za-z0-9_\n" "_" < "$d/comm")
          timeout 60 py-spy record --nonblocking -d 20 -r 50 -f raw -o "/steplog/pyspy-$pid-$name.raw" -p "$pid" > /dev/null 2>&1 &
      done
      top -b -H -n 2 -d 5 -w 200 > /steplog/top-decode.txt 2>&1
      wait' > "$D/steplog/pyspy.log" 2>&1
  echo "$(date -u +%T) py-spy done" >> "$D/steplog/pyspy.log" ) &
( measure_wait || exit 0; sleep 2700
  bash scripts/trace_decode.sh "http://$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}'):27301" 3 \
      > "$D/trace/trace2.log" 2>&1 ) &

OV=$R/overlay-cpupin/vllm/v1/kv_offload/cpu/gpu_worker.py:$SP/vllm/v1/kv_offload/cpu/gpu_worker.py:ro
DM="$D/instr/vpd_steplog.py:$SP/vpd_steplog.py:ro $D/instr/vpd_steplog.pth:$SP/vpd_steplog.pth:ro $D/steplog:/steplog"
( bash scripts/memguard.sh "$run" 120 ) &
D_MODE=v3comm P_GEM=1 P_UTIL=0.88 D_UTIL=0.95 RENDER_TIMEOUT=600 P_RENDER=4 D_RENDER=4 DURATION=3600 USAGE_LOG=1 \
    CPU_BYTES=1900000000000 TRACE=1 TRACE_AT=1500 D_ENV_X='VPD_STEPLOG_DIR=/steplog' P_MOUNT_X="$OV" D_MOUNT_X="$DM" \
    bash scripts/snap.sh p4d4_run.sh "$tag"
echo $? > "logs/once-$JOB.rc"
bash scripts/finish_round.sh "logs/once-$JOB.rc" "$run" a0-instr

# Phase-0 analyses (on this node; the node is released afterwards).
zgrep -a "vpd_steplog" "$D"/vpd-decode.stream.log* 2>/dev/null | head -20 > "$O/steplog-patch-lines.txt"
ls -la "$D/steplog" > "$O/steplog-files.txt" 2>&1
python3 "$B/scripts/steplog_summary.py" "$D/steplog" "$D/agentx-c104/benchmark.log" 300 > "$O/steplog_summary.txt" 2>&1
find "$D/trace" -name "*.json*" -type f | sort | while read -r f; do
    python3 scripts/trace_cat.py "$f" --top 40 > "$O/trace-cat-$(basename "$f" | sed 's/\.pt\.trace\.json.*//' | cut -c1-80).txt" 2>&1
done
for f in "$D"/steplog/pyspy-*.raw; do
    [ -s "$f" ] && python3 scripts/collapse_top.py "$f" 40 > "$O/$(basename "$f" .raw).txt" 2>&1
done
cp "$D/steplog/top-decode.txt" "$D/steplog/pyspy.log" "$O/" 2>/dev/null
cp "$B/../vllm-p4d4-crusoe-20261117/results/a0-instr/"{summary.txt,window_tput.txt,samples_summary.txt,decode_pressure.txt,dp_balance.txt,health.txt,score.txt} "$O/" 2>/dev/null
gzip -k -f "$D"/steplog/steps-*.csv 2>/dev/null
echo "$(date -u +%m%d-%T) a0 phase-0 analyses in $O" >> logs/summary.out
wait
