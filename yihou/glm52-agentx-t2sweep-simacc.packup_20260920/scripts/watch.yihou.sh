#!/usr/bin/env bash
# Lightweight sweep watcher. Records fault-count + a liveness sample each
# interval to monitor.log; exits when a Memory access fault appears OR the sweep
# process dies. It does NOT act — the driver agent reads monitor.log and decides.
set -uo pipefail
W="/home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd/results/yihou-agentx-hicache"
PREFIX="${PREFIX:-t2f}"
DLOG="$W/$PREFIX-launch/server-logs/decode-0.log"
MON="$W/notes/monitor.log"
INT="${INT:-120}"
echo "watch start $(date -u +%FT%TZ) int=${INT}s" >> "$MON"
while true; do
  ts="$(date -u +%FT%TZ)"
  faults=$(grep -c "Memory access fault" "$DLOG" 2>/dev/null || true)
  faults=${faults:-0}
  sweep_alive=$(pgrep -f 'sweep\.yiho[u]' >/dev/null 2>&1 && echo Y || echo N)
  bench_alive=$(pgrep -f 'agentx_ben[c]h' >/dev/null 2>&1 && echo Y || echo N)
  # current point log = newest <prefix>-agentx-c*.log
  clog=$(ls -t "$W"/$PREFIX-agentx-c*.log 2>/dev/null | head -1)
  live=$(grep -aE 'tot in=|returned=' "$clog" 2>/dev/null | tail -1 | tr -s ' ' | cut -c1-160)
  # /home free: the CONC=128 death on 2026-09-19 was the shared NFS volume hitting
  # 0 bytes, not a GPU fault. Sampling it here lets us tell an infra repeat from a
  # real fault at the moment of death.
  homefree=$(df -h /home 2>/dev/null | awk 'NR==2{print $4" "$5}')
  echo "$ts faults=$faults sweep=$sweep_alive bench=$bench_alive home=$homefree | $live" >> "$MON"
  if [[ "$faults" != "0" && "$faults" != "-1" ]]; then
    echo "$ts FAULT-DETECTED count=$faults" >> "$MON"; exit 10
  fi
  if [[ "$sweep_alive" == "N" ]]; then
    echo "$ts SWEEP-ENDED" >> "$MON"; exit 0
  fi
  sleep "$INT"
done
