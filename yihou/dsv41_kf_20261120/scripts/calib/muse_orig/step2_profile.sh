#!/bin/bash
# Step 2 (node host side, run under nohup): Muse TP4 server under the fork json-tool (mode window) + unscored
# C40 AgentX load; one 20-VERIFY-step window per rank at running >= THR; then drain, finalize, stop.
# Modeled on scripts/prof_cru.sh (same server_launch.py / wait_then_client.py / stop_owned.py chain).
# Usage: nohup bash step2_profile.sh <grant.json> <tag> [conc=40] [thr=20] > $NFS/profiles/<tag>-calib.driver.log 2>&1 &
set -u
source "$(dirname "$0")/env.sh"
export YIHOU_MUSE_MODELS=/mnt/m2m_nobackup/yihou/muse_agentx_opt_20261027/models YIHOU_MUSE_GRANT=$1
tag=$2; conc=${3:-40}; thr=${4:-20}
ctag=${tag%-*}-load-${tag##*-}
T=$NFS/profiles/$tag-calib                      # = CALIB_TRACE_DIR in the container
ts() { date -u +%FT%TZ; }
trap 'rc=$?; echo "$(ts) driver exit rc=$rc"' EXIT   # every exit path leaves a line the caller can wait on
running() { curl -s -m5 localhost:28000/metrics | grep -E '^sglang:num_running_reqs\{' | grep 'tp_rank="0"' | awk '{print int($2)}' | head -1; }
[ -f "$LCP/build/BUILD_OK" ] || { echo "$(ts) no BUILD_OK"; exit 1; }
[ -e "$T" ] && { echo "$(ts) trace dir exists; use a fresh tag"; exit 1; }
mkdir -p "$T"
cd "$NFS/workspace/scripts" || exit 1
while docker ps --format '{{.Names}}' | grep -q '^yihou-muse-opt-'; do sleep 20; done
echo "$(ts) launch $tag"
python3 server_launch.py ../iterations/$tag/config.json --score-duration ${CALIB_SCORE_S:-900} --warmup-budget-seconds ${CALIB_WB_S:-900} \
  --startup-budget-seconds ${CALIB_SB_S:-1800} || { echo "$(ts) launch failed"; exit 1; }
python3 wait_then_client.py $ctag --concurrency $conc --expect-interval 24 --expect-mem 0.86 -- --duration ${CALIB_SCORE_S:-900} \
  --warmup 2 --mode screen --kv-offloading dram --warmup-budget-seconds ${CALIB_WB_S:-900} --tp 4 \
  --image-label lmsysorg/sglang-rocm:muse-spark-open-mi355-1106@sha256:d231181df079a5b271a0459d024492c84a36caf8cb26ee019abad29e17eea4d2 \
  || echo "$(ts) client launch failed $ctag"
# trigger: running >= thr for 3 consecutive 10 s samples (as load_profile_trigger.sh), max 150 min
n=0; fired=0
for i in $(seq 1 900); do
  r=$(running); echo "$(ts) running=${r:-NA}"
  if [ "${r:-0}" -ge "$thr" ]; then n=$((n + 1)); else n=0; fi
  if [ "$n" -ge 3 ]; then touch "$T/start_window"; fired=1; echo "$(ts) start_window"; break; fi
  curl -s -m5 -o /dev/null -w '%{http_code}' localhost:28000/health | grep -q 200 || { echo "$(ts) server unhealthy"; break; }
  sleep 10
done
# wait for the window to close on every rank (steps-<pid>.json), max 10 min
if [ "$fired" = 1 ]; then
  for i in $(seq 1 60); do
    ns=$(ls "$T"/steps-*.json 2>/dev/null | wc -l)
    [ "$ns" -ge 4 ] && break
    sleep 10
  done
  echo "$(ts) steps files: $(ls "$T"/steps-*.json 2>/dev/null | wc -l)  (expect 4 = TP ranks)"
fi
# drain: stop client, wait until running == 0 (max 10 min), then finalize
docker stop -t 60 yihou-muse-opt-client-$ctag >/dev/null 2>&1
for i in $(seq 1 60); do r=$(running); [ "${r:-0}" -eq 0 ] && break; sleep 10; done
echo "$(ts) running after drain=${r:-NA}; finalize"
touch "$T/finalize"
for i in $(seq 1 120); do
  nf=0; for s in "$T"/steps-*.json; do [ -e "$s" ] || continue; p=${s##*/steps-}; p=${p%.json}; [ -e "$T/finalized-$p" ] && nf=$((nf + 1)); done
  ns=$(ls "$T"/steps-*.json 2>/dev/null | wc -l)
  [ "$ns" -gt 0 ] && [ "$nf" -ge "$ns" ] && break
  sleep 10
done
echo "$(ts) finalized $nf of $ns window ranks; trace bytes $(du -sb "$T" | cut -f1)"
docker logs yihou-muse-opt-$tag > "$NFS/iterations/$tag/docker.log" 2>&1
python3 stop_owned.py $tag --drain-timeout 600 | tail -1
echo "$(ts) $tag done"
