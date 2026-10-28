#!/usr/bin/env bash
# guard_llr (task llrepro 2026-10-24, ON 135): while a run_llr round is in its AgentX PROFILING phase, stop OUR AgentX
# client (docker stop yihou-llr-agentx) if the decode on 136 logs no "Decode batch" line for 300 s, or logs an HSA/HIP
# fatal. run_llr.yihou.sh then saves logs and stops the stack as usual. Touches nothing else.
# v2 (10:55Z): arm only after "Phase profiling (profiling) started" - during warm-up the decode logs no batch lines for
# ~19 min (llr7 first Decode batch 03:13 vs AgentX start 02:54); v1 stopped llr9b in warm-up by mistake.
D=/shared_nfs/yihou_glm52/llrepro_20261023; L=$D/run_llr.$1.yihou.log
ts(){ date -u +%FT%TZ; }
for i in $(seq 1 960); do grep -q "Phase profiling (profiling) started" $L 2>/dev/null && break; grep -q "agentx rc=" $L 2>/dev/null && { echo "$(ts) run ended before profiling"; exit 0; }; sleep 15; done
grep -q "Phase profiling (profiling) started" $L || { echo "$(ts) no profiling phase seen"; exit 0; }
echo "$(ts) armed (profiling phase started)"
sleep 300
while ! grep -q "agentx rc=" $L; do
  n=$(ssh -n -o BatchMode=yes crsuse2-m2m-136 "docker logs --since 300s yihou-llr-decode 2>&1 | grep -c 'Decode batch'")
  f=$(ssh -n -o BatchMode=yes crsuse2-m2m-136 "docker logs --since 90s yihou-llr-decode 2>&1 | grep -c -E 'HSA_STATUS_ERROR|Memory access fault|Fatal Python error'")
  if [[ "${n:-x}" == 0 || ( -n "$f" && "$f" != 0 ) ]]; then
    echo "$(ts) GUARD: decode batches in 300 s=$n fatal lines=$f -> docker stop yihou-llr-agentx"
    docker stop -t 30 yihou-llr-agentx; break
  fi
  sleep 60
done
echo "$(ts) GUARD_DONE"
