#!/usr/bin/env bash
# prof_llr18 (task llrepro 2026-10-25, plan F, diagnostic): llr18 = llr14 with decode CUDA graphs disabled, so the torch
# profiler records the kernels of the verify forward (llr7r: the verify graph had 0 kernel records). 10 min into the
# profiling phase: decode torch profile 30 steps; copy the traces out; then stop OUR AgentX client (the score of an eager
# decode is not a result). Touches only yihou-llr-* containers.
set -uo pipefail
D=/shared_nfs/yihou_glm52/llrepro_20261023; L=$D/run_llr.llr18.yihou.log; O=$D/trace_llr18.yihou; mkdir -p $O
ts(){ date -u +%FT%TZ; }
for i in $(seq 1 480); do grep -q "Phase profiling (profiling) started" $L 2>/dev/null && break; sleep 15; done
grep -q "Phase profiling (profiling) started" $L || { echo "$(ts) no profiling phase"; exit 1; }
echo "$(ts) profiling phase seen; sleeping 600 s"; sleep 600
ssh -n -o BatchMode=yes crsuse2-m2m-136 "docker logs --since 120s yihou-llr-decode 2>&1 | grep 'Decode batch' | tail -8" > $O/decode_batches_before.txt
echo "$(ts) decode start_profile"
curl -s -m 30 -X POST http://10.245.154.168:27301/start_profile -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/yihou_prof/decode","num_steps":30,"activities":["CPU","GPU"],"profile_prefix":"llr18-decode-eager"}' > $O/start_profile.decode.json 2>&1
echo " rc=$? $(cat $O/start_profile.decode.json)"
sleep 240
ssh -n -o BatchMode=yes crsuse2-m2m-136 "docker cp yihou-llr-decode:/tmp/yihou_prof/decode $O/decode" 2>&1 | tail -2
ls -la $O/decode
echo "$(ts) stopping our AgentX client (diagnostic run)"; docker stop -t 30 yihou-llr-agentx
echo "$(ts) PROF18_DONE"
