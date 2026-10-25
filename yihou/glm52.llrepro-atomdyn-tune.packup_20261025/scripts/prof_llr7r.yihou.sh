#!/usr/bin/env bash
# prof_llr7r (task llrepro bottleneck report 2026-10-24, ON 135): during the llr7r replicate (= llr7 config), 25 min into
# the AgentX profiling phase: 1 s RDMA/CPU sampling on both nodes, py-spy dumps of the decode, a torch profile of the
# decode (40 steps), then 5 min later of the prefill (20 steps). Traces stay inside the containers (/tmp/yihou_prof)
# and are docker-cp'd to NFS after run_llr has stopped them. Read-only otherwise; deletes nothing.
set -uo pipefail
D=/shared_nfs/yihou_glm52/llrepro_20261023; L=$D/run_llr.llr7r.yihou.log; O=$D/trace_llr7r.yihou; mkdir -p $O
ts(){ date -u +%FT%TZ; }
DEC=http://10.245.154.168:27301; PRE=http://10.245.148.209:27201
for i in $(seq 1 480); do grep -q "Phase profiling (profiling) started" $L 2>/dev/null && break; sleep 15; done
grep -q "Phase profiling (profiling) started" $L || { echo "$(ts) no profiling phase"; exit 1; }
echo "$(ts) profiling phase seen; sleeping 1500 s"; sleep 1500
# 1 s rail counters + per-core CPU for 120 s on both nodes (background)
for n in 135 136; do ssh -n -o BatchMode=yes crsuse2-m2m-$n "setsid nohup bash $D/node_probe1s.yihou.sh $O/probe 120 >/dev/null 2>&1 < /dev/null &"; done
# py-spy: busiest decode threads (3 dumps, 10 s apart)
for k in 1 2 3; do
  ssh -n -o BatchMode=yes crsuse2-m2m-136 "docker exec yihou-llr-decode sh -c 'for p in \$(pgrep -f \"sglang|python3\"); do echo \"=== pid \$p \$(cat /proc/\$p/comm) cpu=\$(ps -o %cpu= -p \$p)\"; timeout 20 py-spy dump --nonblocking --pid \$p 2>&1 | head -80; done'" > $O/pyspy.decode.$k.txt 2>&1
  sleep 10
done
echo "$(ts) decode start_profile"
curl -s -m 30 -X POST $DEC/start_profile -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/yihou_prof/decode","num_steps":40,"activities":["CPU","GPU"],"profile_prefix":"llr7r-decode"}' > $O/start_profile.decode.json 2>&1
echo " rc=$? $(cat $O/start_profile.decode.json)"
sleep 300
echo "$(ts) prefill start_profile"
curl -s -m 30 -X POST $PRE/start_profile -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/yihou_prof/prefill","num_steps":20,"activities":["CPU","GPU"],"profile_prefix":"llr7r-prefill"}' > $O/start_profile.prefill.json 2>&1
echo " rc=$? $(cat $O/start_profile.prefill.json)"
# wait for run_llr to finish (it saves logs and docker-stops; containers are kept), then copy the traces out
for i in $(seq 1 480); do grep -q RUN_LLR_DONE $L && break; sleep 30; done
docker cp yihou-llr-prefill0:/tmp/yihou_prof/prefill $O/prefill 2>&1 | tail -2
ssh -n -o BatchMode=yes crsuse2-m2m-136 "docker cp yihou-llr-decode:/tmp/yihou_prof/decode $O/decode" 2>&1 | tail -2
du -sh $O/* ; echo "$(ts) PROF_DONE"
