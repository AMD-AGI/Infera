#!/usr/bin/env bash
# prof_llr19 (task llrepro 2026-10-25, plan F via rocprofv3 attach): llr19 = llr14 config unchanged (also a 2nd replicate).
# 20 min into the AgentX profiling phase, attach rocprofv3 (kernel + memcpy trace, csv) to the decode schedulers DP0 and
# DP2 for 3 s each, one after the other, so the CUDA-graph verify kernels are recorded (torch profiler recorded none).
# Outputs are written inside the decode container (/tmp/yihou_rp) and docker-cp'd to NFS after the run. Touches only
# our yihou-llr-decode container.
set -uo pipefail
D=/shared_nfs/yihou_glm52/llrepro_20261023; L=$D/run_llr.llr19.yihou.log; O=$D/trace_llr19.yihou; mkdir -p $O
ts(){ date -u +%FT%TZ; }
for i in $(seq 1 480); do grep -q "Phase profiling (profiling) started" $L 2>/dev/null && break; sleep 15; done
grep -q "Phase profiling (profiling) started" $L || { echo "$(ts) no profiling phase"; exit 1; }
echo "$(ts) profiling phase seen; sleeping 1200 s"; sleep 1200
ssh -n -o BatchMode=yes crsuse2-m2m-136 "docker exec yihou-llr-decode sh -c 'ps -eo pid,comm,args | grep -E \"sglang::schedul\" | grep -v grep'" > $O/sched_pids.txt
cat $O/sched_pids.txt
for k in 1 3; do   # 2nd and 4th scheduler lines = DP?; record the mapping from args below
  pid=$(awk -v k=$k 'NR==k{print $1}' $O/sched_pids.txt)
  [ -n "$pid" ] || continue
  echo "$(ts) attach rocprofv3 to pid $pid (line $k)"
  ssh -n -o BatchMode=yes crsuse2-m2m-136 "docker logs --since 60s yihou-llr-decode 2>&1 | grep 'Decode batch' | tail -8" > $O/batches_before.$pid.txt
  ssh -n -o BatchMode=yes crsuse2-m2m-136 "docker exec yihou-llr-decode sh -c 'mkdir -p /tmp/yihou_rp && timeout 180 /opt/rocm/bin/rocprofv3 --attach $pid --attach-duration-msec 3000 --kernel-trace --memory-copy-trace -f csv -d /tmp/yihou_rp/p$pid -o p$pid'" > $O/rocprof.$pid.log 2>&1
  echo "$(ts) rc=$? $(tail -2 $O/rocprof.$pid.log)"
  sleep 60
done
for i in $(seq 1 480); do grep -q RUN_LLR_DONE $L && break; sleep 30; done
ssh -n -o BatchMode=yes crsuse2-m2m-136 "docker cp yihou-llr-decode:/tmp/yihou_rp $O/rp" 2>&1 | tail -2
du -sh $O/rp; echo "$(ts) PROF19_DONE"
