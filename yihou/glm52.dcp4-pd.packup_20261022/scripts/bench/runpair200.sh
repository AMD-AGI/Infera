#!/bin/bash
# runpair200.sh TAG0 ENV0 TAG1 ENV1: like runpair.sh, but one decodebench of ~200K-token prompts (175000 words), waves 16,24,32 x 1200.
set -euo pipefail
B=/shared_nfs/yihou_glm52/dcp_pd_20261021/bench
T0=$1; E0=$2; T1=$3; E1=$4
SHA=$SHA NODE=v3-060 CPUSET=0-117 PID_OFFSET=30000 EXTRA_ENV="$E0 HSA_ENABLE_SDMA=0" $B/serve.sh yihou-dcp-bench-$T0-sim 0,1,2,3 30000 4 1 3.61 80 64
SHA=$SHA NODE=v3-060 CPUSET=118-235 PID_OFFSET=0 EXTRA_ENV="$E1 HSA_ENABLE_SDMA=0" $B/serve.sh yihou-dcp-bench-$T1-sim 4,5,6,7 31000 4 1 3.61 80 64
for i in $(seq 1 120); do
  n=$(cat $(ls -t $B/logs/bench-$T0-sim.*.log | head -1) $(ls -t $B/logs/bench-$T1-sim.*.log | head -1) | grep -c "ready to roll" || true)
  [ "$n" = 2 ] && break; sleep 10
done
echo "up=$n $(date -u +%T)"
docker run -d --rm --name yihou-dcp-bench-dbench-$T0-$T1 --network host -v $B:$B -w /tmp --entrypoint bash \
  lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c "
  for x in $T0:30000 $T1:31000; do t=\${x%%:*}; p=\${x##*:};
    python3 -I $B/decodebench.py http://127.0.0.1:\$p $B/results/decodebench.\$t-200k.json 175000 1200 16,24,32 > $B/results/decodebench.\$t-200k.txt 2>&1 &
  done; wait"
