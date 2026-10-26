#!/bin/bash
# runall.sh <K> <port> : run the 4 request sets against one server, sequentially
D=/shared_nfs/yihou_glm52/llrepro_20261023/acc.yihou
K=$1; port=$2
mkdir -p $D/results
for s in agentx real; do for m in greedy sampled; do
  docker exec yihou-acc-client python3 $D/bench.py $port $D/req_$s.jsonl $m $D/results/k${K}_${s}_${m}.jsonl 8 > $D/results/k${K}_${s}_${m}.log 2>&1
done; done
echo done > $D/results/k${K}.done
