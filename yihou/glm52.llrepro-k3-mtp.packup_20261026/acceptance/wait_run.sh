#!/bin/bash
# wait_run.sh <K> <port> : wait for /health then runall
D=/shared_nfs/yihou_glm52/llrepro_20261023/acc.yihou
for i in $(seq 1 120); do
  curl -sf -m5 127.0.0.1:$2/health >/dev/null && sleep 20 && exec bash $D/runall.sh $1 $2
  docker ps --format '{{.Names}}' | grep -qx yihou-acc-k$1 || { echo "server gone" > $D/results/k$1.fail; exit 1; }
  sleep 10
done
echo timeout > $D/results/k$1.fail
