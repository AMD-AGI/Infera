#!/bin/bash
# run_check.sh TAG PORT -> mixrun/results/check.<TAG>.json/.txt (detached client container)
TAG=$1; PORT=$2; M=/shared_nfs/yihou_glm52/dcp_pd_20261021/mixrun; mkdir -p $M/results
docker run -d --rm --name yihou-dcp-check-$TAG --network host -v $M:$M -w /tmp --entrypoint bash \
  lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c "python3 -I $M/check.py http://127.0.0.1:$PORT $M/results/check.$TAG.json > $M/results/check.$TAG.txt 2>&1"
