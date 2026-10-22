#!/bin/bash
# run_check40.sh TAG PORT -> mixrun/results/check40.<TAG>.{txt,json}
TAG=$1; PORT=$2; M=/shared_nfs/yihou_glm52/dcp_pd_20261021/mixrun
docker run -d --rm --name yihou-dcp-check40-$TAG --network host -v $M:$M -w /tmp --entrypoint bash \
  lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c "python3 -I $M/check.py http://127.0.0.1:$PORT $M/results/check.$TAG.json > $M/results/check.$TAG.txt 2>&1; python3 -I $M/check40.py http://127.0.0.1:$PORT $M/results/check40.$TAG.json > $M/results/check40.$TAG.txt 2>&1"
