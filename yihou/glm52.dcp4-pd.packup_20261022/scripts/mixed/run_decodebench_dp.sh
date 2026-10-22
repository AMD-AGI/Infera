#!/bin/bash
# run_decodebench.sh TAG PORT [PROMPT_TOKENS OUT_TOKENS WAVES] -> bench/results/decodebench.<TAG>.{json,txt}
TAG=$1; PORT=$2; shift 2; M=/shared_nfs/yihou_glm52/dcp_pd_20261021/mixrun; mkdir -p $M/results
docker run -d --rm --name yihou-dcp-mix-dbench-$TAG --network host -v $M:$M -w /tmp --entrypoint bash \
  lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c "python3 -I $M/decodebench_dp.py http://127.0.0.1:$PORT $M/results/decodebench.$TAG.json $* > $M/results/decodebench.$TAG.txt 2>&1"
