#!/bin/bash
# run_conc.sh TAG PORT N [N_MATH] -> bench/results/conc.<TAG>.json/.txt (detached client container)
TAG=$1; PORT=$2; N=$3; NM=${4:-0}; M=/shared_nfs/yihou_glm52/dcp_pd_20261021/bench; mkdir -p $M/results
docker run -d --rm --name yihou-dcp-bench-conc-$TAG --network host -v $M:$M -w /tmp --entrypoint bash \
  lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c "python3 -I $M/conc_check.py http://127.0.0.1:$PORT $N $M/results/conc.$TAG.json $NM > $M/results/conc.$TAG.txt 2>&1"
