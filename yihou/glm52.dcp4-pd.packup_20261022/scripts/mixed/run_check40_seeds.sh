#!/bin/bash
# run_check40_seeds.sh TAG URL MODEL SEED... -> check40 only, one result per seed (sequential, detached container)
TAG=$1; URL=$2; MODEL=$3; shift 3; M=/shared_nfs/yihou_glm52/dcp_pd_20261021/mixrun
cmd=""; for s in "$@"; do cmd+="CHECK_SEED=$s python3 -I $M/check40.py $URL $M/results/check40.$TAG-s$s.json > $M/results/check40.$TAG-s$s.txt 2>&1; "; done
docker run -d --rm --name yihou-dcp-pd-070-seeds-$TAG --network host -e CHECK_MODEL=$MODEL -v $M:$M -w /tmp --entrypoint bash \
  lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c "$cmd"
