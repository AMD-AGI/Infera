#!/bin/bash
# run_check40_pd.sh TAG URL MODEL -> mixrun/results/check{,40}.<TAG>.{txt,json} against a PD router (detached client)
TAG=$1; URL=$2; MODEL=$3; M=/shared_nfs/yihou_glm52/dcp_pd_20261021/mixrun
docker run -d --rm --name yihou-dcp-pd-070-check40-$TAG --network host -e CHECK_MODEL=$MODEL -v $M:$M -w /tmp --entrypoint bash \
  lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c "python3 -I $M/check.py $URL $M/results/check.$TAG.json > $M/results/check.$TAG.txt 2>&1; python3 -I $M/check40.py $URL $M/results/check40.$TAG.json > $M/results/check40.$TAG.txt 2>&1"
