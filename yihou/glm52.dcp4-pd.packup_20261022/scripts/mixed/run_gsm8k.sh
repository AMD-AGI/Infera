#!/bin/bash
# run_gsm8k.sh TAG PORT [N] -> mixrun/results/gsm8k.<TAG>/ (sgl-eval 0.1.3, GLM-5.2 preset: T=1.0 top_p=0.95 max_tokens 200000, thinking via enable_thinking=true)
TAG=$1; PORT=$2; N=${3:-200}; MAXTOK=${4:-}; M=/shared_nfs/yihou_glm52/dcp_pd_20261021/mixrun; V=$M/venv-sgleval; mkdir -p $M/results/gsm8k.$TAG
docker run -d --rm --name yihou-dcp-gsm8k-$TAG --network host -v $M:$M -w /tmp -e HOME=$M/results/gsm8k.$TAG --entrypoint bash \
  lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020 -c "$V/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:$PORT/v1 --model amd/GLM-5.2-MXFP4 --load-preset-from-model-id zai-org/GLM-5.2-FP8 --chat-template-kwarg enable_thinking=true --num-examples $N ${MAXTOK:+--max-tokens $MAXTOK} --out-dir $M/results/gsm8k.$TAG > $M/results/gsm8k.$TAG/run.log 2>&1"
