#!/bin/bash
# client.sh : start the long-lived client container (python tools, tokenizer) for building/sending requests
L=/shared_nfs/yihou_glm52/llrepro_20261023
M=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
docker run -d --name yihou-acc-client --network host -v $L:$L -v $M:$M:ro \
  --entrypoint sleep infera-sglang:v0522-fdbafb60 infinity
