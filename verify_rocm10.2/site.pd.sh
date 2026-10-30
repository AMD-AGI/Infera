# Site values for examples/sglang_2p1d_glm5.2 on crsuse2-m2m-141, single node:
# 1 x TP1*PP4 prefill on GPUs 0-3, TP4 + DP4-attention decode on GPUs 4-7.
export PREFILL_NODE_IP=10.245.148.188
export DECODE_NODE_IP=10.245.148.188
export DECODE_SSH=crsuse2-m2m-141
export MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
export IMAGE="${IMAGE:-infera-sglang:rocm102}"
export WORK_DIR=/home/liyingli/rocm102-verify/work/pd
export PREFILL_WORKERS=1
export PREFILL0_GPUS=0,1,2,3
export DECODE_GPUS=4,5,6,7
export CTR=r102_pd
