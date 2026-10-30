# Site values for examples/sglang_mix_glm5.2 on crsuse2-m2m-141 (source before engine/*.sh).
export NODE_IP=10.245.148.188
export MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
export IMAGE="${IMAGE:-infera-sglang:rocm102}"
export WORK_DIR=/home/liyingli/rocm102-verify/work/mix
export GPUS=0,1,2,3
export CTR=r102_mix
