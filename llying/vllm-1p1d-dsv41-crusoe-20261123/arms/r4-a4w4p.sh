# R4: R1 plus FP4 activations in prefill's MoE (a4w4 instead of the default a8w4; VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4,
# an opt-in validated for deepseek_v41 in vLLM). MoE GEMMs are 26% of prefill GPU time and MI355X runs FP4 MFMA at
# twice the FP8 rate. Decode keeps a8w4. Numerics change, so this arm runs the accuracy gate (GATE=1).
# shellcheck disable=SC1091
source /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/arms/r1-capacity.sh
P_ENV_X+=(VLLM_ROCM_USE_AITER_MOE_A4W4_DSV4=1)
