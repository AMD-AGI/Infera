# R15: r4-hipvmm + the prefill leg runs each MoE layer's shared expert on vLLM's auxiliary stream, overlapped with the
# routed experts' all-gather / GEMM / reduce-scatter (EXPERIMENT 4.11, throughput candidates). vLLM does this only for
# steps of at most VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD (256) tokens; prefill steps are 16k, so the shared expert
# (an unfused FP8 MLP under DP2, ~46 TFLOP per step, ~5% of the step) ran serially. Decode is unchanged.
source /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/arms/r4-hipvmm.sh
P_ENV_X+=(VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=65536)
