# arm /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/arms/r15-shexp.sh
# R15: r4-hipvmm + the prefill leg runs each MoE layer's shared expert on vLLM's auxiliary stream, overlapped with the
# routed experts' all-gather / GEMM / reduce-scatter (EXPERIMENT 4.11, throughput candidates). vLLM does this only for
# steps of at most VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD (256) tokens; prefill steps are 16k, so the shared expert
# (an unfused FP8 MLP under DP2, ~46 TFLOP per step, ~5% of the step) ran serially. Decode is unchanged.
source /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/arms/r4-hipvmm.sh
P_ENV_X+=(VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=65536)

# pipeline11 diagnostics
P_ENV_X+=(VLLM_UVA_VMM=1 VLLM_UVA_VMM_MIN_BYTES=1048576) D_ENV_X+=(VLLM_UVA_VMM=1 VLLM_UVA_VMM_MIN_BYTES=1048576)
P_ENV_X+=(LD_PRELOAD=/usr/local/lib/libhip_gem_alloc.so:/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-r15-shexp-1125T1810/hostalloc_log.so HOSTALLOC_LOG_DIR=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-r15-shexp-1125T1810/hostalloc) D_ENV_X+=(LD_PRELOAD=/usr/local/lib/libhip_gem_alloc.so:/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-r15-shexp-1125T1810/hostalloc_log.so HOSTALLOC_LOG_DIR=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-r15-shexp-1125T1810/hostalloc)
P_ENV_X+=(VPD_STEPLOG_DIR=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-r15-shexp-1125T1810/steplog-p) D_ENV_X+=(VPD_STEPLOG_DIR=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-r15-shexp-1125T1810/steplog-d VPD_STEPLOG_FLUSH_S=0 VPD_STEPLOG_BEGIN=1)
P_ARGS_X+=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-r15-shexp-1125T1810/trace-p","torch_profiler_with_stack":false}')
D_ARGS_X+=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-r15-shexp-1125T1810/trace-d","torch_profiler_with_stack":false}')
