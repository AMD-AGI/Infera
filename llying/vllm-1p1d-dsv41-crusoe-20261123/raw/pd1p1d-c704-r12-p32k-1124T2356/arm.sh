# arm /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/arms/r12-p32k.sh
# R12: r4-hipvmm plus 32,768-token prefill steps (the recipe's 16,384). A prefill step costs ~20 ms fixed (about one
# read of the rank's weights) plus ~22 us per token (EXPERIMENT 5.1), so doubling the step amortizes the fixed part:
# about -2.6% prefill time per token, and prefill is the bottleneck. The MoE kernels see M=32768, so this arm runs the
# accuracy gate. Decode is unchanged.
# shellcheck disable=SC1091
source /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/arms/r4-hipvmm.sh
P_ARGS_X+=(--max-num-batched-tokens 32768)

# pipeline9 diagnostics
P_ENV_X+=(VPD_STEPLOG_DIR=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-r12-p32k-1124T2356/steplog-p) D_ENV_X+=(VPD_STEPLOG_DIR=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-r12-p32k-1124T2356/steplog-d)
P_ARGS_X+=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-r12-p32k-1124T2356/trace-p","torch_profiler_with_stack":false}')
D_ARGS_X+=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-r12-p32k-1124T2356/trace-d","torch_profiler_with_stack":false}')
