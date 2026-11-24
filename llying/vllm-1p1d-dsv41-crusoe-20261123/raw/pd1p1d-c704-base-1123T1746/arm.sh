# arm /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/arms/base.sh
# Baseline: the AUS recipe as is (kit defaults: P and D at gpu-memory-utilization 0.9, decode prefill-schedule-interval
# 11, CUDA graphs up to 1024 tokens). pipeline3.sh still adds the per-step recorder and the profiler hook.

# pipeline3 diagnostics
P_ENV_X+=(VPD_STEPLOG_DIR=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-base-1123T1746/steplog-p) D_ENV_X+=(VPD_STEPLOG_DIR=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-base-1123T1746/steplog-d)
P_ARGS_X+=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-base-1123T1746/trace-p","torch_profiler_with_stack":false}')
D_ARGS_X+=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/pd1p1d-c704-base-1123T1746/trace-d","torch_profiler_with_stack":false}')
