# arm /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/arms/r4-hipvmm.sh
# R4 plus Engram tables in HIP VMM host memory on both legs (VLLM_ROCM_ENGRAM_HIP_VMM, an opt-in in vLLM) instead of
# a /dev/shm file registered with cudaHostRegister. The /dev/shm pages are ordinary movable pages that KFD tracks as
# userptr: every migration under the 2 x 94 GiB tables (direct compaction, khugepaged) makes KFD evict the process's
# GPU queues until the whole range is revalidated, and every GPU stall in hangB0 coincided with such migrations.
# Driver-allocated VMM host memory is not migrated and is mapped with 2 MiB pages. Numerics unchanged.
# shellcheck disable=SC1091
source /home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/arms/r4-a4w4p.sh
P_ENV_X+=(VLLM_ROCM_ENGRAM_HIP_VMM=1)
D_ENV_X+=(VLLM_ROCM_ENGRAM_HIP_VMM=1)

# pipeline_hang3 diagnostics
P_ENV_X+=(VPD_STEPLOG_DIR=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/hang3-r4-hipvmm-p0-1125T0604/steplog-p) D_ENV_X+=(VPD_STEPLOG_DIR=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123/logs/hang3-r4-hipvmm-p0-1125T0604/steplog-d)
