# One-node P/D with DP attention, prefill leg: DP2 on GPUs 0-1 (TP1 per rank, MoE tensor-parallel across the ranks, no
# EP), Engram tables in host memory and no --monokernel, as InferenceX's MI355X DP2 points; DSpark 5; Mooncake producer.
# Distinct DP RPC / master / scan ports per engine (DP_RPC_PORT, DP_MASTER_PORT, SCAN_PORT): every engine shares the
# container.
GPUS=${GPUS:-0,1}
PORT=${PORT:-30000}
source "$W/configs/agg_tp4.sh"
source "$W/configs/argtools.sh"
args_drop --monokernel
args_set --tensor-parallel-size 1
args_set --data-parallel-size 2
args_set --data-parallel-rpc-port ${DP_RPC_PORT:-29610}
args_set --engram-config '{"cpu_offload":true}'
WORKER_ENV+=(VLLM_PORT=${SCAN_PORT:-40000} VLLM_DP_MASTER_PORT=${DP_MASTER_PORT:-29511})
KV_ROLE=kv_producer BOOT_PORT=${BOOT_PORT:-28398}
source "$W/configs/pd_mooncake_common.sh"
