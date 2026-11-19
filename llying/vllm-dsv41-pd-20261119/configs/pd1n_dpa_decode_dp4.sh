# One-node P/D with DP attention, decode leg: DP4 on GPUs 4-7 (TP1 per rank, MoE tensor-parallel across the ranks, no
# EP), Engram tables in host memory, no --monokernel, no prefix caching or KV events; DSpark 5; Mooncake consumer.
# INFERA_VLLM_DP_ENGINE_LB=1 registers the engine as one worker and lets vLLM's DP balancer pick the rank (the router's
# input-token booking left GLM-5.2 decode ranks unevenly full; vllm_glm52_agentx m14).
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-30001}
source "$W/configs/agg_tp4.sh"
source "$W/configs/argtools.sh"
args_drop --monokernel
args_set --tensor-parallel-size 1
args_set --data-parallel-size 4
args_set --data-parallel-rpc-port 29620
args_set --engram-config '{"cpu_offload":true}'
VLLM_ARGS+=(--no-enable-prefix-caching --no-enable-kv-events --kv-events off)
# SPEC_GOLDEN=1: InferenceX's AgentX throughput acceptance on the generation role (golden length 3.51, DSpark 5,
# thinking on); the prefill keeps block rejection, as infx/srt_slurm/synthetic_acceptance.py sets it.
if [ "${SPEC_GOLDEN:-0}" = 1 ]; then
    args_set --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
fi
WORKER_ENV+=(VLLM_PORT=41000 VLLM_DP_MASTER_PORT=29531 INFERA_VLLM_DP_ENGINE_LB=1)
KV_ROLE=kv_consumer BOOT_PORT=${BOOT_PORT:-28399}
source "$W/configs/pd_mooncake_common.sh"
