# One-node P/D decode leg as one TP4 engine on GPUs 4-7 (no DP ranks, so no DP dummy batches): the agg_tp4 baseline
# without --monokernel (its 48-row launch is for small batches), no prefix caching or KV events, Mooncake consumer, and
# DSpark 5 verified with InferenceX's golden acceptance length 3.51. Each TP rank pulls the full (replicated) KV of a
# request from the prefill, 4x the transfer volume of the DP4 decode.
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-30001}
source "$W/configs/agg_tp4.sh"
source "$W/configs/argtools.sh"
args_drop --monokernel
args_set --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51,"enable_adaptive_verification":false}'
VLLM_ARGS+=(--no-enable-prefix-caching --no-enable-kv-events --kv-events off)
[ -z "${STACK_DUMP_DIR:-}" ] || VLLM_ARGS+=(--watchdog-config "{\"dump_dir\":\"$STACK_DUMP_DIR\",\"timeout\":60}")
KV_ROLE=kv_consumer BOOT_PORT=${BOOT_PORT:-28399}
source "$W/configs/pd_mooncake_common.sh"
