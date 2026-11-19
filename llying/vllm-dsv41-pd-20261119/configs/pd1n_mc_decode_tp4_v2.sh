# One-node P/D decode leg as one TP4 engine on GPUs 4-7 (no DP ranks): agg_tp4 without --monokernel, no prefix caching
# or KV events, Mooncake consumer. Knobs (off by default): GOLDEN_AL (synthetic acceptance with InferenceX's golden
# length; unset = block rejection, for correctness probes), SPEC_K, D_UTIL, D_MAXSEQS, D_ENGRAM_HOST, D_EXTRA_ARGS,
# STACK_DUMP_DIR.
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-30001}
source "$W/configs/agg_tp4.sh"
source "$W/configs/argtools.sh"
args_drop --monokernel
source "$W/configs/spec_k.sh"
[ -z "${D_UTIL:-}" ] || args_set --gpu-memory-utilization "$D_UTIL"
[ -z "${D_MAXSEQS:-}" ] || args_set --max-num-seqs "$D_MAXSEQS"
# D_ENGRAM_HOST=1: Engram tables in host memory (agg_tp4 keeps them in HBM, ~34 GiB per rank of KV space).
[ -z "${D_ENGRAM_HOST:-}" ] || args_set --engram-config '{"cpu_offload":true}'
VLLM_ARGS+=(--no-enable-prefix-caching --no-enable-kv-events --kv-events off)
[ -z "${STACK_DUMP_DIR:-}" ] || VLLM_ARGS+=(--watchdog-config "{\"dump_dir\":\"$STACK_DUMP_DIR\",\"timeout\":60}")
# shellcheck disable=SC2206
[ -z "${D_EXTRA_ARGS:-}" ] || VLLM_ARGS+=($D_EXTRA_ARGS)
KV_ROLE=kv_consumer BOOT_PORT=${BOOT_PORT:-28399}
source "$W/configs/pd_mooncake_common.sh"
