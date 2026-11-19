# One-node P/D over Mooncake, decode leg: the agg_tp4 baseline on GPUs 4-7 as the KV consumer, without prefix caching
# or KV events (as the SGLang and B300 decodes run; the router has no cache to aim at on the decode).
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-30001}
source "$W/configs/agg_tp4.sh"
VLLM_ARGS+=(--no-enable-prefix-caching --no-enable-kv-events --kv-events off)
KV_ROLE=kv_consumer BOOT_PORT=${BOOT_PORT:-28399}
source "$W/configs/pd_mooncake_common.sh"
