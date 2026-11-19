# Fallback prefill leg for the one-node DP-attention P/D: TP2 on GPUs 0-1 (agg_tp2.sh, Engram tables in host memory);
# Mooncake producer. Used when the DP2 prefill does not come up.
GPUS=${GPUS:-0,1}
PORT=${PORT:-30000}
source "$W/configs/agg_tp2.sh"
WORKER_ENV+=(VLLM_PORT=40000)
KV_ROLE=kv_producer BOOT_PORT=${BOOT_PORT:-28398}
source "$W/configs/pd_mooncake_common.sh"
