# One-node P/D over Mooncake, prefill leg: the agg_tp4 baseline on GPUs 0-3 as the KV producer. It keeps DSpark:
# the draft layers' KV is produced at prefill and moves with the rest.
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-30000}
source "$W/configs/agg_tp4.sh"
KV_ROLE=kv_producer BOOT_PORT=${BOOT_PORT:-28398}
source "$W/configs/pd_mooncake_common.sh"
