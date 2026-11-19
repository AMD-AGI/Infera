# One-node P/D, decode leg: the agg_tp4 baseline on GPUs 4-7 as a MoRIIO consumer. See pd1n_prefill_tp4.sh.
GPUS=${GPUS:-4,5,6,7}
PORT=${PORT:-30001}
VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-0}
source "$W/configs/agg_tp4.sh"
VLLM_ARGS+=(--kv-transfer-config "{\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"http_port\":\"$PORT\",\"handshake_port\":\"16500\",\"notify_port\":\"16600\",\"tp_size\":4,\"host_ip\":\"$(node_ip)\"}}")
