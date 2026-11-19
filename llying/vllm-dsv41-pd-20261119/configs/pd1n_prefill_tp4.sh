# One-node P/D, prefill leg: the agg_tp4 baseline on GPUs 0-3 as a MoRIIO producer (WRITE mode: prefill pushes KV).
# host_ip pins MoRIIO's handshake/notify address to the advertised worker address the router puts in the request id
# (infera otherwise sets VLLM_HOST_IP to an RDMA-rail IP). Python frontend: the router passes the MoRIIO peer
# addresses in X-Request-Id, which vLLM's Python OpenAI server honours.
GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-30000}
VLLM_USE_RUST_FRONTEND=${VLLM_USE_RUST_FRONTEND:-0}
source "$W/configs/agg_tp4.sh"
VLLM_ARGS+=(--kv-transfer-config "{\"kv_connector\":\"MoRIIOConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"http_port\":\"$PORT\",\"handshake_port\":\"16300\",\"notify_port\":\"16400\",\"tp_size\":4,\"host_ip\":\"$(node_ip)\"}}")
