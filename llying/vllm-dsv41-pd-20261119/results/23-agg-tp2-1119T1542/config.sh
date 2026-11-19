# The InferenceX MI355X vLLM AgentX main curve: TP2 (2 GPUs per server, concurrency 1-64) with the Engram tables in
# pinned host memory, as its override_tp2_c64 (1024-row graph capture, 16384-token chunks); the base args are
# agg_tp4.sh's.
GPUS=${GPUS:-0,1}
PORT=${PORT:-30000}
source "$W/configs/agg_tp4.sh"
for i in "${!VLLM_ARGS[@]}"; do
    case "${VLLM_ARGS[$i]}" in
    --tensor-parallel-size) VLLM_ARGS[$((i + 1))]=2 ;;
    --engram-config) VLLM_ARGS[$((i + 1))]='{"cpu_offload":true}' ;;
    esac
done
