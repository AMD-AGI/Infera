# One DP2 engine (TP1 per rank, MoE tensor-parallel over the 2 ranks, Engram in host memory, no --monokernel), the
# engine shape of InferenceX's top MI355X vLLM point (dsv41flash mi355x-fp4-mtp agentic.yaml override_dp2_umbp_c384:
# 2 GPUs, c384). ENG (0-3) picks the GPU pair and every port, so up to four engines share one container.
#   ROLE        prefill | decode | agg
#   KV_DTYPE    --kv-cache-dtype (InferenceX DP2: fp8_e4m3, per-tensor fp8 rows, vllm#66036; must match on P and D)
#   PSI         --prefill-schedule-interval (InferenceX DP2: 11; B300 P/D: 2)
#   UTIL, MAXSEQS, LONG_PREFILL, EXTRA_ARGS; SPEC_K / GOLDEN_AL via spec_k.sh (GOLDEN_AL unset: block rejection)
#   UMBP=1      agg only: UMBPStoreConnector to the standalone server at $UMBP_SOCK (DRAM KV store)
#   STACK_DUMP_DIR  vLLM's stall watchdog
ENG=${ENG:?ENG}
GPUS=${GPUS:-$((2 * ENG)),$((2 * ENG + 1))}
PORT=${PORT:-$((30000 + ENG))}
source "$W/configs/agg_tp4.sh"
source "$W/configs/argtools.sh"
args_drop --monokernel
args_set --tensor-parallel-size 1
args_set --data-parallel-size 2
args_set --data-parallel-rpc-port $((29610 + 2 * ENG))
args_set --engram-config '{"cpu_offload":true}'
[ -z "${KV_DTYPE:-}" ] || args_set --kv-cache-dtype "$KV_DTYPE"
[ -z "${PSI:-}" ] || args_set --prefill-schedule-interval "$PSI"
[ -z "${UTIL:-}" ] || args_set --gpu-memory-utilization "$UTIL"
[ -z "${MAXSEQS:-}" ] || args_set --max-num-seqs "$MAXSEQS"
[ -z "${LONG_PREFILL:-}" ] || args_set --long-prefill-token-threshold "$LONG_PREFILL"
source "$W/configs/spec_k.sh"
[ -z "${STACK_DUMP_DIR:-}" ] || VLLM_ARGS+=(--watchdog-config "{\"dump_dir\":\"$STACK_DUMP_DIR\",\"timeout\":60}")
# shellcheck disable=SC2206
[ -z "${EXTRA_ARGS:-}" ] || VLLM_ARGS+=($EXTRA_ARGS)
WORKER_ENV+=(VLLM_PORT=$((40000 + 2000 * ENG)) VLLM_DP_MASTER_PORT=$((29511 + 2 * ENG)))
# kf fix (11-22, run kfp-a): INFERA_VLLM_DP_ENGINE_LB=1 only on decode, as llying's v2 configs. On a prefill engine it makes
# infera register dp_size=None, so the router sends the base engine_id while Mooncake registers <id>_dp<rank> -> every KV
# pull fails "remote engine_id ... not found from bootstrap server".
case "${ROLE:?ROLE}" in
    prefill)
        KV_ROLE=kv_producer BOOT_PORT=$((28390 + ENG))
        source "$W/configs/pd_mooncake_common.sh" ;;
    decode)
        WORKER_ENV+=(INFERA_VLLM_DP_ENGINE_LB=1)
        VLLM_ARGS+=(--no-enable-prefix-caching --no-enable-kv-events --kv-events off)
        KV_ROLE=kv_consumer BOOT_PORT=$((28390 + ENG))
        source "$W/configs/pd_mooncake_common.sh" ;;
    agg)
        if [ "${UMBP:-0}" = 1 ]; then
            VLLM_ARGS+=(--kv-transfer-config "{\"kv_connector\":\"UMBPStoreConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"mode\":\"standalone\",\"endpoint\":\"unix://${UMBP_SOCK:?UMBP_SOCK}\"}}")
        fi ;;
    *) echo "unknown ROLE $ROLE" >&2; exit 2 ;;
esac
