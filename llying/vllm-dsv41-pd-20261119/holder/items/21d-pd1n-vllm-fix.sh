#!/usr/bin/env bash
# End-to-end check of the vLLM fix for the one-node Mooncake P/D hang: the patched mooncake_connector.py (producer
# sends KV on FINISHED_STOPPED too; vllm-overlay/SOURCE) mounted over the image's, with the stock Rust router (whose
# prefill body has no min_tokens / ignore_eos), concurrent legs, the 21 probe set. Expect STOPPED prefill finishes to
# remain and no decode "Timeout waiting for P side ready".
set -uo pipefail
RUN=21d-pd1n-vllm-fix-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
MR_MODE=${MR_MODE:-$(rdma_mr_mode)}
SITE=/usr/local/lib/python3.12/dist-packages
say "start on $(hostname -s), job $SLURM_JOB_ID, MR_MODE=$MR_MODE, vllm overlay from $(cat "$W/vllm-overlay/SOURCE")"
node_state before
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
remove_ours
trap cleanup EXIT
start_guard
export EXTRA_DOCKER_ARGS="-v $W/vllm-overlay/mooncake_connector.py:$SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:ro"
DBG="VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=120 VLLM_SERVER_DEV_MODE=1"
pd_deploy "$W/configs/pd1n_mc_prefill_tp4.sh" "$W/configs/pd1n_mc_decode_tp4.sh" "$DBG VLLM_LOGGING_LEVEL=DEBUG" "$DBG" || exit 1
say "overlay in container: $(docker exec "${CTR_PREFIX}eng" grep -c FINISHED_STOPPED $SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py) FINISHED_STOPPED lines"
router_up router-D ROUTER_BACKEND=rust ROUTER_POLICY=round-robin || { say "router D failed"; exit 1; }
probe_set D-rust-vllmfix
sed 's/\x1b\[[0-9;]*m//g' "$R/prefill.log" | grep -a "MooncakeConnector request_finished" > "$R/D-request-finished.txt"
say "D prefill request_finished by status: $(grep -oE 'request_status=[A-Z_a-z.]+' "$R/D-request-finished.txt" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
node_state after
say "end"
