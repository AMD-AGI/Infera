#!/usr/bin/env bash
# Fix check for the one-node Mooncake P/D hang (21-pd1n-hang): the prefill leg finished FINISHED_STOPPED (its one
# sampled token was a stop token; DSv4.1 producers drop the last prompt token and sample a guess at it) and vLLM's
# Mooncake producer hands KV over only on FINISHED_LENGTH_CAPPED. The infera overlay's prefill_single_token now sends
# min_tokens=1 and ignore_eos=true on the prefill leg. Python router (concurrent legs; 0.3.3's Rust router builds its
# own prefill body), same probe set as 21; expect no P-side-ready timeouts and only LENGTH_CAPPED prefill finishes.
set -uo pipefail
RUN=21b-pd1n-p-min-tokens-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
MR_MODE=${MR_MODE:-$(rdma_mr_mode)}
say "start on $(hostname -s), job $SLURM_JOB_ID, MR_MODE=$MR_MODE"
node_state before
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
remove_ours
trap cleanup EXIT
start_guard
DBG="VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=120 VLLM_SERVER_DEV_MODE=1"
pd_deploy "$W/configs/pd1n_mc_prefill_tp4.sh" "$W/configs/pd1n_mc_decode_tp4.sh" "$DBG VLLM_LOGGING_LEVEL=DEBUG" "$DBG" || exit 1
docker exec "${CTR_PREFIX}eng" grep -c "min_tokens" /usr/local/lib/python3.12/dist-packages/infera/router/disagg_protocols/base.py > "$R/overlay-check.txt" 2>&1
router_up router-C ROUTER_BACKEND=python ROUTER_POLICY=round-robin || { say "router C failed"; exit 1; }
probe_set C-python-concurrent-mintok
sed 's/\x1b\[[0-9;]*m//g' "$R/prefill.log" | grep -a "MooncakeConnector request_finished" > "$R/C-request-finished.txt"
say "C prefill request_finished by status: $(grep -oE 'request_status=[A-Z_a-z.]+' "$R/C-request-finished.txt" | sort | uniq -c | tr -s ' ' | tr '\n' ';') (overlay min_tokens lines: $(cat "$R/overlay-check.txt"))"
node_state after
say "end"
