#!/usr/bin/env bash
# Diagnose the one-node Mooncake P/D hang of 20-pd1n-mooncake-1119T1215 (decode pulls wait for "P side ready" until
# VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT: GSM8K 2/200, 16k long-context 26/64, random 8k/1k 0/80). Same deployment, with
# the pull timeout at 120 s, DEBUG logging on the prefill (MooncakeConnector request_finished records each prefill
# leg's finish status) and VLLM_SERVER_DEV_MODE=1 (the Rust kv-aware router flushes the prefix cache to anchor the
# KV-event chain). A: Rust router, concurrent legs. B: Python router with INFERA_VLLM_MOONCAKE_SERIAL=1 (decode leg
# posted only after the prefill answered). Same probe set in both.
set -uo pipefail
RUN=21-pd1n-hang-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
MR_MODE=${MR_MODE:-$(rdma_mr_mode)}
say "start on $(hostname -s), job $SLURM_JOB_ID, amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null), MR_MODE=$MR_MODE"
node_state before
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
remove_ours
trap cleanup EXIT
start_guard
DBG="VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=120 VLLM_SERVER_DEV_MODE=1"
pd_deploy "$W/configs/pd1n_mc_prefill_tp4.sh" "$W/configs/pd1n_mc_decode_tp4.sh" "$DBG VLLM_LOGGING_LEVEL=DEBUG" "$DBG" || exit 1

router_up router-A ROUTER_BACKEND=rust ROUTER_POLICY=round-robin || { say "router A failed"; exit 1; }
probe_set A-rust-concurrent
sed 's/\x1b\[[0-9;]*m//g' "$R/prefill.log" | grep -a "MooncakeConnector request_finished" > "$R/A-request-finished.txt"
say "A prefill request_finished by status: $(grep -oE 'request_status=[A-Z_a-z.]+' "$R/A-request-finished.txt" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"

router_up router-B ROUTER_BACKEND=python ROUTER_POLICY=round-robin INFERA_VLLM_MOONCAKE_SERIAL=1 || { say "router B failed"; exit 1; }
probe_set B-python-serial
sed 's/\x1b\[[0-9;]*m//g' "$R/prefill.log" | grep -a "MooncakeConnector request_finished" > "$R/AB-request-finished.txt"
say "A+B prefill request_finished by status: $(grep -oE 'request_status=[A-Z_a-z.]+' "$R/AB-request-finished.txt" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
node_state after
say "end"
