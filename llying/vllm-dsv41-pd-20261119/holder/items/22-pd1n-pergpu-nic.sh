#!/usr/bin/env bash
# One-node Mooncake P/D with one NIC per GPU (MC_TOPO=pergpu: GPU k uses only ionic_k) instead of infera 0.3.3's
# all-GPUs-on-ionic_0 default. Same-node transfers then cross rails / subnets (prefill GPU k on ionic_k, decode GPU 4+k
# on ionic_4+k). Checks that QPs come up, transfers stay correct, and the KV traffic spreads over the NICs.
set -uo pipefail
RUN=22-pd1n-pergpu-nic-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
MR_MODE=${MR_MODE:-$(rdma_mr_mode)}
say "start on $(hostname -s), job $SLURM_JOB_ID, amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null), MR_MODE=$MR_MODE"
node_state before
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
remove_ours
trap cleanup EXIT
start_guard
DBG="VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=120 VLLM_SERVER_DEV_MODE=1"
pd_deploy "$W/configs/pd1n_mc_prefill_tp4.sh" "$W/configs/pd1n_mc_decode_tp4.sh" "$DBG" "$DBG" MC_TOPO=pergpu || exit 1
say "mooncake HCAs: $(grep -ahoE 'Found [0-9]+ HCAs|custom topology from: \S+|for ionic_[0-9]+' "$R/prefill.log" "$R/decode.log" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
router_up router ROUTER_BACKEND=rust ROUTER_POLICY=round-robin || { say "router failed"; exit 1; }
tx0="$(nic_tx)"
probe_set pergpu
docker exec "${CTR_PREFIX}eng" timeout 1200 vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
    --base-url "http://127.0.0.1:$ROUTER_PORT" --model "$SERVED" --tokenizer "$MODEL" --trust-remote-code \
    --dataset-name random --random-input-len 8192 --random-output-len 1024 --ignore-eos \
    --num-prompts 64 --max-concurrency 16 --percentile-metrics ttft,tpot,itl,e2el \
    --save-result --result-dir "$R" --result-filename bench-c16.json > "$R/bench-c16.log" 2>&1
say "bench c16: $(grep -aE 'Successful requests|Total token throughput|Mean TTFT|Mean TPOT' "$R/bench-c16.log" | tr -s ' ' | tr '\n' ' ')"
tx1="$(nic_tx)"
say "NIC tx bytes/4 delta: $(python3 -c 'import sys
a = dict(x.split("=") for x in sys.argv[1].split()); b = dict(x.split("=") for x in sys.argv[2].split())
print(" ".join(f"{k}={(int(b[k]) - int(a[k])) * 4 / 2**30:.1f}GiB" for k in sorted(a) if a[k] and b.get(k)))' "$tx0" "$tx1")"
say "QP / handshake errors: $(grep -aciE 'RTR|\[110\]|handshake.*fail|Re-establish|ERR_' "$R/prefill.log" "$R/decode.log" | tr '\n' ' ')"
node_state after
say "end"
