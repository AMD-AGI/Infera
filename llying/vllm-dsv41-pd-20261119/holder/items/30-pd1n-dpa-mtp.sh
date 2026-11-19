#!/usr/bin/env bash
# One-node P/D with DP attention and DSpark (MTP): prefill DP2 on GPUs 0-1 + decode DP4 on GPUs 4-7 (configs
# pd1n_dpa_*), Mooncake with one NIC per GPU, the vLLM STOPPED-prefill fix mounted (vllm-overlay), Rust router.
# Falls back to a TP2 prefill if the DP2 prefill does not come up. Then: smoke, GSM8K / 16k probes, random 8k/1k at
# c16 / c64, and one InferenceX AgentX fast point (CONC, default 32) behind the Rust kv-aware router with the SGLang
# recipe's P/D flags, if the job has the time left.
set -uo pipefail
RUN=30-pd1n-dpa-mtp-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
MR_MODE=${MR_MODE:-$(rdma_mr_mode)}
SITE=/usr/local/lib/python3.12/dist-packages
left_min() {  # minutes left in this job (squeue %L: [D-]HH:MM:SS or MM:SS)
    local l d=0 h=0 m=0 t
    l="$(squeue -h -j "${SLURM_JOB_ID:?}" -o %L 2>/dev/null)"
    [[ "$l" == *-* ]] && { d=${l%%-*}; l=${l#*-}; }
    IFS=: read -ra t <<<"$l"
    if ((${#t[@]} == 3)); then h=${t[0]} m=${t[1]}; elif ((${#t[@]} == 2)); then m=${t[0]}; fi
    echo $((10#$d * 1440 + 10#$h * 60 + 10#$m))
}
say "start on $(hostname -s), job $SLURM_JOB_ID ($(left_min) min left), amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null), MR_MODE=$MR_MODE"
node_state before
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
remove_ours
trap cleanup EXIT
start_guard
export EXTRA_DOCKER_ARGS="-v $W/vllm-overlay/mooncake_connector.py:$SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:ro"
PE="VLLM_SERVER_DEV_MODE=1"
PCFG=$W/configs/pd1n_dpa_prefill_dp2.sh DCFG=$W/configs/pd1n_dpa_decode_dp4.sh P_DESC="DP2 (GPUs 0-1)"
if ! pd_deploy "$PCFG" "$DCFG" "$PE" "" MC_TOPO=pergpu; then
    collect_logs; mkdir -p "$R/attempt-dp2" && mv "$R"/prefill.log "$R"/decode.log "$R"/docker-*.log "$R/attempt-dp2/" 2>/dev/null
    remove_ours
    PCFG=$W/configs/pd1n_tp2_prefill.sh P_DESC="TP2 (GPUs 0-1)"
    say "DP2 prefill deployment failed; retrying with a TP2 prefill"
    pd_deploy "$PCFG" "$DCFG" "$PE" "" MC_TOPO=pergpu || exit 1
fi
say "deployed: prefill $P_DESC + decode DP4 (GPUs 4-7); KV $(grep -ahoE 'GPU KV cache size: [0-9,]+ tokens' "$R/prefill.log" "$R/decode.log" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
rocm-smi --showmeminfo vram --csv > "$R/vram-ready.csv" 2>&1
router_up router ROUTER_BACKEND=rust ROUTER_POLICY=round-robin || { say "router failed"; exit 1; }
curl -s "http://127.0.0.1:$ROUTER_PORT/v1/workers" > "$R/workers.json"
say "router workers: $(python3 -c 'import json,sys; print(sorted((w.get("disagg_mode"), w.get("url"), w.get("dp_size")) for w in json.load(open(sys.argv[1]))["workers"]))' "$R/workers.json" 2>&1 | head -c 400)"
chat "What is the capital of France? Answer with one word." 512 > "$R/smoke1.json"
say "smoke1: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(repr((d["choices"][0]["message"].get("content") or "")[:100]), d.get("usage"))' "$R/smoke1.json" 2>&1 | head -c 400)"
probe_set dpa
for c in 16 64; do
    docker exec "${CTR_PREFIX}eng" timeout 1200 vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
        --base-url "http://127.0.0.1:$ROUTER_PORT" --model "$SERVED" --tokenizer "$MODEL" --trust-remote-code \
        --dataset-name random --random-input-len 8192 --random-output-len 1024 --ignore-eos \
        --num-prompts $((c * 4)) --max-concurrency "$c" --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir "$R" --result-filename "bench-c$c.json" > "$R/bench-c$c.log" 2>&1
    say "bench c$c: $(grep -aE 'Successful requests|Total token throughput|Mean TTFT|Mean TPOT' "$R/bench-c$c.log" | tr -s ' ' | tr '\n' ' ')"
done
say "spec decode (decode leg): $(grep -aE 'SpecDecoding metrics' "$R/decode.log" | tail -1 | grep -oE 'Mean acceptance length: [0-9.]+')"

# AgentX fast needs ~45 min (client venv, warmup, the 1200 s profile, scoring).
if (( $(left_min) < 55 )); then
    say "AgentX fast skipped: $(left_min) min left in job $SLURM_JOB_ID"
else
    router_up router-kvaware ROUTER_BACKEND=rust ROUTER_POLICY=kv-aware \
        ROUTER_EXTRA_ARGS="--kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on" \
        || { say "kv-aware router failed"; exit 1; }
    pg=2; [[ "$P_DESC" == TP2* ]] && pdpa=false || pdpa=true
    CONC=${CONC:-32}
    say "AgentX fast c$CONC starting ($(left_min) min left)"
    CONC=$CONC FAST=1 PREFILL_GPUS=$pg DECODE_GPUS=4 PREFILL_DP_ATTN=$pdpa DECODE_DP_ATTN=true \
        timeout $(( ($(left_min) - 5) * 60 )) bash "$W/scripts/agentx_fast.sh" "$R/agentx-c$CONC"
    rc=$?
    say "AgentX fast c$CONC rc=$rc: $(grep -aE 'tok/s/gpu|Output token throughput|Request throughput|failed|ERROR' "$R/agentx-c$CONC/runner.log" | tail -6 | tr -s ' ' | tr '\n' ' ' | head -c 900)"
    ls "$R/agentx-c$CONC" > "$R/agentx-c$CONC.files" 2>&1
fi
node_state after
say "end"
