#!/usr/bin/env bash
# One-node P/D bring-up: infera router (Python backend; 0.3.3's Rust router has no vLLM MoRIIO protocol) + a TP4
# prefill on GPUs 0-3 and a TP4 decode on GPUs 4-7, KV over MoRIIO (RDMA, WRITE mode). Checks pairing, transfer
# correctness (GSM8K and 16k-token needle/GSM8K against the agg run), and a short throughput sanity check.
set -uo pipefail
RUN=20-pd1n-moriio-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
PCFG=${PCFG:-$W/configs/pd1n_prefill_tp4.sh} DCFG=${DCFG:-$W/configs/pd1n_decode_tp4.sh}
say "start on $(hostname -s) ($(node_ip)), job $SLURM_JOB_ID, $PCFG + $DCFG"
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
remove_ours
trap cleanup EXIT
start_guard
start_etcd || { say "etcd failed"; exit 1; }
start_engine_ctr eng || { say "engine container / infera install failed"; exit 1; }
fix_libionic eng > "$R/libionic.log" 2>&1 || say "no active RDMA port in the container: $(tail -2 "$R/libionic.log" | tr '\n' ' ')"
docker exec "${CTR_PREFIX}eng" bash -c 'ibv_devinfo 2>&1 | grep -E "hca_id|state|link_layer"' > "$R/ibv.txt" 2>&1
cp "$PCFG" "$R/prefill-config.sh"; cp "$DCFG" "$R/decode-config.sh"
exec_bg eng router env ROUTER_POLICY=round-robin bash "$W/scripts/ctr_router.sh"
t0=$SECONDS
exec_bg eng prefill bash "$W/scripts/ctr_worker.sh" "$PCFG"
exec_bg eng decode bash "$W/scripts/ctr_worker.sh" "$DCFG"
wait_router || say "router not healthy yet"
wait_worker prefill 30000 7200 & wp=$!
wait_worker decode 30001 7200 & wd=$!
wait $wp; rp=$?; wait $wd; rd=$?
if ((rp || rd)); then
    say "workers failed after $((SECONDS - t0)) s (prefill rc=$rp decode rc=$rd)"
    say "prefill: $(grep -aE 'Error|error:|Traceback|assert' "$R/prefill.log" | tail -4 | tr '\n' ' ' | head -c 800)"
    say "decode: $(grep -aE 'Error|error:|Traceback|assert' "$R/decode.log" | tail -4 | tr '\n' ' ' | head -c 800)"
    exit 1
fi
say "workers ready in $((SECONDS - t0)) s"
sleep 10
curl -s "http://127.0.0.1:$ROUTER_PORT/v1/workers" > "$R/workers.json"
say "router workers: $(head -c 700 "$R/workers.json")"

chat "What is the capital of France? Answer with one word." 512 > "$R/smoke1.json"
say "smoke1: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; print(repr((m.get("content") or "")[:200]), "| usage", d.get("usage"))' "$R/smoke1.json" 2>&1 | head -c 600)"
chat "Write a haiku about GPUs." 256 1 > "$R/smoke2.sse"
say "smoke2 (stream): $(grep -c '^data:' "$R/smoke2.sse") SSE events, done=$(grep -c 'data: \[DONE\]' "$R/smoke2.sse")"
say "moriio log lines: prefill $(grep -aci moriio "$R/prefill.log"), decode $(grep -aci moriio "$R/decode.log"); errors: $(grep -aiE 'moriio.*(error|fail|timeout)' "$R/prefill.log" "$R/decode.log" | tail -2 | tr '\n' ' ' | head -c 400)"

docker exec "${CTR_PREFIX}eng" timeout 3600 python3 "$W/scripts/gsm8k_probe.py" "$W/data/gsm8k_probe_prompts.jsonl" \
    "http://127.0.0.1:$ROUTER_PORT" "$SERVED" "$R/gsm8k.jsonl" --limit 200 --concurrency 32 --max-tokens 8192 > "$R/gsm8k.log" 2>&1
say "gsm8k: $(tail -1 "$R/gsm8k.log")"
docker exec "${CTR_PREFIX}eng" timeout 3600 python3 "$W/scripts/longctx_probe.py" "$W/data/gsm8k_probe_prompts.jsonl" \
    "http://127.0.0.1:$ROUTER_PORT" "$SERVED" "$R/longctx16k.jsonl" --ctx-tokens 16000 --n 64 --concurrency 16 > "$R/longctx16k.log" 2>&1
say "longctx 16k: $(grep -aE '^(needle|gsm8k): n=' "$R/longctx16k.log" | tr '\n' ' ')"

for c in 4 16; do
    docker exec "${CTR_PREFIX}eng" timeout 2400 vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
        --base-url "http://127.0.0.1:$ROUTER_PORT" --model "$SERVED" --tokenizer "$MODEL" --trust-remote-code \
        --dataset-name random --random-input-len 8192 --random-output-len 1024 --ignore-eos \
        --num-prompts $((c * 4)) --max-concurrency "$c" --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir "$R" --result-filename "bench-c$c.json" > "$R/bench-c$c.log" 2>&1
    say "bench c$c: $(grep -aE 'Successful requests|Output token throughput|Total token throughput|Mean TTFT|Mean TPOT' "$R/bench-c$c.log" | tr -s ' ' | tr '\n' ' ')"
done
say "spec decode (decode leg): $(grep -aE 'SpecDecoding metrics|acceptance length' "$R/decode.log" | tail -1 | tr -s ' ' | head -c 400)"
say "end"
