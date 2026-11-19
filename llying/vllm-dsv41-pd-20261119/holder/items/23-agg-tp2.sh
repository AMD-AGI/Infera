#!/usr/bin/env bash
# The InferenceX MI355X vLLM main-curve layout (TP2, Engram tables in host memory) behind the infera router on GPUs
# 0-1, then smoke, GSM8K probe, 16k long context, a short throughput sanity check, and a Rust kv-aware router pass.
set -uo pipefail
RUN=23-agg-tp2-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
CFG=${CFG:-$W/configs/agg_tp2.sh}
say "start on $(hostname -s) ($(node_ip)), job $SLURM_JOB_ID, config $CFG"
node_state before
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
remove_ours
r="$(bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/clear_foreign_containers.sh "$SLURM_JOB_ID" 2>&1)"
[ -z "$r" ] || say "foreign containers: $r"
trap cleanup EXIT
start_guard
start_etcd || { say "etcd failed"; exit 1; }
start_engine_ctr eng || { say "engine container / infera install failed: $(tail -3 "$R/install-eng.log" | tr '\n' ' ')"; exit 1; }
cp "$CFG" "$R/config.sh"
exec_bg eng router env ROUTER_POLICY=round-robin bash "$W/scripts/ctr_router.sh"
t0=$SECONDS
exec_bg eng worker0 bash "$W/scripts/ctr_worker.sh" "$CFG"
wait_router || say "router not healthy yet: $(tail -3 "$R/router.log" | tr '\n' ' ')"
if ! wait_worker worker0 30000 7200; then
    say "worker failed after $((SECONDS - t0)) s: $(grep -aE 'Error|error:|Traceback' "$R/worker0.log" | tail -5 | tr '\n' ' ')"
    exit 1
fi
say "worker ready in $((SECONDS - t0)) s; $(grep -aoE 'Available KV cache memory: [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [^ ]+ tokens per request: [0-9.]+x' "$R/worker0.log" | sort -u | tr '\n' ' ')"
rocm-smi --showmeminfo vram --csv > "$R/vram-ready.csv" 2>&1
sleep 10
curl -s "http://127.0.0.1:$ROUTER_PORT/v1/workers" > "$R/workers.json"
curl -s "http://127.0.0.1:$ROUTER_PORT/v1/models" > "$R/models.json"
say "router workers: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); w=d.get("workers", d) if isinstance(d, dict) else d; print(len(w), "worker(s)", [x.get("worker_id", x.get("url")) for x in w] if isinstance(w, list) else str(d)[:300])' "$R/workers.json" 2>&1)"

# Smoke: non-streaming and streaming chat through the router.
chat "What is the capital of France? Answer with one word." 512 > "$R/smoke1.json"
say "smoke1: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; print(repr((m.get("content") or "")[:200]), "| reasoning", len(m.get("reasoning") or m.get("reasoning_content") or ""), "chars | usage", d.get("usage"))' "$R/smoke1.json" 2>&1 | head -c 600)"
chat "Write a haiku about GPUs." 256 1 > "$R/smoke2.sse"
say "smoke2 (stream): $(grep -c '^data:' "$R/smoke2.sse") SSE events, done=$(grep -c 'data: \[DONE\]' "$R/smoke2.sse")"

# Correctness: 200 GSM8K questions, temperature 0.
docker exec "${CTR_PREFIX}eng" timeout 3600 python3 "$W/scripts/gsm8k_probe.py" "$W/data/gsm8k_probe_prompts.jsonl" \
    "http://127.0.0.1:$ROUTER_PORT" "$SERVED" "$R/gsm8k.jsonl" --limit 200 --concurrency 32 --max-tokens 8192 > "$R/gsm8k.log" 2>&1
say "gsm8k: $(tail -1 "$R/gsm8k.log")"
docker exec "${CTR_PREFIX}eng" timeout 3600 python3 "$W/scripts/longctx_probe.py" "$W/data/gsm8k_probe_prompts.jsonl" \
    "http://127.0.0.1:$ROUTER_PORT" "$SERVED" "$R/longctx16k.jsonl" --ctx-tokens 16000 --n 64 --concurrency 16 > "$R/longctx16k.log" 2>&1
say "longctx 16k: $(grep -aE '^(needle|gsm8k): n=' "$R/longctx16k.log" | tr '\n' ' ')"

# Throughput sanity (not the AgentX metric): random 8k/1k at concurrency 4 and 16.
for c in 4 16; do
    docker exec "${CTR_PREFIX}eng" timeout 2400 vllm bench serve --backend openai-chat --endpoint /v1/chat/completions \
        --base-url "http://127.0.0.1:$ROUTER_PORT" --model "$SERVED" --tokenizer "$MODEL" --trust-remote-code \
        --dataset-name random --random-input-len 8192 --random-output-len 1024 --ignore-eos \
        --num-prompts $((c * 4)) --max-concurrency "$c" --percentile-metrics ttft,tpot,itl,e2el \
        --save-result --result-dir "$R" --result-filename "bench-c$c.json" > "$R/bench-c$c.log" 2>&1
    say "bench c$c: $(grep -aE 'Successful requests|Output token throughput|Total token throughput|Mean TTFT|Mean TPOT' "$R/bench-c$c.log" | tr -s ' ' | tr '\n' ' ')"
done
say "spec decode: $(grep -aE 'SpecDecoding metrics|acceptance length' "$R/worker0.log" | tail -2 | tr -s ' ' | tr '\n' ' ' | head -c 500)"

# kv-aware router pass: restart only the router, as the Rust backend (the Python one tokenizes every prompt in its
# event loop for kv-aware, U15), and check that the KV-event path holds up with DSv4.1.
docker exec "${CTR_PREFIX}eng" pkill -f "infera-router|infera.server" ; sleep 5
exec_bg eng router-kvaware env ROUTER_BACKEND=rust ROUTER_POLICY=kv-aware bash "$W/scripts/ctr_router.sh"
if wait_router; then
    sleep 20
    chat "What is the capital of France? Answer with one word." 512 > "$R/smoke-kvaware.json"
    docker exec "${CTR_PREFIX}eng" timeout 1800 python3 "$W/scripts/gsm8k_probe.py" "$W/data/gsm8k_probe_prompts.jsonl" \
        "http://127.0.0.1:$ROUTER_PORT" "$SERVED" "$R/gsm8k-kvaware.jsonl" --limit 32 --concurrency 16 --max-tokens 8192 > "$R/gsm8k-kvaware.log" 2>&1
    say "kv-aware: $(tail -1 "$R/gsm8k-kvaware.log"); router warnings: $(grep -aciE 'warn|error|exception' "$R/router-kvaware.log")"
else
    say "kv-aware router did not come up: $(tail -3 "$R/router-kvaware.log" | tr '\n' ' ')"
fi
node_state after
say "end"
