#!/usr/bin/env bash
# One-node P/D bring-up over Mooncake: infera Rust router + a TP4 prefill on GPUs 0-3 and a TP4 decode on GPUs 4-7
# (agg_tp4 baseline args on both), KV over RDMA loopback. VRAM registration follows the amdgpu version (lib.sh
# rdma_mr_mode): peer memory on 6.16, amdgpu GEM KV pools on 6.19. Checks pairing, transfer correctness
# (GSM8K, 16k needle / GSM8K, to compare with the agg run), a short throughput sanity check, then the Rust kv-aware
# router with the SGLang recipe's P/D flags.
set -uo pipefail
RUN=20-pd1n-mooncake-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
PCFG=${PCFG:-$W/configs/pd1n_mc_prefill_tp4.sh} DCFG=${DCFG:-$W/configs/pd1n_mc_decode_tp4.sh}
MR_MODE=${MR_MODE:-$(rdma_mr_mode)}
say "start on $(hostname -s) ($(node_ip)), job $SLURM_JOB_ID, amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null), MR_MODE=$MR_MODE"
node_state before
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
remove_ours
trap cleanup EXIT
start_guard
start_etcd || { say "etcd failed"; exit 1; }
start_engine_ctr eng || { say "engine container / infera install failed: $(tail -3 "$R/install-eng.log" | tr '\n' ' ')"; exit 1; }
fix_libionic eng > "$R/libionic.log" 2>&1 || say "no active RDMA port in the container: $(tail -2 "$R/libionic.log" | tr '\n' ' ')"
say "libionic: $(tail -1 "$R/libionic.log")"
GEM_SO=""
if [ "$MR_MODE" = gem ]; then
    build_gem eng > "$R/gem-build.log" 2>&1 || { say "GEM allocator build failed: $(tail -3 "$R/gem-build.log" | tr '\n' ' ')"; exit 1; }
    say "GEM allocator $GEM_SO"
fi
cp "$PCFG" "$R/prefill-config.sh"; cp "$DCFG" "$R/decode-config.sh"; cp "$W/configs/pd_mooncake_common.sh" "$R/"
exec_bg eng router env ROUTER_BACKEND=rust ROUTER_POLICY=round-robin bash "$W/scripts/ctr_router.sh"
t0=$SECONDS
exec_bg eng prefill env MR_MODE="$MR_MODE" GEM_SO="$GEM_SO" bash "$W/scripts/ctr_worker.sh" "$PCFG"
sleep 30
exec_bg eng decode env MR_MODE="$MR_MODE" GEM_SO="$GEM_SO" bash "$W/scripts/ctr_worker.sh" "$DCFG"
wait_router || say "router not healthy yet: $(tail -3 "$R/router.log" | tr '\n' ' ')"
wait_worker prefill 30000 7200 & wp=$!
wait_worker decode 30001 7200 & wd=$!
wait $wp; rp=$?; wait $wd; rd=$?
if ((rp || rd)); then
    say "workers failed after $((SECONDS - t0)) s (prefill rc=$rp decode rc=$rd)"
    say "prefill: $(grep -aE 'Error|error:|Traceback|assert' "$R/prefill.log" | tail -4 | tr '\n' ' ' | head -c 800)"
    say "decode: $(grep -aE 'Error|error:|Traceback|assert' "$R/decode.log" | tail -4 | tr '\n' ' ' | head -c 800)"
    exit 1
fi
say "workers ready in $((SECONDS - t0)) s; mooncake: $(grep -ahoE 'Found [0-9]+ HCAs|MOONCAKE_DISABLE_HIP_DMABUF[^,]*|KV cache allocated from amdgpu GEM BOs' "$R/prefill.log" "$R/decode.log" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
rocm-smi --showmeminfo vram --csv > "$R/vram-ready.csv" 2>&1
sleep 10
curl -s "http://127.0.0.1:$ROUTER_PORT/v1/workers" > "$R/workers.json"
say "router workers: $(python3 -c 'import json,sys; print(sorted((w.get("disagg_mode"), w.get("url")) for w in json.load(open(sys.argv[1]))["workers"]))' "$R/workers.json" 2>&1 | head -c 400)"

chat "What is the capital of France? Answer with one word." 512 > "$R/smoke1.json"
say "smoke1: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; print(repr((m.get("content") or "")[:200]), "| usage", d.get("usage"))' "$R/smoke1.json" 2>&1 | head -c 600)"
chat "Write a haiku about GPUs." 256 1 > "$R/smoke2.sse"
say "smoke2 (stream): $(grep -c '^data:' "$R/smoke2.sse") SSE events, done=$(grep -c 'data: \[DONE\]' "$R/smoke2.sse")"
say "mooncake errors: prefill $(grep -aciE 'transfer.*(fail|error)|-302|ERR_' "$R/prefill.log"), decode $(grep -aciE 'transfer.*(fail|error)|-302|ERR_' "$R/decode.log")"

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

# The production router: Rust, kv-aware, with the SGLang recipe's P/D flags.
docker exec "${CTR_PREFIX}eng" pkill -f "infera-router|infera.server"; sleep 5
exec_bg eng router-kvaware env ROUTER_BACKEND=rust ROUTER_POLICY=kv-aware \
    ROUTER_EXTRA_ARGS="--kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on" \
    bash "$W/scripts/ctr_router.sh"
if wait_router; then
    sleep 20
    docker exec "${CTR_PREFIX}eng" timeout 1800 python3 "$W/scripts/gsm8k_probe.py" "$W/data/gsm8k_probe_prompts.jsonl" \
        "http://127.0.0.1:$ROUTER_PORT" "$SERVED" "$R/gsm8k-kvaware.jsonl" --limit 64 --concurrency 16 --max-tokens 8192 > "$R/gsm8k-kvaware.log" 2>&1
    say "rust kv-aware: $(tail -1 "$R/gsm8k-kvaware.log"); router warn/error lines: $(grep -aciE 'warn|error' "$R/router-kvaware.log")"
else
    say "rust kv-aware router did not come up: $(tail -3 "$R/router-kvaware.log" | tr '\n' ' ')"
fi
node_state after
say "end"
