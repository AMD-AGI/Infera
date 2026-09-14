#!/usr/bin/env bash
# Synthetic concurrency sweep or the InferenceX AgentX-MVP workload.
# Usage: bash bench.sh synthetic [CONC ...] | agentic CONC | agentic-sweep [CONC ...]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"
acquire_bench_lock

MODE="${1:-synthetic}"
shift || true
URL="http://$PREFILL_IP:$ROUTER_PORT"
OUT="$RUN_ROOT/$MODE"
mkdir -p "$OUT"

case "$MODE" in
synthetic)
    concurrencies=("${@:-8 16 32 64 128}")
    for concurrency in ${concurrencies[*]}; do
        requests=$((concurrency * 10))
        warmup="$concurrency"
        ((warmup > 8)) && warmup=8
        tag="isl8192_osl1024_c${concurrency}"
        echo "[bench] synthetic $tag requests=$requests"
        ssh $SSH_OPTS "$PREFILL_NODE" docker run --rm --network host \
            --device=/dev/kfd --device=/dev/dri \
            --group-add video --group-add render \
            -v "$MODEL:$MODEL:ro" -v "$OUT:$OUT" "$IMAGE" \
            python3 -m sglang.bench_serving \
                --backend sglang-oai-chat --base-url "$URL" \
                --model "$MODEL" --served-model-name "$SERVED_MODEL" \
                --tokenizer "$MODEL" \
                --dataset-name random \
                --random-input-len 8192 --random-output-len 1024 \
                --random-range-ratio 1.0 \
                --seed 42 \
                --max-concurrency "$concurrency" \
                --num-prompts "$requests" --request-rate inf \
                --warmup-requests "$warmup" \
                --temperature 0.0 --top-p 1.0 \
                --cache-report --output-details \
                --output-file "$OUT/$tag.jsonl" \
            2>&1 | tee "$OUT/$tag.log"

        workers="$(curl -fsS "$URL/v1/workers")"
        WORKERS="$workers" python3 - <<'PY'
import json
import os

workers = json.loads(os.environ["WORKERS"])["workers"]
assert len(workers) == 2, workers
assert {w["disagg_mode"] for w in workers} == {"prefill", "decode"}, workers
PY
    done
    ;;

agentic)
    concurrency="${1:-${AGENTIC_CONCURRENCY:-1}}"
    exec bash "$DIR/agentx_point.sh" "$concurrency" \
        "$RUN_ROOT/agentx/c$concurrency"
    ;;

agentic-sweep)
    exec bash "$DIR/run_agentx_sweep.sh" "$@"
    ;;

*)
    echo "usage: $0 synthetic [concurrency ...] | agentic CONC | agentic-sweep [CONC ...]" >&2
    exit 64
    ;;
esac

echo "[bench] done: $OUT"
