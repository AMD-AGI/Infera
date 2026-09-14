#!/usr/bin/env bash
# Run the InferenceX GSM8K evaluation against an already-running 1P1D router.
# Usage: bash eval/gsm8k.sh  # requires a running stack
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DIR/config.sh"
acquire_bench_lock

OUT="${GSM8K_OUT_DIR:-$RUN_ROOT/correctness/gsm8k}"
REMOTE_RUN_ROOT="$REMOTE_WORK_ROOT/$RUN_ID"
REMOTE_OUT="$REMOTE_RUN_ROOT/correctness/gsm8k"
WORKSPACE_ROOT="$(cd "$DIR/../../.." && pwd)"

rm -rf "$OUT"
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1

flush_cache() {
    local role="$1" url="$2" attempt status
    local body="$OUT/flush_${role}.txt"
    for attempt in $(seq 1 30); do
        if ! status="$(curl -sS -m 20 -o "$body" -w '%{http_code}' \
            -X POST "$url/flush_cache")"; then
            echo "[gsm8k] $role flush request failed" >&2
            return 1
        fi
        [[ "$status" == "200" ]] && return 0
        if [[ "$status" == "400" && "$attempt" -lt 30 ]]; then
            sleep 2
            continue
        fi
        echo "[gsm8k] $role flush failed with HTTP $status: $(<"$body")" >&2
        return 1
    done
}

ssh $SSH_OPTS "$PREFILL_NODE" "test -r '$MODEL/tokenizer_config.json'" || {
    echo "[gsm8k] tokenizer not readable at $MODEL on $PREFILL_NODE" >&2
    exit 66
}

# Keep this module independent from long-context: start the score run on an
# idle, empty KV cache even when it is invoked directly.
flush_cache prefill "http://$PREFILL_IP:$PREFILL_PORT"
flush_cache decode "http://$DECODE_IP:$DECODE_PORT"

ssh $SSH_OPTS "$PREFILL_NODE" \
    "docker rm -f glm52-lm-eval-client >/dev/null 2>&1 || true
     rm -rf '$REMOTE_OUT' && mkdir -p '$REMOTE_OUT'"
cat >"$OUT/client.env" <<EOF
PORT=$ROUTER_PORT
MODEL=$MODEL
MODEL_NAME=$SERVED_MODEL
MODEL_PREFIX=$MODEL_PREFIX
EVAL_CONCURRENT_REQUESTS=$GSM8K_CONCURRENT_REQUESTS
EVAL_LIMIT=$CORRECTNESS_EVAL_LIMIT
HOST_UID=$(id -u)
HOST_GID=$(id -g)
EOF

set +e
ssh $SSH_OPTS "$PREFILL_NODE" docker run -i --rm \
    --name glm52-lm-eval-client \
    --network host --ipc host --shm-size 32g \
    -v "$WORKSPACE_ROOT:$WORKSPACE_ROOT" \
    -v "$REMOTE_RUN_ROOT:$REMOTE_RUN_ROOT" \
    --mount "type=bind,src=$MODEL,dst=$MODEL,readonly" \
    --env-file "$OUT/client.env" \
    "$IMAGE" bash -s -- "$REMOTE_OUT" "$INFERENCEX_DIR" <<'LM_EVAL_CLIENT' \
    2>&1 | tee "$OUT/runner.log"
set -e
out=$1
ix=$2
trap "chown -R $HOST_UID:$HOST_GID $out" EXIT
cd "$ix"
source benchmarks/benchmark_lib.sh
run_lm_eval \
    --port "$PORT" \
    --task utils/evals/gsm8k.yaml \
    --results-dir "$out"
result="$(find "$out" -type f -name "results*.json" -print -quit)"
test -n "$result"
python3 utils/evals/validate_scores.py \
    --model-prefix "$MODEL_PREFIX" \
    --results-glob "$result"
LM_EVAL_CLIENT
eval_rc=${PIPESTATUS[0]}
set -e

set +e
ssh $SSH_OPTS "$PREFILL_NODE" "tar -C '$REMOTE_OUT' -cf - ." \
    | tar -C "$OUT" -xf -
copy_rc=$?
set -e

(( eval_rc == 0 )) || {
    echo "[gsm8k] failed with rc=$eval_rc; copied available artifacts" >&2
    exit "$eval_rc"
}
(( copy_rc == 0 )) || {
    echo "[gsm8k] evaluation passed but artifact copy failed" >&2
    exit "$copy_rc"
}

result="$(find "$OUT" -type f -name "results*.json" -print -quit)"
test -n "$result"
echo "[gsm8k] PASS: $result"
