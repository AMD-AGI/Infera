#!/usr/bin/env bash
# Replay the agentic trace against any router, from one fixed client.
#
#   bash temp/run-agentic.sh docker-rust 10.21.9.44:8000 \
#        http://10.21.9.44:30001 http://10.21.9.41:31501
#   ARM=k8s-rust ... bash temp/run-agentic.sh k8s-rust 10.43.x.y:8000 <flush urls...>
#
# Args: <tag> <router host:port> [flush url ...]
#
# This is the bench half of examples/glm5.2_gfx942/run_agentic_trace.sh, with the
# deployment half factored out so the same client can point at either arm. That
# matters more than it sounds: the client's tokenizer, dataset, concurrency limiter
# and scoring all sit on this side of the wire, so running two different clients
# would put those in the comparison too.
#
# Kept because the tuning loop (tune-cycle.sh) drives it and the results in
# results/agentic/ came out of it. For a fresh deployment use the example's
# bench_client.sh instead -- same client, configured from env.sh.
#
# The client always runs here on tw041 in a container off the same image, whatever
# it is measuring, and reaches the router over the network in both arms.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TAG="${1:?usage: run-agentic.sh <tag> <router host:port> [flush url ...]}"
ROUTER="${2:?router host:port}"
shift 2
FLUSH_URLS=("$@")

REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
# The scorer and the trace converter live with the example, not here -- this
# script only supplies the client container around them.
WORK="${WORK:-$REPO/examples/glm5.2_gfx942}"
IMAGE="${IMAGE:-infera:sglang-gfx942-glm52}"
DATA_DIR="${DATA_DIR:-/tmp/infera-agentic-data}"
MODEL="${MODEL:-/tmp/infera-models/GLM-5.2-FP8}"
TRACE="${TRACE:-/data/cc_traces_100k.json}"
OUT_DIR="${OUT_DIR:-$HERE/exec-logs/agentic}"

# OUTPUT_LEN must be the value the dataset was built with, or the replayed turn
# lengths drift from the recorded ones and the scorer's ideal stops applying.
OUTPUT_LEN="${OUTPUT_LEN:-220}"
NUM_PROMPTS="${NUM_PROMPTS:-60}"   # conversations, not requests
CONC="${CONC:-16}"

# The scorer's ideal is page-aligned, so a wrong page size silently moves the
# efficiency number. Its own default is 64, and the ported README says SGLang
# forces 64 for GLM-5.2's DSA attention -- but both legs of this deployment report
# `page_size = 1` from /get_server_info, and the router reports kv_block_size 1 to
# match. Read it from the server rather than trusting either default.
PAGE_SIZE="${PAGE_SIZE:-}"

mkdir -p "$OUT_DIR"
NAME="${TAG}_c${CONC}_n${NUM_PROMPTS}"
DETAILS="/out/${NAME}.jsonl"

curl -sf -m 10 "http://$ROUTER/health" >/dev/null \
  || { echo "[agentic] router not answering at $ROUTER" >&2; exit 1; }

# The name the router answers to is the engine's --model-path, which is a mount
# path and therefore differs between the arms: /models/GLM-5.2-FP8 under k8s,
# /tmp/infera-models/GLM-5.2-FP8 under docker. The tokenizer, meanwhile, has to be
# a path that exists in this client container. So the two cannot be the same string
# and the request's model field has to come from the server.
# Sending the wrong one is not a clean 404 either -- the router answers
# `no active mixed worker for model="..."` and every request fails in milliseconds,
# which reads like a dead fleet rather than a name mismatch.
SERVED_MODEL="${SERVED_MODEL:-}"
if [[ -z "$SERVED_MODEL" ]]; then
  SERVED_MODEL=$(curl -sf -m 10 "http://$ROUTER/v1/models" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
  [[ -n "$SERVED_MODEL" ]] || { echo "[agentic] could not read served model name" >&2; exit 1; }
  echo "[agentic] served model name from the router: $SERVED_MODEL"
fi

if [[ -z "$PAGE_SIZE" ]]; then
  PAGE_SIZE=$(curl -sf -m 10 "${FLUSH_URLS[0]:-http://$ROUTER}/get_server_info" 2>/dev/null \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("page_size",""))' 2>/dev/null)
  [[ -n "$PAGE_SIZE" ]] || { echo "[agentic] could not read page_size; pass PAGE_SIZE=" >&2; exit 1; }
  echo "[agentic] page_size read from the server: $PAGE_SIZE"
fi

# Blocks left by an earlier run inflate the hit rate, and flush_cache is a no-op
# while requests are in flight, so this has to happen with the fleet idle.
for u in "${FLUSH_URLS[@]}"; do
  printf "[agentic] flush %s -> " "$u"
  curl -sf -m 20 -X POST "$u/flush_cache" >/dev/null 2>&1 && echo ok || echo "FAILED"
done
sleep 5

# --output-file appends, so a stale file would break the single-JSON scorer input.
rm -f "$OUT_DIR/${NAME}.jsonl"

echo "[agentic] $TAG router=$ROUTER conc=$CONC convs=$NUM_PROMPTS -> $OUT_DIR/${NAME}.*"

# --warmup-requests 0: in multi-turn mode a warmup request replays a whole
# conversation, which pre-warms the very cache the run is measuring.
docker run --rm --network host \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -v "$DATA_DIR:/data:ro" -v "$MODEL:$MODEL:ro" \
  -v "$WORK:/work:ro" -v "$OUT_DIR:/out" \
  --entrypoint python3 "$IMAGE" \
  -m sglang.benchmark.serving \
    --backend sglang-oai-chat \
    --host "${ROUTER%:*}" --port "${ROUTER#*:}" \
    --model "$MODEL" --tokenizer "$MODEL" \
    --served-model-name "$SERVED_MODEL" \
    --dataset-name agentic-trace --dataset-path "$TRACE" \
    --sharegpt-output-len "$OUTPUT_LEN" \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$CONC" \
    --warmup-requests 0 \
    --cache-report --output-details --output-file "$DETAILS" \
  2>&1 | tee "$OUT_DIR/${NAME}.log"

# The tool's own input/cache summary is wrong in multi-turn mode (it keeps the
# conversation-level prompt_len for every turn); this is the number to read.
echo
docker run --rm \
  -v "$DATA_DIR:/data:ro" -v "$WORK:/work:ro" -v "$OUT_DIR:/out" \
  --entrypoint python3 "$IMAGE" \
  /work/score_agentic_trace.py "$TRACE" "$DETAILS" "$NUM_PROMPTS" \
    --page-size "$PAGE_SIZE" \
  2>&1 | tee "$OUT_DIR/${NAME}.score.txt"
