#!/usr/bin/env bash
# Replay a Mooncake-format trace against the router with AIPerf.
# RUNS ON THE COMPUTE NODE. Adapted from
# examples/sglang_1p1d_glm5.2/engine/trace_replay.sh.
#
# AIPerf runs in its OWN container, never in the engine container: the engine
# image ships Python 3.10 and AIPerf needs >= 3.11 (this image is 3.13). Here
# that container is on the same host as the engine, which is what WORKERS bounds.
#
# The image's ENTRYPOINT is ["/bin/bash","-c"], so the whole aiperf invocation
# goes in as ONE string. Passing it as argv silently runs `aiperf` with the rest
# as positional parameters instead of flags.
#
# Four container settings are load-bearing and each fails quietly if dropped:
#   --user + HOME=/tmp                   the image runs as uid 1000 (nvs) but
#                                        writes into a host-owned directory
#   AIPERF_DATASET_MMAP_CACHE_DIR        HOME=/app inside the image, so without
#   AIPERF_DATASET_MMAP_BASE_PATH        this the dataset cache dies with the
#                                        --rm container (re-tokenizing the trace
#                                        every run) and the mmap data lands on
#                                        the node's root fs via the overlay
#   PYTHONPATH keeps the image's own     prepending ours must not drop
#     entries                            /usr/local/lib/python3.13/dist-packages:/app
#   HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE  AIPerf 0.12 sends existing local dirs
#                                        through snapshot_download; the
#                                        sitecustomize patch restores direct
#                                        local-directory loading
set -u
MY_IP="${NODE_IP:?}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
SERVED="${SERVED:-glm5.3-flash}"
MODEL="${MODEL:?}"
MODEL_MOUNT="${MODEL_MOUNT:?}"
AIPERF_IMAGE="${AIPERF_IMAGE:?}"
AIPERF_OUT="${AIPERF_OUT:?}"
AIPERF_TRACE="${AIPERF_TRACE:?}"
SCRIPTS="${SCRIPTS:?}"
TRACE_END_MS="${TRACE_END_MS:-120000}"
MAX_CONC="${MAX_CONC:-256}"
WORKERS="${WORKERS:-16}"
BLOCK_SIZE="${BLOCK_SIZE:-512}"
REQ_TIMEOUT="${REQ_TIMEOUT:-900}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"

COMPAT="$SCRIPTS/pythonpath"
RUN_DIR="$AIPERF_OUT/$TAG"
URL="http://$MY_IP:$ROUTER_PORT"

echo "===== preflight ====="
[ -r "$AIPERF_TRACE" ] || { echo "  ABORT: trace not readable: $AIPERF_TRACE"; exit 1; }
# AIPerf treats --tokenizer as a local path only when it exists, and as a HF repo
# id otherwise -- which HF_HUB_OFFLINE then turns into a hard failure at startup.
{ [ -r "$MODEL/tokenizer_config.json" ] || [ -r "$MODEL/tokenizer.json" ]; } \
  || { echo "  ABORT: no tokenizer files under $MODEL"; exit 1; }
curl -sf -m10 "$URL/health" >/dev/null || { echo "  ABORT: no answer from router at $URL"; exit 1; }
n=$(curl -s -m10 "$URL/v1/workers" | tr -d ' \r' | grep -c '"disagg_mode":"mixed"')
[ "${n:-0}" -ge 1 ] || { echo "  ABORT: no mixed worker registered"; exit 1; }
# Auto-promotion to fixed-schedule hinges on the first record carrying a
# timestamp. Without one the replay runs but ignores the trace's pacing, which is
# the entire point of a replay.
head -1 "$AIPERF_TRACE" | grep -q '"timestamp"' \
  || echo "  WARN: first record has no \"timestamp\" -- pacing will be ignored"
echo "  trace, tokenizer, router and worker all OK"

mkdir -p "$RUN_DIR/logs" "$AIPERF_OUT/.mmap_cache" "$AIPERF_OUT/.mmap"

# Source and destination are the same path everywhere, so a path means the same
# thing inside and outside. Deduped because docker rejects two mounts on one
# destination, and the trace living under AIPERF_OUT is a natural layout.
MOUNTS=(); SEEN="|"
add_mount(){
  local src="$1" ro="${2:-}"
  case "$SEEN" in *"|$src|"*) return 0 ;; esac
  SEEN="$SEEN$src|"
  MOUNTS+=(-v "$src:$src${ro:+:ro}")
}
add_mount "$AIPERF_OUT"
add_mount "$MODEL_MOUNT" ro
add_mount "$(dirname "$AIPERF_TRACE")" ro
add_mount "$COMPAT" ro

# --fixed-schedule-auto-offset shifts the first timestamp to 0 so the replay
# starts immediately; the end offset is then measured from that shifted origin.
# --concurrency is a CEILING, not a target: fixed-schedule sends at the recorded
# timestamps regardless. ignore_eos is what makes output length equal the trace's
# output_length -- the mooncake_trace loader sets max_tokens but injects no
# min_tokens, so a thinking model would otherwise replay short.
CMD="aiperf profile \
--model $SERVED --tokenizer $MODEL \
--endpoint-type chat --streaming \
--url $URL \
--input-file $AIPERF_TRACE --custom-dataset-type mooncake_trace \
--isl-block-size $BLOCK_SIZE \
--workers-max $WORKERS \
--request-timeout-seconds $REQ_TIMEOUT \
--no-gpu-telemetry --ui none \
--artifact-dir $RUN_DIR \
--fixed-schedule --fixed-schedule-auto-offset --fixed-schedule-end-offset $TRACE_END_MS \
--concurrency $MAX_CONC \
--extra-inputs temperature:1.0 top_p:0.95 ignore_eos:true"

echo "===== replay ====="
echo "  trace    : $AIPERF_TRACE"
echo "  window   : 0..${TRACE_END_MS} ms (auto-offset)"
echo "  endpoint : $URL  (model $SERVED)"
echo "  conc<=   : $MAX_CONC   workers: $WORKERS"
echo "  out      : $RUN_DIR"
echo "  started  : $(date -Is)"

docker run --rm --name "aiperf_$TAG" --network=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH="$COMPAT:/usr/local/lib/python3.13/dist-packages:/app" \
  -e AIPERF_DATASET_MMAP_CACHE_DIR="$AIPERF_OUT/.mmap_cache" \
  -e AIPERF_DATASET_MMAP_BASE_PATH="$AIPERF_OUT/.mmap" \
  "${MOUNTS[@]}" \
  "$AIPERF_IMAGE" "$CMD" 2>&1 | tee "$RUN_DIR/logs/replay_console.log"
rc="${PIPESTATUS[0]}"

echo "  finished : $(date -Is) rc=$rc"
echo "--- artifacts ---"
find "$RUN_DIR" -maxdepth 2 -type f -printf '  %10s  %p\n' 2>/dev/null | sort -k2
if [ "$rc" = "0" ]; then echo "AIPERF_OK $RUN_DIR"; else echo "AIPERF_FAIL rc=$rc"; fi
exit "$rc"
