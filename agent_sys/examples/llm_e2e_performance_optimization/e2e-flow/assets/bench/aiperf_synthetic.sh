#!/usr/bin/env bash
# A fixed-shape synthetic load against the router with AIPerf.
# RUNS ON THE COMPUTE NODE.
#
# **This is `../../../integration-demo/assets/bench/aiperf_replay.sh` with the
# trace swapped for AIPerf's synthetic generator, and nothing else changed.**
# Mission M1.2.3.4 asks for a 1k-in / 1k-out, concurrency-16, three-minute load;
# that package's script asks for a Mooncake trace replayed at its recorded
# pacing. The two differ in exactly one block — `--input-file … --fixed-schedule
# …` becomes `--synthetic-input-tokens-* --output-tokens-* --concurrency
# --benchmark-duration` — and the reason this is a sibling rather than a flag on
# that one is that the flags are mutually exclusive: `--fixed-schedule` ignores
# `--concurrency`, which it treats as a ceiling, so a "duration and concurrency"
# run cannot be expressed as an argument to a replay.
#
# Everything else is carried across **verbatim and for the reasons recorded
# there**, each of which fails quietly if dropped:
#
#   AIPerf runs in its OWN container, never the engine's: the engine image ships
#   Python 3.10 and AIPerf needs >= 3.11.
#
#   The image's ENTRYPOINT is ["/bin/bash","-c"], so the whole invocation goes in
#   as ONE string. Passing it as argv silently runs `aiperf` with the rest as
#   positional parameters instead of flags.
#
#   --user + HOME=/tmp                 the image runs as uid 1000 but writes into
#                                      a host-owned directory
#   AIPERF_DATASET_MMAP_CACHE_DIR      HOME=/app inside the image, so without
#   AIPERF_DATASET_MMAP_BASE_PATH      these the cache dies with the --rm
#                                      container and the mmap data lands on the
#                                      node's root fs via the overlay
#   PYTHONPATH keeps the image's own   prepending ours must not drop
#     entries                          /usr/local/lib/python3.13/dist-packages:/app
#   HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE  AIPerf 0.12 sends existing local dirs
#                                      through snapshot_download; the
#                                      sitecustomize patch restores direct
#                                      local-directory loading
set -u
MY_IP="${NODE_IP:?}"
ROUTER_PORT="${ROUTER_PORT:?}"
SERVED="${SERVED:?}"
MODEL="${MODEL:?}"
MODEL_MOUNT="${MODEL_MOUNT:?}"
AIPERF_IMAGE="${AIPERF_IMAGE:?}"
AIPERF_OUT="${AIPERF_OUT:?}"
SCRIPTS="${SCRIPTS:?}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONCURRENCY="${CONCURRENCY:-16}"
DURATION_S="${DURATION_S:-180}"
WORKERS="${WORKERS:-16}"
REQ_TIMEOUT="${REQ_TIMEOUT:-900}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"

COMPAT="$SCRIPTS/pythonpath"
RUN_DIR="$AIPERF_OUT/$TAG"
URL="http://$MY_IP:$ROUTER_PORT"

echo "===== preflight ====="
{ [ -r "$MODEL/tokenizer_config.json" ] || [ -r "$MODEL/tokenizer.json" ]; } \
  || { echo "  ABORT: no tokenizer files under $MODEL"; exit 1; }
curl -sf -m10 "$URL/health" >/dev/null || { echo "  ABORT: no answer from router at $URL"; exit 1; }
echo "  tokenizer and router OK"

mkdir -p "$RUN_DIR/logs" "$AIPERF_OUT/.mmap_cache" "$AIPERF_OUT/.mmap"

MOUNTS=(); SEEN="|"
add_mount(){
  local src="$1" ro="${2:-}"
  case "$SEEN" in *"|$src|"*) return 0 ;; esac
  SEEN="$SEEN$src|"
  MOUNTS+=(-v "$src:$src${ro:+:ro}")
}
add_mount "$AIPERF_OUT"
add_mount "$MODEL_MOUNT" ro
add_mount "$COMPAT" ro

# `--synthetic-input-tokens-stddev 0` and `--output-tokens-stddev 0`: the shape
# mission M1.2.3.4 names is a *fixed* 1k/1k, and AIPerf's default is a
# distribution around the mean. A spread here would make two runs of the same
# deployment differ by more than the deployment does.
#
# `ignore_eos:true` **with** `min_tokens`, and the pairing is specific to this
# script rather than inherited from the replay one.
#
# **Corrected by m2, who checked the sibling rather than taking my word.** I had
# written "the same reason the replay script sets it" — `aiperf_replay.sh` does
# **not** set `min_tokens`, and its own comment says why: the `mooncake_trace`
# loader gives every request a `max_tokens` from the trace, so `ignore_eos` alone
# pins the length to what the trace asked for. Their sealed evidence agrees —
# `output_sequence_length` avg 111.9, min 64, max 160, std 28.3, which is
# trace-shaped rather than model-shaped.
#
# A **synthetic fixed-shape** load has no per-request maximum to be pinned to, so
# the floor has to be supplied here or a model that stops early replays short and
# every per-token number then describes a shorter workload than the one asked
# for. Measured with both set: `output_sequence_length` 1024.06, min 1024, max
# 1025, std 0.24. Different loader, different requirement, both correct.
CMD="aiperf profile \
--model $SERVED --tokenizer $MODEL \
--endpoint-type chat --streaming \
--url $URL \
--synthetic-input-tokens-mean $ISL --synthetic-input-tokens-stddev 0 \
--output-tokens-mean $OSL --output-tokens-stddev 0 \
--concurrency $CONCURRENCY \
--benchmark-duration $DURATION_S \
--workers-max $WORKERS \
--request-timeout-seconds $REQ_TIMEOUT \
--no-gpu-telemetry --ui none \
--artifact-dir $RUN_DIR \
--extra-inputs temperature:1.0 top_p:0.95 ignore_eos:true min_tokens:$OSL"

echo "===== load ====="
echo "  shape    : ${ISL} in / ${OSL} out, concurrency ${CONCURRENCY}, ${DURATION_S}s"
echo "  endpoint : $URL  (model $SERVED)"
echo "  out      : $RUN_DIR"
echo "  started  : $(date -Is)"

docker run --rm --name "aiperf_$TAG" --network=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e PYTHONPATH="$COMPAT:/usr/local/lib/python3.13/dist-packages:/app" \
  -e AIPERF_DATASET_MMAP_CACHE_DIR="$AIPERF_OUT/.mmap_cache" \
  -e AIPERF_DATASET_MMAP_BASE_PATH="$AIPERF_OUT/.mmap" \
  "${MOUNTS[@]}" \
  "$AIPERF_IMAGE" "$CMD" 2>&1 | tee "$RUN_DIR/logs/load_console.log"
rc="${PIPESTATUS[0]}"

echo "  finished : $(date -Is) rc=$rc"
echo "--- artifacts ---"
find "$RUN_DIR" -maxdepth 3 -type f -printf '  %10s  %p\n' 2>/dev/null | sort -k2
if [ "$rc" = "0" ]; then echo "AIPERF_OK $RUN_DIR"; else echo "AIPERF_FAIL rc=$rc"; fi
exit "$rc"
