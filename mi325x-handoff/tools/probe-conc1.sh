#!/usr/bin/env bash
# Single-user latency probe: fixed-length random prompts at concurrency 1.
#
# The tuning target is tok/s/user = 1000/TPOT. Decode cost depends on KV length,
# not on whether that KV was cached, so random prompts are a faithful and much
# cheaper stand-in for the agentic trace when only TPOT is being compared.
#
#   TAG=base ISL=68288 bash probe-conc1.sh <router host:port> [flush urls...]
set -euo pipefail

ROUTER="${1:?usage: probe-conc1.sh <host:port> [flush_url...]}"; shift
IMAGE="${IMAGE:-infera:sglang-gfx942-glm52-5c9cb3f}"
MODEL="${MODEL:?MODEL must point at the weights dir}"
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exec-logs/probe}"

ISL="${ISL:-68288}"          # agentic trace per-turn input p50
OSL="${OSL:-220}"            # agentic trace output_len
NREQ="${NREQ:-5}"
TAG="${TAG:-probe}"
NAME="${TAG}_isl${ISL}_c1"

mkdir -p "$OUT_DIR"

# Gotcha 3: flush_cache is a no-op while requests are in flight and still looks
# like it worked. Refuse to measure rather than silently measure a warm cache.
for u in "$@"; do
  code=$(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST "$u/flush_cache")
  [ "$code" = "200" ] || { echo "[probe] flush $u returned $code, fleet not idle" >&2; exit 1; }
  echo "[probe] flush $u -> ok"
done

echo "[probe] $NAME router=$ROUTER isl=$ISL osl=$OSL n=$NREQ"

docker run --rm --network host \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -v "$MODEL:$MODEL:ro" -v "$OUT_DIR:/out" \
  --entrypoint python3 "$IMAGE" -m sglang.benchmark.serving \
    --backend sglang-oai-chat \
    --host "${ROUTER%%:*}" --port "${ROUTER##*:}" \
    --model "$MODEL" --tokenizer "$MODEL" --served-model-name "$MODEL" \
    --dataset-name random \
    --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1.0 \
    --num-prompts "$NREQ" --max-concurrency 1 \
    --warmup-requests 0 --cache-report --output-details \
    --output-file "/out/${NAME}.jsonl" \
  2>&1 | tee "$OUT_DIR/${NAME}.log"

awk -v name="$NAME" '
  /^Mean TPOT/         {tpot=$NF}
  /^Median TPOT/       {mtpot=$NF}
  /^Mean TTFT/         {ttft=$NF}
  /^Mean ITL/          {itl=$NF}
  /^Output token throughput/ {thr=$NF}
  END {
    printf "\n=== %s ===\n", name
    printf "  mean TTFT        %10.1f ms\n", ttft
    printf "  mean TPOT        %10.2f ms  -> %.1f tok/s/user\n", tpot, 1000/tpot
    printf "  median TPOT      %10.2f ms  -> %.1f tok/s/user\n", mtpot, 1000/mtpot
    printf "  mean ITL         %10.2f ms\n", itl
    printf "  MTP accept len   %10.2f tokens/step\n", itl/tpot
    printf "  output tput      %10.2f tok/s\n", thr
  }' "$OUT_DIR/${NAME}.log" | tee "$OUT_DIR/${NAME}.summary.txt"
