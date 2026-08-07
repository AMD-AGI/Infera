#!/bin/bash
# Task 1 — fixlen sweep, InferenceX-aligned, against the MIX router.
# ISL = percentile x 10%  (operator: "实际输入=分位×10%"): 7400 / 15500 / 23500,
# paired OSL 320 / 3300 / 17000. conc {1,8,16,24}. One frozen server.
#
# sglang.bench_serving flags carried from the reference bench.sh / fixlen_round.sh:
#   --random-range-ratio 1.0  pins every prompt to exactly ISL (a delta, not a distribution)
#   --temperature 1.0 --top-p 0.95  the checkpoint's own generation_config (NOT greedy)
#   --cache-report  needs --enable-cache-report on the worker; column is residue on
#                   dataset=random (no shared prefix by construction) — note it, don't trust it.
# Report output_throughput / total_token_throughput; divide by 8 for per-GPU.
set -u
export DOCKER_CONFIG="${DOCKER_CONFIG:-/var/tmp/dockercfg_yihou}"; mkdir -p "$DOCKER_CONFIG"
MY_IP="${MY_IP:?}"; CTR="${CTR:-glm52_mix}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
SERVED="${SERVED:-glm5.2-mxfp4}"; MODEL="${MODEL:?MODEL=weights dir for tokenizer}"
URL="http://$MY_IP:$ROUTER_PORT"
OUT="${OUT:-/tmp/mix_fixlen}"
SHAPES="${SHAPES:-p50 p90 p99}"
CONCS="${CONCS:-1 8 16 24}"
docker exec "$CTR" bash -c "mkdir -p $OUT"

for S in $SHAPES; do
  case "$S" in
    p50) ISL=7400;  OSL=320   ;;
    p90) ISL=15500; OSL=3300  ;;
    p99) ISL=23500; OSL=17000 ;;
    *) echo "unknown shape $S"; continue ;;
  esac
  for C in $CONCS; do
    NPROMPTS=$((C * 10)); WARM=$((C < 8 ? C : 8)); [ "$C" -ge 8 ] && WARM=$((C*2))
    # warmup = 2xC per InferenceX alignment; cap keeps conc=1 cheap
    TAG="mix_${S}_isl${ISL}_osl${OSL}_c${C}"
    echo "===== $TAG : ISL=$ISL OSL=$OSL conc=$C n=$NPROMPTS warm=$WARM ====="
    docker exec "$CTR" bash -c "python3 -m sglang.bench_serving \
      --backend sglang-oai-chat --base-url $URL \
      --model $SERVED --tokenizer $MODEL \
      --dataset-name random --random-input-len $ISL --random-output-len $OSL \
      --random-range-ratio 1.0 \
      --max-concurrency $C --num-prompts $NPROMPTS --request-rate inf \
      --warmup-requests $WARM \
      --cache-report --temperature 1.0 --top-p 0.95 \
      --output-file $OUT/$TAG.jsonl --output-details 2>&1 | tail -35"
    echo
  done
done
echo "fixlen done -> $CTR:$OUT/*.jsonl"
