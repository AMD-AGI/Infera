#!/usr/bin/env bash
# what: the fixed-length sweep (mission task 1), InferenceX-aligned, against the router.
# why : sglang's own bench_serving ships inside the image, so nothing extra to install.
# how : run ON the node.  ARMS="p50 p90 p99" CONCS="1 8 16 24" bash mix_bench_fixlen.sh
#
# ISL is the 10% FRESH REMAINDER of Case A's inputs, not the full prompt: the agentic
# workload runs at an 89-90% prefix-cache hit rate, so only ~10% of each prompt is actually
# computed. --dataset-name random builds every prompt independently (no shared prefix by
# construction), so the sent length IS the computed length and the two line up.
#
#   arm   Case-A input   ISL sent   OSL
#   p50      74,000        7,400     320
#   p90     155,000       15,500   3,300
#   p99     235,000       23,500  17,000
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/mix_common.sh"

require_env MY_IP; require_env MODEL
URL="http://$MY_IP:$ROUTER_PORT"
OUT="${OUT:-/mnt/vast/c_huggingface/glm52_mix_20260806/results/fixlen}"
ARMS="${ARMS:-p50 p90 p99}"
CONCS="${CONCS:-1 8 16 24}"
TAG_PREFIX="${TAG_PREFIX:-base}"

isl_of(){ case "$1" in p50) echo 7400;; p90) echo 15500;; p99) echo 23500;; *) die "unknown arm $1";; esac; }
osl_of(){ case "$1" in p50) echo 320;;  p90) echo 3300;;  p99) echo 17000;; *) die "unknown arm $1";; esac; }

docker exec "$CTR" mkdir -p "$OUT"

for ARM in $ARMS; do
  ISL=$(isl_of "$ARM"); OSL=$(osl_of "$ARM")
  for C in $CONCS; do
    # num-prompts = conc x 10, the InferenceX convention (benchmark_lib.sh
    # run_benchmark_serving). A single fixed N would leave the high arms with too few
    # requests to reach steady state and their percentiles unusable.
    N=$((C * 10))
    WARM=$([ "$C" -lt 8 ] && echo "$C" || echo 8)
    TAG="${TAG_PREFIX}_${ARM}_isl${ISL}_osl${OSL}_c${C}"
    log "=== $TAG (n=$N warm=$WARM) ==="
    # Three flags are load-bearing in non-obvious ways:
    #   --random-range-ratio 1.0 pins every prompt to exactly ISL. The default draws
    #     uniformly and the percentiles then mix request sizes.
    #   --temperature 1.0 --top-p 0.95 are the checkpoint's own generation_config, and are
    #     deliberately NOT greedy: at temperature 0 this reasoning model repeats on a long
    #     prompt, MTP predicts the loop perfectly, acceptance pins at 4.00, and the run
    #     reads like KV corruption.
    #   --cache-report needs the server's --enable-cache-report. Its column is meaningless
    #     on this dataset (no shared prefix by construction) — it is here to confirm that,
    #     not to be read as a hit rate.
    docker exec "$CTR" bash -c "python3 -m sglang.bench_serving \
      --backend sglang-oai-chat --base-url $URL \
      --model $SERVED --tokenizer ${TOKENIZER:-$MODEL} \
      --dataset-name random --random-input-len $ISL --random-output-len $OSL \
      --random-range-ratio 1.0 \
      --max-concurrency $C --num-prompts $N --request-rate inf \
      --warmup-requests $WARM \
      --cache-report --temperature 1.0 --top-p 0.95 \
      --output-file $OUT/$TAG.jsonl --output-details 2>&1 | tail -32"
  done
done

log "fixlen sweep done -> $OUT/*.jsonl (per-request ttfts/itls are inside)"
