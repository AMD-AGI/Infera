#!/usr/bin/env bash
# One fixlen round of the sweep. Runs ON the prefill node, inside bench_run.
#
# Paired ISL/OSL percentiles, P99 dropped (operator instruction):
#     p50   ISL  74,000  OSL   320
#     p90   ISL 155,000  OSL 3,300
# x conc {1,32,64,128} = 8 rounds, all against ONE server frozen at the config
# sized for the p90 pair. No per-workload retuning.
#
# --random-range-ratio 1.0 pins every prompt to exactly ISL: a fixlen sweep wants
# a delta, not a distribution. (The default 0.0 would draw uniformly.)
#
# Sampling is the model's own generation_config (temp 1.0 / top_p 0.95). NOT
# temperature 0: greedy decoding sends this reasoning model into repetition on a
# long prompt, MTP predicts the loop perfectly, accept len pins at 4.00, and the
# run reads as KV corruption.
#
#   PAIR=p50|p90  C=<conc>  [N=<num-prompts>]
set -u
PAIR="${PAIR:?p50|p90}"
C="${C:?concurrency}"
CTR="${CTR:-bench_run}"
R="${R:-http://10.2.122.10:8100}"
W=/mnt/vast/c_huggingface/bench_20260801
OUT="$W/results"

case "$PAIR" in
  p50) ISL=74000;  OSL=320  ;;
  p90) ISL=155000; OSL=3300 ;;
  *) echo "bad PAIR=$PAIR" >&2; exit 1 ;;
esac

# Enough requests to be more than a warm-up at each concurrency, without paying
# for samples the percentiles cannot use. At conc=1 the wall clock is the
# constraint; at conc=128 the queue is.
if [ -z "${N:-}" ]; then
  case "$C" in
    1)   N=$([ "$PAIR" = p50 ] && echo 8   || echo 4)   ;;
    32)  N=$([ "$PAIR" = p50 ] && echo 64  || echo 32)  ;;
    64)  N=$([ "$PAIR" = p50 ] && echo 128 || echo 64)  ;;
    128) N=$([ "$PAIR" = p50 ] && echo 256 || echo 128) ;;
    *)   N=$((C * 2)) ;;
  esac
fi

TAG="fixlen_${PAIR}_c${C}"
mkdir -p "$OUT"

echo "===== $TAG : ISL=$ISL OSL=$OSL conc=$C n=$N ====="
echo "--- kvd before"
docker exec "$CTR" python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock \
  > "$OUT/${TAG}.kvd_before.json"
cat "$OUT/${TAG}.kvd_before.json"
docker exec "$CTR" bash -c 'wc -l < /tmp/router.log' > "$OUT/${TAG}.routerline"

docker exec "$CTR" python3 -m sglang.bench_serving \
  --backend sglang-oai-chat --base-url "$R" --model glm5.2-mxfp4 \
  --tokenizer /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 \
  --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
  --random-range-ratio 1.0 \
  --max-concurrency "$C" --num-prompts "$N" \
  --cache-report --temperature 1.0 --top-p 0.95 \
  --output-file "$OUT/${TAG}.jsonl" --output-details \
  --warmup-requests 0 \
  2>&1 | tee "$OUT/${TAG}.log" | tail -40

echo "--- kvd after"
docker exec "$CTR" python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock \
  > "$OUT/${TAG}.kvd_after.json"
cat "$OUT/${TAG}.kvd_after.json"
echo "===== $TAG done ====="
