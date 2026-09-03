#!/usr/bin/env bash
# Fixed-length sweep for the GLM-5.3 "big" MIX arm, shaped to be readable
# against the GLM-5.2 MIX fixlen baseline in
#   infera.glm52.mix.experiment/fixlen.glm52.mix.packup_20260806/
#
# Derived from ../scripts/fixlen_sweep.sh, with the baseline harness's own
# invocation (scripts/mix_bench_fixlen.sh) folded back in so the two are
# comparable request-for-request:
#   * bench against the ROUTER url, not the engine port -- the baseline did,
#     and the router is part of what is being measured.
#   * --request-rate inf and --warmup-requests min(conc,8), as the baseline.
#   * --cache-report. This comment used to say the column is "MEANINGLESS on
#     --dataset-name random (no shared prefix by construction)". RETRACTED
#     2026-09-02: that is false and the column is load-bearing. sglang seeds
#     `random` from --seed (default 42) before shuffling ShareGPT, so with
#     --num-prompts = 10 x conc each arm's prompt list is a strict PREFIX of
#     the next arm's and every arm re-sends the previous arm's prompts.
#     Measured hit rates 12.38 / 49.51 / 66.05 % at conc 8/16/24 against a
#     source-derived prediction of 12.5 / 50.0 / 66.7 %. READ THIS COLUMN:
#     without it you cannot tell how much prefill work an arm actually did.
#     See RECIPE.md, section "RETRACTION".
#   * --output-file ... --output-details, so per-request ttfts[]/itls[] survive.
#
# Three flags are load-bearing and are NOT defaults:
#   --random-range-ratio 1.0  pins every prompt to exactly ISL. The default
#                             draws uniformly and the percentiles then mix
#                             request sizes -- a fixed-length sweep wants a
#                             delta, not a distribution.
#   --temperature 1.0 --top-p 0.95  the checkpoint's own generation_config
#                             defaults, deliberately NOT greedy: at temperature
#                             0 this reasoning model falls into repetition on a
#                             long prompt.
#   --num-prompts 10 x conc   the InferenceX convention, so each arm reaches
#                             steady state.
#
# NOT the same operating point as the baseline. The baseline is GLM-5.2-MXFP4
# at TP8 with DP-attention + MTP + kvd all ON, on a whole node. This is TP4 on
# half a node with none of the three. Do not quote the ratio as a speedup.
set -uo pipefail
CTR="${CTR:-glm53_big_mix}"
HOST="${HOST:?HOST=ip the router binds to}"
PORT="${PORT:-8110}"
SERVED="${SERVED:-glm-5.3-mxfp4}"
MODEL="${MODEL:-/perf_apps/data/models/GLM-5.3-MXFP4}"
ARMS="${ARMS:-p50}"
CONCS="${CONCS:-1 8 16 24}"
OUTDIR="${OUTDIR:-/apps/yihou/glm53.series.workspace_20260901/bigmodel/results}"
CSV="${CSV:-$OUTDIR/fixlen_big.csv}"
URL="http://$HOST:$PORT"

isl_of(){ case "$1" in p50) echo 7400;; p90) echo 15500;; p99) echo 23500;; *) echo BAD;; esac; }
osl_of(){ case "$1" in p50) echo 320;;  p90) echo 3300;;  p99) echo 17000;; *) echo BAD;; esac; }

mkdir -p "$OUTDIR"
docker exec "$CTR" mkdir -p /tmp/fixlen
[ -f "$CSV" ] || echo "arm,isl,osl,conc,completed,req_s,out_tok_s,total_tok_s,ttft_p50_ms,ttft_p99_ms,tpot_mean_ms,e2e_p50_ms" > "$CSV"

for ARM in $ARMS; do
  ISL=$(isl_of "$ARM"); OSL=$(osl_of "$ARM")
  [ "$ISL" = BAD ] && { echo "unknown arm $ARM" >&2; exit 2; }
  for C in $CONCS; do
    N=$((C * 10))
    WARM=$([ "$C" -lt 8 ] && echo "$C" || echo 8)
    TAG="big_${ARM}_isl${ISL}_osl${OSL}_c${C}"
    echo "### $TAG (n=$N warm=$WARM) $(date -u +%H:%M:%S)" >&2
    docker exec "$CTR" bash -c "python3 -m sglang.bench_serving \
      --backend sglang-oai-chat --base-url $URL \
      --model $SERVED --tokenizer $MODEL \
      --dataset-name random --random-input-len $ISL --random-output-len $OSL \
      --random-range-ratio 1.0 \
      --max-concurrency $C --num-prompts $N --request-rate inf \
      --warmup-requests $WARM \
      --cache-report --temperature 1.0 --top-p 0.95 \
      --output-file /tmp/fixlen/$TAG.jsonl --output-details" \
      > "$OUTDIR/$TAG.txt" 2>&1
    python3 - "$CSV" "$ARM" "$ISL" "$OSL" "$C" "$OUTDIR/$TAG.txt" <<'PY'
import re, sys
csv, arm, isl, osl, conc, src = sys.argv[1:7]
t = open(src, errors='replace').read()
def g(pat, default=''):
    m = re.search(pat + r'\s*:\s*([0-9.]+)', t)
    return m.group(1) if m else default
row = [arm, isl, osl, conc, g('Successful requests'), g(r'Request throughput \(req/s\)'),
       g(r'Output token throughput \(tok/s\)'), g(r'Total token throughput \(tok/s\)'),
       g(r'Median TTFT \(ms\)'), g(r'P99 TTFT \(ms\)'), g(r'Mean TPOT \(ms\)'),
       g(r'Median E2E Latency \(ms\)')]
open(csv, 'a').write(','.join(row) + '\n')
print('  ->', ','.join(row))
PY
  done
done
echo "done -> $CSV" >&2
