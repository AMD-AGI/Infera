#!/usr/bin/env bash
# Fixed-length sweep, same shape as the GLM-5.2 MIX fixlen baseline in
# infera.glm52.mix.experiment/fixlen.glm52.mix.packup_20260806/.
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
set -u
CTR="${CTR:-glm53_mix}"
HOST="${HOST:?HOST=ip the worker binds to}"
PORT="${PORT:-30000}"
MODEL="${MODEL:-glm5.3-flash-mxfp4}"
ISL="${ISL:-7400}"; OSL="${OSL:-320}"; ARM="${ARM:-p50}"
CONCS="${CONCS:-1 8 16 24}"
OUT="${OUT:-/apps/yihou/glm53.series.workspace_20260901/results/fixlen_${ARM}.csv}"
mkdir -p "$(dirname "$OUT")"
echo "arm,isl,osl,conc,completed,req_s,out_tok_s,total_tok_s,ttft_p50_ms,ttft_p99_ms,tpot_mean_ms,e2e_p50_ms" > "$OUT"
for c in $CONCS; do
  echo "### $ARM isl=$ISL osl=$OSL conc=$c" >&2
  docker exec "$CTR" python3 -m sglang.bench_serving --backend sglang-oai-chat \
    --host "$HOST" --port "$PORT" --model "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
    --random-range-ratio 1.0 --max-concurrency "$c" --num-prompts $((c * 10)) \
    --temperature 1.0 --top-p 0.95 2>&1 | tee /tmp/bench_last.txt >&2
  python3 - "$OUT" "$ARM" "$ISL" "$OSL" "$c" <<'PY'
import re, sys
out, arm, isl, osl, conc = sys.argv[1:6]
t = open('/tmp/bench_last.txt', errors='replace').read()
def g(pat, default=''):
    m = re.search(pat + r'\s*:\s*([0-9.]+)', t)
    return m.group(1) if m else default
row = [arm, isl, osl, conc, g('Successful requests'), g(r'Request throughput \(req/s\)'),
       g(r'Output token throughput \(tok/s\)'), g(r'Total token throughput \(tok/s\)'),
       g('Median TTFT \\(ms\\)'), g('P99 TTFT \\(ms\\)'), g('Mean TPOT \\(ms\\)'),
       g('Median E2E Latency \\(ms\\)')]
open(out, 'a').write(','.join(row) + '\n')
print('  ->', ','.join(row))
PY
done
echo "done -> $OUT" >&2
