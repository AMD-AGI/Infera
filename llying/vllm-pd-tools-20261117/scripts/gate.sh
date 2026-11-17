#!/usr/bin/env bash
# Correctness gate against a running deployment, the same checks as pd-repro-crusoe-20261116/scripts/chain_v2.sh:
# smoke (completion, chat, stream, ~66k needle twice), GSM8K PROBE_N prompts (garbled / bad start / runaway),
# long context LONGCTX_N x CTX tokens (needle + GSM8K after filler), real MTP acceptance from the engines' counters.
#   gate.sh LABEL URL "METRICS_URL [METRICS_URL...]"
# Env: PROBE_N (200), LONGCTX_N (64), CTX (16000), PREFILL_NUM_WORKERS (0 = no router registration check), SERVED.
# One summary line goes to $R/logs/summary.out; everything else to $R/results/gate-LABEL/.
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
label=$1 url=$2 murls=$3
SERVED=${SERVED:-glm5.2-mxfp4}
PROBE_N=${PROBE_N:-200} LONGCTX_N=${LONGCTX_N:-64} CTX=${CTX:-16000}
o=$R/results/gate-$label
mkdir -p "$o"
P=$R/scripts/probes
samples=$o/gsm8k.jsonl
zcat "$R/data/gsm8k_probe_prompts.jsonl.gz" > "$samples"
# shellcheck disable=SC2086
python3 "$R/scripts/metrics_snap.py" snap "$o/m0.json" $murls
t0=$(date +%s)
PREFILL_NUM_WORKERS=${PREFILL_NUM_WORKERS:-0} timeout 1800 python3 "$P/smoke.py" "$url" "$SERVED" "$o/smoke" > "$o/smoke.out" 2>&1
smoke_pass=$(grep -c '"passed": true' "$o/smoke.out")
smoke_all=$(grep -c '"passed"' "$o/smoke.out")
timeout 3600 python3 "$P/gsm8k_probe.py" "$samples" "$url" "$SERVED" "$o/probe.jsonl" --limit "$PROBE_N" --concurrency 64 > "$o/probe.out" 2>&1
python3 "$P/probe_stats.py" "$o/probe.jsonl" --review "$o/probe-review.txt" > "$o/probe-stats.out" 2>&1
timeout 3600 python3 "$P/longctx_probe.py" "$samples" "$url" "$SERVED" "$o/longctx.jsonl" --n "$LONGCTX_N" --concurrency 32 \
    --ctx-tokens "$CTX" > "$o/longctx.out" 2>&1
# shellcheck disable=SC2086
python3 "$R/scripts/metrics_snap.py" snap "$o/m1.json" $murls
acc=$(python3 "$R/scripts/metrics_snap.py" diff "$o/m0.json" "$o/m1.json")
echo "$acc" > "$o/metrics-delta.json"
line="$(date -u +%m%d-%T) gate $label: smoke $smoke_pass/$smoke_all; probe $(tail -1 "$o/probe-stats.out" | tr -s ' ');"
line+=" longctx $(grep -E '^(needle|gsm8k):' "$o/longctx.out" | tr '\n' ' ' | tr -s ' '); metrics $acc; $(( $(date +%s) - t0 )) s"
echo "$line" | tee -a "$R/logs/summary.out"
