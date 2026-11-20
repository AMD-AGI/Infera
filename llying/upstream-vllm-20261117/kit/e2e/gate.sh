#!/usr/bin/env bash
# Correctness gate against one running server, the same checks as track A's gate.sh (its probes and data are used
# read-only from the track A work dir): smoke, GSM8K PROBE_N prompts, long context LONGCTX_N x CTX tokens (needle and
# GSM8K after filler), MTP acceptance from the server's counters.
#   gate.sh LABEL PORT OUTDIR
# Env: PROBE_N (200), LONGCTX_N (32), CTX (16000). Prints one summary line (also in OUTDIR/gate-LABEL/summary.txt).
set -uo pipefail
A=/home/liyingli/bench_agentx/vllm-trackA-20261118
label=$1 port=$2 out=$3
url=http://127.0.0.1:$port murl=http://127.0.0.1:$port/metrics
PROBE_N=${PROBE_N:-200} LONGCTX_N=${LONGCTX_N:-32} CTX=${CTX:-16000}
o=$out/gate-$label
P=$A/scripts/probes
mkdir -p "$o"
samples=$o/gsm8k.jsonl
zcat "$A/data/gsm8k_probe_prompts.jsonl.gz" > "$samples"
python3 "$A/scripts/metrics_snap.py" snap "$o/m0.json" "$murl"
t0=$(date +%s)
PREFILL_NUM_WORKERS=0 timeout 1800 python3 "$P/smoke.py" "$url" glm5.2-mxfp4 "$o/smoke" > "$o/smoke.out" 2>&1
smoke_pass=$(grep -c '"passed": true' "$o/smoke.out")
smoke_all=$(grep -c '"passed"' "$o/smoke.out")
timeout 3600 python3 "$P/gsm8k_probe.py" "$samples" "$url" glm5.2-mxfp4 "$o/probe.jsonl" --limit "$PROBE_N" \
    --concurrency 64 > "$o/probe.out" 2>&1
python3 "$P/probe_stats.py" "$o/probe.jsonl" --review "$o/probe-review.txt" > "$o/probe-stats.out" 2>&1
timeout 3600 python3 "$P/longctx_probe.py" "$samples" "$url" glm5.2-mxfp4 "$o/longctx.jsonl" --n "$LONGCTX_N" \
    --concurrency 32 --ctx-tokens "$CTX" > "$o/longctx.out" 2>&1
python3 "$A/scripts/metrics_snap.py" snap "$o/m1.json" "$murl"
acc=$(python3 "$A/scripts/metrics_snap.py" diff "$o/m0.json" "$o/m1.json")
line="$(date -u +%m%d-%T) gate $label: smoke $smoke_pass/$smoke_all; probe $(tail -1 "$o/probe-stats.out" | tr -s ' ');"
line+=" longctx $(grep -E '^(needle|gsm8k):' "$o/longctx.out" | tr '\n' ' ' | tr -s ' '); metrics $acc; $(( $(date +%s) - t0 )) s"
echo "$line" | tee "$o/summary.txt"
