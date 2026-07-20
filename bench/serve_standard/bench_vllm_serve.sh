#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# =============================================================================
# Canonical cross-team serving benchmark  (vllm bench serve)
# -----------------------------------------------------------------------------
# WHY THIS SCRIPT EXISTS
#   Other teams quote `vllm bench serve` numbers. Our in-house run_cliff reports
#   a different (systematically ~25% higher) throughput because it measures a
#   tighter wall (pure asyncio.gather window, prompts pre-built, median of warm
#   iters) and ONLY ever measures the warm/cache-hit regime. That makes our
#   numbers impossible to line up with anyone else's. This script standardises
#   on `vllm bench serve` so results ARE comparable, and records enough metadata
#   that a run is fully reproducible.
#
# PROTOCOL
#   * dataset = random, fixed --random-input-len / --random-output-len
#   * fixed --seed, each concurrency run TWICE with the same seed:
#       run1 = COLD  (no cache; real prefill compute throughput)
#       run2 = WARM  (identical prompts -> prefix/L3 cache hit; the KV-offload win)
#   * concurrency == num-prompts (one wave), swept over a fixed set
#   * everything (model / GPU / image / versions / server config) captured to a
#     manifest so the result dir is self-describing.
#
# USAGE
#   BASE_URL=http://localhost:8833 IMAGE=infera/engine-vllm:dev \
#     bash bench/serve_standard/bench_vllm_serve.sh
#
# Common overrides (env vars): BASE_URL MODEL TOKENIZER ISL OSL SEED
#   CONCURRENCIES IMAGE GPU OUTDIR
# =============================================================================
set -uo pipefail

# ---- config (all env-overridable) -------------------------------------------
BASE_URL="${BASE_URL:-http://localhost:8833}"
MODEL="${MODEL:-gpt-oss-120b}"
TOKENIZER="${TOKENIZER:-}"                 # empty -> auto from server /v1/models root
ISL="${ISL:-60000}"
OSL="${OSL:-1}"
# SEED empty => PER-ROUND seed = the concurrency value N. This keeps each
# concurrency's prompt set DISJOINT from every other N's, so run1 is a genuine
# COLD run (its prompts were never generated — hence never cached — by a
# larger N). run1 and run2 of the SAME N share the seed so run2 is warm.
# Set SEED=<int> to force one fixed seed across all rounds (legacy behaviour;
# then a larger N pre-warms a smaller N and the smaller N's cold is invalid).
SEED="${SEED:-}"
CONCURRENCIES="${CONCURRENCIES:-16 32 64 128 196 256}"
IMAGE="${IMAGE:-unknown}"                  # pass the docker image tag explicitly
GPU="${GPU:-${ROCR_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-unknown}}}"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-./bench_results/vllm_serve_${STAMP}}"

mkdir -p "$OUTDIR"

# ---- auto-detect tokenizer (local model path) from the server ---------------
models_json="$(curl -fsS --max-time 5 "${BASE_URL%/}/v1/models" 2>/dev/null || true)"
srv_root="$(printf '%s' "$models_json" | python3 -c \
  'import sys,json;d=json.load(sys.stdin);print(d["data"][0].get("root",""))' 2>/dev/null || true)"
srv_maxlen="$(printf '%s' "$models_json" | python3 -c \
  'import sys,json;d=json.load(sys.stdin);print(d["data"][0].get("max_model_len",""))' 2>/dev/null || true)"
[ -z "$TOKENIZER" ] && TOKENIZER="${srv_root:-$MODEL}"

# ---- capture metadata -------------------------------------------------------
vllm_ver="$(python3 -c 'import vllm;print(vllm.__version__)' 2>/dev/null || echo n/a)"
hipfile_ver="$(python3 -c 'import hipfile;print(getattr(hipfile,"__version__","?"))' 2>/dev/null || echo n/a)"
gpu_name="$(rocm-smi --showproductname 2>/dev/null | grep -m1 -i 'card series\|product name' \
  | sed 's/.*: //' || true)"
[ -z "$gpu_name" ] && gpu_name="$(rocminfo 2>/dev/null | grep -m1 -i 'Marketing Name' | sed 's/.*: *//' || echo n/a)"

MANIFEST="$OUTDIR/manifest.txt"
{
  echo "# vllm serve standard benchmark"
  echo "date_utc          : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname          : $(hostname)"
  echo "base_url          : $BASE_URL"
  echo "image_tag         : $IMAGE"
  echo "gpu_visible       : $GPU"
  echo "gpu_name          : $gpu_name"
  echo "vllm_version      : $vllm_ver"
  echo "hipfile_version   : $hipfile_ver"
  echo "model             : $MODEL"
  echo "tokenizer         : $TOKENIZER"
  echo "server_model_root : ${srv_root:-n/a}"
  echo "server_max_model_len: ${srv_maxlen:-n/a}"
  echo "isl               : $ISL"
  echo "osl               : $OSL"
  echo "seed              : ${SEED:-per-round (=concurrency N)}"
  echo "concurrencies     : $CONCURRENCIES"
  echo "runs_per_conc     : 2 (run1=cold, run2=warm)"
} | tee "$MANIFEST"
echo

# ---- sweep ------------------------------------------------------------------
SUMMARY="$OUTDIR/summary.csv"
echo "concurrency,run,phase,successful,failed,duration_s,total_tok_s,mean_ttft_ms,p99_ttft_ms" > "$SUMMARY"

extract() {  # $1=label  reads a vllm bench block on stdin -> appends a csv row
  awk -v c="$2" -v r="$3" -v ph="$4" '
    /Successful requests:/ {s=$NF}
    /Failed requests:/     {f=$NF}
    /Benchmark duration/   {d=$NF}
    /Total token throughput/ {t=$NF}
    /Mean TTFT/            {mt=$NF}
    /P99 TTFT/             {pt=$NF}
    END {printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n", c,r,ph,s,f,d,t,mt,pt}'
}

for N in $CONCURRENCIES; do
  # per-round seed = N (unless SEED was set explicitly); same across run1/run2
  round_seed="${SEED:-$N}"
  for R in 1 2; do
    phase=$([ "$R" = 1 ] && echo cold || echo warm)
    echo "########## concurrency=$N (num-prompts=$N)  run=$R [$phase]  seed=$round_seed ##########"
    raw="$OUTDIR/raw_N${N}_run${R}.json"
    out="$(vllm bench serve \
      --base-url "$BASE_URL" --model "$MODEL" --tokenizer "$TOKENIZER" \
      --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
      --save-result --result-dir "$OUTDIR" --result-filename "raw_N${N}_run${R}.json" \
      --ignore-eos --seed "$round_seed" --max-concurrency "$N" --num-prompts "$N" 2>&1)"
    printf '%s\n' "$out" | grep -iE \
      "Successful requests|Failed requests|Benchmark duration|Total token throughput|Mean TTFT|Median TTFT|P99 TTFT"
    printf '%s\n' "$out" | extract x "$N" "$R" "$phase" >> "$SUMMARY"
  done
done

# ---- final cold/warm table --------------------------------------------------
echo
echo "================ SUMMARY  (cold = run1, warm = run2) ================"
if command -v column >/dev/null 2>&1; then
  column -t -s, "$SUMMARY"
else
  # portable fallback when util-linux `column` isn't installed (slim images)
  awk -F, '{for(i=1;i<=NF;i++)printf "%-14s",$i; print ""}' "$SUMMARY"
fi
echo
echo "results: $OUTDIR"
echo "  manifest.txt   - run metadata"
echo "  summary.csv    - parsed cold/warm table"
echo "  raw_N*_run*.json - vllm bench serve raw output per point"
