#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: check that this deployment serves the model CORRECTLY, not just quickly.
# why : smoke.sh proves the features are on and bench.sh proves the stack is fast; neither
#       reads a single answer. MXFP4 weights, an fp8_e4m3 KV cache, sparse attention on ROCm,
#       KV moved by mooncake, EAGLE speculation and a radix cache shared between requests are
#       six independent ways to serve fluent, fast, WRONG tokens — and all six look green in
#       smoke and bench.
# how : DO NOT run this directly — run it through cluster/<your-cluster>.sh:
#         bash cluster/<your-cluster>.sh lm_eval probe   # ~40 s, gates the rest
#         bash cluster/<your-cluster>.sh lm_eval quick   # ~2 min, GSM8K 200
#         bash cluster/<your-cluster>.sh lm_eval full    # ~13 min, the full set twice
#         bash cluster/<your-cluster>.sh lm_eval         # probe, then quick
#
# The evaluator is sglang.test.run_eval, which ships inside the engine image (sglang 0.5.17)
# and talks OpenAI /v1/chat/completions, so nothing is installed into $CTR. Read "Accuracy"
# in the README before quoting a number from here: it says what a score can and cannot show.
#
# Knobs (all optional):
#   NUM_EXAMPLES   questions per eval. quick defaults to 200, full to everything
#   THREADS=32     concurrency, and an accuracy variable too — a score that holds at 1 and
#                  drops at 256 indicts batching rather than the model
#   MAX_TOKENS     generation cap, default 2048 (GLM-5.2 bills reasoning against it)
#   TEMP=0.0       greedy, and NOT byte-reproducible here — see tools/probe_accuracy.py
#   TOP_P=1.0
#   REPEAT=1       run each eval N times and report the spread
#   FULL_EVALS     what `full` runs. Default: gsm8k mixed_prefix_gsm8k
#
# Three of run_eval's eleven evals cannot run here: humaneval (no human_eval package in the
# image), math (grades with a second model, needs an OpenAI key) and mmmu (vision). gsm8k,
# mixed_prefix_gsm8k, mmlu, gpqa, aime25 and mgsm_en were each measured against this image;
# mgsm and longbench_v2 were not tried. gpqa and aime25 need long chains of thought and score
# near chance at the 2048 default, so raise MAX_TOKENS a long way before believing them.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"

require_env PREFILL_NODE; require_env PREFILL_IP

SSH_CMD="${SSH_CMD:-ssh -o StrictHostKeyChecking=no}"
on(){ local h="$1"; shift; $SSH_CMD "$h" "$*" </dev/null; }

URL="http://$PREFILL_IP:$ROUTER_PORT"
KIT="$(cd "$DIR/.." && pwd)"
OUT="${LM_EVAL_OUT:-$KIT/lm_eval}"
DATA_DIR="$OUT/data"
RUN_DIR="$OUT/$(date +%Y%m%d_%H%M%S)"

THREADS="${THREADS:-32}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMP="${TEMP:-0.0}"
TOP_P="${TOP_P:-1.0}"
REPEAT="${REPEAT:-1}"

# mixed_prefix_gsm8k asks the same questions as gsm8k behind partially-shared few-shot
# prefixes. Pairing them is the point of `full`: an absolute score has no baseline, but the
# gap between these two measures prefix reuse and needs none.
FULL_EVALS="${FULL_EVALS:-gsm8k mixed_prefix_gsm8k}"

# The leg runs with --reasoning-parser glm45, so `content` holds only the answer.
# --thinking-mode emits the matching chat_template_kwargs; without it an eval scores 0.00
# against a healthy deployment.
THINKING="${THINKING:---thinking-mode glm-45}"

# run_eval fetches this to /tmp on first use. Staged on the host instead, so a container
# restart does not re-fetch it and an offline node can be primed by hand.
GSM8K_URL="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

# ---- container plumbing --------------------------------------------------------------------
# docker cp rather than a bind mount: common.sh mounts INFERA_SRC read-only and only when it
# is set, and this has to work when it is not. $PREFILL_NODE reads $KIT directly — the
# wrappers already require KIT_DIR to resolve identically on both nodes.
stage(){
  on "$PREFILL_NODE" "test -r '$1'" || die "not readable on $PREFILL_NODE: $1"
  on "$PREFILL_NODE" "docker cp '$1' '$CTR:$2'" >/dev/null \
    || die "could not copy $1 into $CTR on $PREFILL_NODE"
}

ensure_dataset(){
  mkdir -p "$DATA_DIR" || die "could not create $DATA_DIR"
  if [ ! -s "$DATA_DIR/test.jsonl" ]; then
    log "fetching the GSM8K test split -> $DATA_DIR/test.jsonl"
    curl -sSfL -o "$DATA_DIR/test.jsonl" "$GSM8K_URL" \
      || die "could not download GSM8K. Place the file at $DATA_DIR/test.jsonl by hand."
  fi
  # A short file is a truncated download, which would shrink every gsm8k run and move its score.
  local n; n="$(wc -l < "$DATA_DIR/test.jsonl")"
  [ "$n" -eq 1319 ] || warn "GSM8K test split has $n rows, expected 1319 — scores will not be comparable"
  stage "$DATA_DIR/test.jsonl" "/tmp/gsm8k_test.jsonl"
}

# ---- probe -----------------------------------------------------------------------------------
probe(){
  log "=== probe: is a score from this deployment interpretable? ==="
  log "  ~30 requests, about a minute. Reads answers, not just status codes."
  stage "$DIR/tools/probe_accuracy.py" "/tmp/probe_accuracy.py"
  on "$PREFILL_NODE" "docker exec $CTR python3 /tmp/probe_accuracy.py \
      --url '$URL' --model '$SERVED' \
      --max-tokens $MAX_TOKENS --temperature $TEMP --top-p $TOP_P"
}

# ---- one eval ---------------------------------------------------------------------------------
# Questions scored, counted out of run_eval's html report. The interval needs this number and
# the result json does not carry it. Counted rather than assumed because the dataset size is
# the wrong answer: GSM8K ships 1319 rows but gsm8k scores 1314 and mixed_prefix_gsm8k 1299 —
# GSM8KEval slices the few-shot examples off the evaluation set so they cannot leak into it.
# Nor is it a success count: a request that fails past its retries is scored 0 and stays in
# the report. `Correct Answer` is HTML_JINJA's, shared by every eval used here.
count_scored(){ grep -c 'Correct Answer' "$1" 2>/dev/null; }

# run_eval writes /tmp/<eval>_<model>.{json,html} inside $CTR. Both come out: the json is the
# number, the html is every prompt and completion — the only place to see WHY a score moved.
run_one(){
  local name="$1" limit="${NUM_EXAMPLES:-}" n_arg="" data_arg="" scored
  local json="/tmp/${name}_${SERVED//\//_}.json"
  local html="/tmp/${name}_${SERVED//\//_}.html"
  [ -n "$limit" ] && n_arg="--num-examples $limit"
  # Both gsm8k variants read the staged file; the others fetch their own.
  case "$name" in gsm8k|mixed_prefix_gsm8k) data_arg="--gsm8k-data-path /tmp/gsm8k_test.jsonl" ;; esac

  log "--- $name (${limit:-all} questions, $THREADS concurrent, repeat=$REPEAT) ---"
  # Otherwise a run that dies before scoring leaves the previous run's files to be copied out.
  on "$PREFILL_NODE" "docker exec $CTR rm -f '$json' '$html'" >/dev/null 2>&1
  # The noise filter runs HERE: piping into grep remotely puts the pipeline in a shell without
  # pipefail, so grep's exit code masks a crashed run_eval. Locally, `set -o pipefail` shows it.
  on "$PREFILL_NODE" "docker exec $CTR python3 -m sglang.test.run_eval \
      --base-url '$URL' --model '$SERVED' --eval-name $name \
      $n_arg --num-threads $THREADS --repeat $REPEAT \
      --max-tokens $MAX_TOKENS --temperature $TEMP --top-p $TOP_P \
      $THINKING $data_arg 2>&1" \
    | grep --line-buffered -vE '^\[aiter\]|[0-9]+%\|' \
    || { warn "  $name failed — see its output above; continuing with the remaining evals"
         return 1; }

  mkdir -p "$RUN_DIR"
  on "$PREFILL_NODE" "docker cp '$CTR:$json' '$RUN_DIR/'" >/dev/null 2>&1 \
    || { warn "  $name wrote no result file — it exited before scoring"; return 1; }
  on "$PREFILL_NODE" "docker cp '$CTR:$html' '$RUN_DIR/'" >/dev/null 2>&1 || true
  scored="$(count_scored "$RUN_DIR/$(basename "$html")")"
  printf '%s\t%s\t%s\n' "$name" "${scored:-}" "$(basename "$json")" >> "$RUN_DIR/.index"
}

# ---- subcommands -------------------------------------------------------------------------------
# A separate script rather than an inlined heredoc, unlike the small python fragments elsewhere
# in this kit: it runs on THIS host, and keeping it addressable means any past result directory
# can be re-summarised, or two of them compared, without re-running a 13-minute eval:
#   python3 engine/tools/summarise_eval.py lm_eval/<baseline> lm_eval/<new>
summarise(){
  [ -d "$RUN_DIR" ] || { warn "no results to summarise"; return 1; }
  python3 "$DIR/tools/summarise_eval.py" "$RUN_DIR"
}

run_set(){
  local label="$1"; shift
  ensure_dataset
  log "=== $label: $* (${NUM_EXAMPLES:-all} questions each) ==="
  [ -n "${NUM_EXAMPLES:-}" ] || log "  the full GSM8K set is ~1300 questions — minutes, not a hang"
  for name in "$@"; do run_one "$name"; done
  log "=== $label results ==="
  summarise
  log "artifacts -> $RUN_DIR  (the .html files hold every prompt and completion)"
}

quick(){ NUM_EXAMPLES="${NUM_EXAMPLES:-200}"; run_set quick gsm8k; }

case "${1:-all}" in
  probe) probe ;;
  quick) quick ;;
  full)  shift; [ $# -gt 0 ] && FULL_EVALS="$*"; run_set full $FULL_EVALS ;;
  # probe gates quick: the failures it catches produce a NUMBER rather than an error, and that
  # number is indistinguishable from a real regression. Two statements rather than
  # `probe && quick || warn`, which would report a failing quick as a failing probe.
  all)   probe || { warn "probe failed — not spending minutes on a score that cannot be read"; exit 1; }
         quick ;;
  *)     die "unknown subcommand '$1'. Use: probe | quick | full [eval ...]" ;;
esac
