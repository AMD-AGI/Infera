#!/usr/bin/env bash
# Score this deployment with sglang's own evaluator. RUNS ON THE COMPUTE NODE.
#
# Ported from examples/sglang_1p1d_glm5.2/engine/lm_eval.sh. The evaluator is
# `sglang.test.run_eval`, which ships inside the engine image and talks OpenAI
# chat completions, so nothing is installed into the container. Measured on this
# image: it takes every flag below, including --thinking-mode.
#
# Three things are load-bearing and each fails as a NUMBER rather than an error:
#
#   --thinking-mode glm-45   the engine runs with --reasoning-parser glm45, so
#                            `content` holds only the answer. Without the matching
#                            chat_template_kwargs a healthy deployment scores 0.00.
#   --gsm8k-data-path        the dataset is staged from the host rather than
#                            fetched, so a container restart does not re-fetch it
#                            and an offline node can be primed by hand. A short
#                            file is a truncated download, and it moves the score.
#   the noise filter runs     HERE, not remotely. Piping into grep on the far side
#                            of a shell without pipefail lets grep's exit code
#                            mask a crashed run_eval.
#
# Scored questions are COUNTED out of the html report rather than assumed from
# --num-examples: gsm8k scores 1314 of GSM8K's 1319 rows and mixed_prefix_gsm8k
# scores 1299, because the evaluator slices the few-shot examples out of the
# evaluation set. The interval needs that number and the result json does not
# carry it.
set -uo pipefail

URL="${URL:?}"
SERVED="${SERVED:?}"
CTR="${CTR:?}"
OUT="${OUT:?OUT=directory to collect results into}"
GSM8K_SRC="${GSM8K_SRC:?}"
EVALS="${EVALS:-gsm8k mixed_prefix_gsm8k}"
NUM_EXAMPLES="${NUM_EXAMPLES:-200}"
THREADS="${THREADS:-32}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TEMP="${TEMP:-0.0}"
TOP_P="${TOP_P:-1.0}"
REPEAT="${REPEAT:-1}"
THINKING="${THINKING:---thinking-mode glm-45}"
# Where sglang's source tree lives inside the image, prepended to PYTHONPATH for
# the evaluator only.
#
# Not always redundant. On rocm/sgl-dev:v0.5.18 the installed `sglang` is a
# NAMESPACE package -- `import sglang` succeeds and `sglang.__file__` is None --
# and `sglang.test` is not part of it, so `python3 -m sglang.test.run_eval` dies
# with "No module named sglang.test.run_eval" even though
# /sgl-workspace/sglang/python/sglang/test/run_eval.py is right there. Adding the
# source root fixes it and is inert on an image where sglang is installed
# properly, because it is the same tree.
SGLANG_SRC="${SGLANG_SRC:-/sgl-workspace/sglang/python}"
# `none` means the model has no reasoning parser; see mix_worker.sh.
[ "$THINKING" = none ] && THINKING=""

mkdir -p "$OUT"
: > "$OUT/.index"

echo "===== dataset ====="
if [ ! -r "$GSM8K_SRC" ]; then
  echo "  ABORT: GSM8K test split not readable: $GSM8K_SRC"
  exit 1
fi
rows="$(wc -l < "$GSM8K_SRC")"
if [ "$rows" -ne 1319 ]; then
  echo "  WARN: the GSM8K test split has $rows rows, expected 1319 — scores will not be comparable"
fi
docker cp "$GSM8K_SRC" "$CTR:/tmp/gsm8k_test.jsonl" >/dev/null \
  || { echo "  ABORT: could not stage the dataset into $CTR"; exit 1; }
echo "  staged $rows rows into $CTR"

rc_all=0
for name in $EVALS; do
  json="/tmp/${name}_${SERVED//\//_}.json"
  html="/tmp/${name}_${SERVED//\//_}.html"
  echo "===== $name ($NUM_EXAMPLES questions, $THREADS concurrent, repeat=$REPEAT) ====="

  # A run that dies before scoring would otherwise leave the PREVIOUS run's
  # files to be copied out and reported as this run's score.
  docker exec "$CTR" rm -f "$json" "$html" >/dev/null 2>&1

  data_arg=""
  case "$name" in
    gsm8k|mixed_prefix_gsm8k) data_arg="--gsm8k-data-path /tmp/gsm8k_test.jsonl" ;;
  esac
  n_arg=""
  [ -n "$NUM_EXAMPLES" ] && n_arg="--num-examples $NUM_EXAMPLES"

  started=$(date +%s)
  docker exec -e PYTHONPATH="$SGLANG_SRC:${PYTHONPATH:-}" "$CTR" python3 -m sglang.test.run_eval \
      --base-url "$URL" --model "$SERVED" --eval-name "$name" \
      $n_arg --num-threads "$THREADS" --repeat "$REPEAT" \
      --max-tokens "$MAX_TOKENS" --temperature "$TEMP" --top-p "$TOP_P" \
      $THINKING $data_arg 2>&1 \
    | grep --line-buffered -vE '^\[aiter\]|[0-9]+%\|' \
    | tee "$OUT/$name.console.log"
  eval_rc="${PIPESTATUS[0]}"
  elapsed=$(( $(date +%s) - started ))

  if [ "$eval_rc" != "0" ]; then
    echo "  $name failed (rc=$eval_rc) — continuing with the remaining evals"
    rc_all=1
    continue
  fi

  if ! docker cp "$CTR:$json" "$OUT/" >/dev/null 2>&1; then
    echo "  $name wrote no result file — it exited before scoring"
    rc_all=1
    continue
  fi
  docker cp "$CTR:$html" "$OUT/" >/dev/null 2>&1 || true

  scored="$(grep -c 'Correct Answer' "$OUT/$(basename "$html")" 2>/dev/null || echo 0)"
  printf '%s\t%s\t%s\t%s\n' "$name" "$scored" "$(basename "$json")" "$elapsed" >> "$OUT/.index"
  echo "  $name: $scored scored in ${elapsed}s -> $(basename "$json")"
done

echo "===== index ====="
cat "$OUT/.index"
[ "$rc_all" = "0" ] && echo "LM_EVAL_OK" || echo "LM_EVAL_PARTIAL"
# Exit 0 even on a partial run: the eval result is evidence for `compare` and for
# `check_acceptance` to weigh, and taking the measurement task down here would
# discard the trace replay that follows.
exit 0
