#!/usr/bin/env bash
# InferenceX's own AgentX accuracy checks against a running deployment with REAL MTP acceptance (no synthetic acceptance),
# run exactly as InferenceX runs them (InferenceX main d490919c, infx/evals/EVALS.md):
#   1. GSM8K: `python3 -m infx.bench eval` = lm-eval local-chat-completions, full test set (1,319), 5-shot, temperature 0,
#      at the curve's highest benchmark concurrency; then `python3 -m infx.evals.validate_scores` (GSM8K floor 0.90,
#      strict-match and flexible-extract).
#   2. AL gate (report-only): `python3 -m infx.bench al-gate`, golden SPEED-Bench coding replay, realized AL from the
#      engine counters must sit in 0.95-1.05 x golden (GLM-5.2 MTP draft 4: 3.33).
# Both run in a throwaway container of the engine image (lm-eval is installed into its python, as InferenceX does).
#   eval.sh OUT_DIR ROUTER_URL CONC [ENGINE_METRICS_URLS]
# Env: IMAGE (engine image), MTP_N (4), SKIP_AL_GATE (0)
set -uo pipefail
. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/site/env.sh
out=${1:?out dir} url=${2:?router url} conc=${3:?conc} murls=${4:-}
IXM=/perf_apps/liyingli/bench_agentx/InferenceX-main-d490919c
IMAGE=${IMAGE:?IMAGE}
MTP_N=${MTP_N:-4}
mkdir -p "$out/gsm8k" "$out/al-gate" "$out/cache" && chmod -R 777 "$out"
# AIPerf (AL gate) binds ZMQ ipc:// sockets under TMPDIR: short, node-local.
axtmp=$(mktemp -d /tmp/lyl-axtmp.XXXXXX) && chmod 777 "$axtmp"
common=(--rm --network host --user "$(id -u):$(id -g)" -e HOME="$out/cache" -e XDG_CACHE_HOME="$out/cache"
    -e UV_CACHE_DIR="$out/cache/uv" -e TMPDIR=/ax-tmp -v "$axtmp:/ax-tmp" -v "$IXM:$IXM:ro" -v "$MODEL:$MODEL:ro" -v "$out:$out"
    -e MODEL="$MODEL" -e MODEL_PATH="$MODEL" -e MODEL_NAME="$SERVED" -e MODEL_PREFIX=glm5.2 -e FRAMEWORK=vllm
    -e PRECISION=fp4 -e SPEC_DECODING=mtp -e EVAL_ONLY=true -e IS_MULTINODE=false -e OPENAI_API_KEY=EMPTY
    -e RUNNER_TYPE=mi355x -e IMAGE="$IMAGE" -w "$IXM/inferencex-e2e" --entrypoint bash)
# The engine image installs packages as root into its system python: give the throwaway container a writable venv copy.
venv='python3 -m venv --system-site-packages "$HOME/venv" >/dev/null && . "$HOME/venv/bin/activate"'
echo "$(ts) GSM8K at C$conc against $url"
docker run --name "vpd-eval-gsm8k" "${common[@]}" "$IMAGE" -c "$venv && export PYTHONSAFEPATH=1 PYTHONPATH=\$PWD && \
    python3 -m infx.bench eval --endpoint '$url' --concurrency '$conc' --stage-to '$out/gsm8k' && \
    python3 -m infx.evals.validate_scores --meta-env '$out/gsm8k/meta_env.json' --results-glob '$out/gsm8k/results*.json'" \
    > "$out/gsm8k.log" 2>&1
g_rc=$?
echo "$(ts) GSM8K rc=$g_rc: $(grep -a -E 'exact_match|PASS|FAIL' "$out/gsm8k.log" | tail -4 | tr '\n' ' ')"
a_rc=skipped
if [[ "${SKIP_AL_GATE:-0}" != 1 ]]; then
    # The golden SPEED-Bench coding prompts are prepared first (python -m infx.bench.speedbench; sha256-checked by the gate).
    docker run --name "vpd-eval-algate" "${common[@]}" -e AL_GATE=true -e AL_GATE_CURVE=glm-5.2-fp8__mtp__native__default-default \
        -e AL_GATE_DRAFT_TOKENS="$MTP_N" -e AL_GATE_DATASET="$out/al-gate/speedbench_coding.jsonl" -e AL_GATE_OUTPUT_DIR="$out/al-gate" \
        -e THINKING_MODE=thinking_on -e RESULT_FILENAME=al_gate -e SERVED_MODEL_NAME="$SERVED" -e AIPERF_SERVER_URL="$url" \
        -e AIPERF_SERVER_METRICS_URLS="$(for m in $murls; do printf '%s/metrics,' "$m"; done | sed 's/,$//')" \
        "$IMAGE" -c "$venv && export PYTHONSAFEPATH=1 PYTHONPATH=\$PWD && python3 -m infx.bench.speedbench --output-dir '$out/al-gate' && \
        python3 -m infx.bench al-gate" > "$out/al-gate.log" 2>&1
    a_rc=$?
    echo "$(ts) AL gate rc=$a_rc: $(grep -a -i -E 'verdict|realized|golden' "$out/al-gate.log" | tail -3 | tr '\n' ' ')"
fi
echo "gsm8k_rc=$g_rc al_gate_rc=$a_rc" > "$out/eval-rc.txt"
exit $g_rc
