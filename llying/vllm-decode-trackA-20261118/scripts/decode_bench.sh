#!/usr/bin/env bash
# Decode-dominated bench against one aggregated engine: N requests sent at once (random IN-token prompts, OUT tokens,
# ignore EOS), so after the prefill wave every request decodes together. Saves the detailed result (per-request ITLs,
# for decode_window.py) and, when the engine runs with --profiler-config, takes one torch trace of TRACE_S seconds
# once the engine is decode-only (trace_when_decoding.sh).
#   decode_bench.sh CONTAINER PORT OUTDIR [N=104] [IN=65536] [OUT=4096]
# Env: TRACE_S (3; 0 = no trace), SEED (0), IMAGE.
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackA-20261118}
ctr=$1 port=$2 out=$3 n=${4:-104} in=${5:-65536} outlen=${6:-4096}
IMAGE=${IMAGE:-vllm/vllm-openai-rocm:v0.34.0}
MODEL=${MODEL:-/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4}
TRACE_S=${TRACE_S:-3}
mkdir -p "$out" && chmod 777 "$out"
t_start=$(date -u +%Y-%m-%dT%H:%M:%S)
watcher=
if [[ "$TRACE_S" != 0 ]]; then
  bash "$R/scripts/trace_when_decoding.sh" "$port" "$TRACE_S" 3000 > "$out/trace.log" 2>&1 &
  watcher=$!
fi
docker run --rm --network host -v "$out:/out" -v "$MODEL:/model:ro" --entrypoint vllm "$IMAGE" bench serve \
  --backend vllm --base-url "http://127.0.0.1:$port" --endpoint /v1/completions --model glm5.2-mxfp4 \
  --tokenizer /model --trust-remote-code --dataset-name random --random-input-len "$in" --random-output-len "$outlen" \
  --random-range-ratio 0 --num-prompts "$n" --max-concurrency "$n" --ignore-eos --seed "${SEED:-0}" \
  --percentile-metrics ttft,tpot,itl,e2el --save-result --save-detailed --result-dir /out \
  --result-filename bench.json > "$out/bench.log" 2>&1
echo "bench rc=$? $(date -u +%T)" >> "$out/bench.log"
# a trace in flight finishes (stop_profile writes the files); a watcher still polling is no longer needed
if [[ -n "$watcher" ]] && ! grep -q start_profile "$out/trace.log"; then kill "$watcher" 2> /dev/null; fi
wait
docker logs --since "$t_start" "$ctr" 2>&1 | grep -a -E "Engine [0-9]+: Avg prompt" > "$out/engine-stats.log"
[[ -s "$out/bench.json" ]] && python3 "$R/scripts/decode_window.py" "$out/bench.json" | tee "$out/window.json"
exit 0
