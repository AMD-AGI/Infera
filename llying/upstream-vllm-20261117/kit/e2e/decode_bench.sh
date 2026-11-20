#!/usr/bin/env bash
# Decode-dominated bench against one server, as track A's decode_bench.sh without the torch trace: N requests sent at
# once (random IN-token prompts, OUT tokens, ignore EOS); the decode step is read from the detailed result by track A's
# decode_window.py (read-only), over the window where every request is decoding.
#   decode_bench.sh PORT OUTDIR [N=104] [IN=65536] [OUT=4096]
set -uo pipefail
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
A=/home/liyingli/bench_agentx/vllm-trackA-20261118
port=$1 out=$2 n=${3:-104} in=${4:-65536} outlen=${5:-4096}
model=$(model_dir) || { echo "model not staged"; exit 1; }
mkdir -p "$out" && chmod 777 "$out"
docker run --rm --label "up.job=${JOB:-0}" --network host -v "$out:/out" -v "$model:/model:ro" --entrypoint vllm \
  "$IMG_NIGHTLY" bench serve --backend vllm --base-url "http://127.0.0.1:$port" --endpoint /v1/completions \
  --model glm5.2-mxfp4 --tokenizer /model --trust-remote-code --dataset-name random --random-input-len "$in" \
  --random-output-len "$outlen" --random-range-ratio 0 --num-prompts "$n" --max-concurrency "$n" --ignore-eos \
  --seed "${SEED:-0}" --percentile-metrics ttft,tpot,itl,e2el --save-result --save-detailed --result-dir /out \
  --result-filename bench.json > "$out/bench.log" 2>&1
echo "bench rc=$? $(date -u +%T)" >> "$out/bench.log"
[[ -s "$out/bench.json" ]] && python3 "$A/scripts/decode_window.py" "$out/bench.json" --bins | tee "$out/window.json"
exit 0
