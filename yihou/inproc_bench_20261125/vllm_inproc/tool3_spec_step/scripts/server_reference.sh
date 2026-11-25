#!/usr/bin/env bash
# Real-server side of the equivalence check: `vllm serve` with llying's decode-leg engine flags (presets/
# dsv41_llying_decode.args + presets/dsv41_decode.env, same DSpark synthetic-acceptance config the bench builds,
# DP2 TP1) on the bench GPUs, with vLLM's own DecodeBenchConnector instead of Mooncake: a new request's first ISL-1
# tokens of KV are filled with dummy values and its last prompt token is decoded, i.e. the PD decode-leg path without
# a prefill node (the vLLM analogue of the GLM-5.2 fake-input decode server). Then a fixed-length, ignore_eos client
# (`vllm bench serve`, random dataset, --random-range-ratio 0 so ISL/OSL are exact) at one concurrency.
# Writes $WORK_DIR/iterations/NAME/{server.log,client.json,client.log,server_reference.json}.
# The bench container must be idle: this uses the same GPUs. Stops (and --rm removes) only its own container.
# Usage: CONC=128 ISL=32768 OSL=1024 NUM=512 WARMUP_REQ=0 GPUS=2,3 bash server_reference.sh NAME
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/lib_v3.sh"
NAME=${1:?usage: server_reference.sh NAME}
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || v3_die "bad NAME"
CONC=${CONC:-128}; ISL=${ISL:-32768}; OSL=${OSL:-1024}; NUM=${NUM:-512}; WARMUP_REQ=${WARMUP_REQ:-0}
ACCEPT=${ACCEPT:-3.51}; DP_SIZE=${DP_SIZE:-2}; TP_SIZE=${TP_SIZE:-1}
FILL=${FILL:-'{"fill_mean":0.015,"fill_std":0.0}'}
# CONNECTOR=canonical-decode-bench (default): the DecodeBenchConnector subclass in connector/canonical_decode_bench.py
#   (the stock one makes the engine refuse fp8_e4m3 KV on ROCm; see that file). CONNECTOR=none: no connector, real
#   prefill of every request (expensive at long ISL).
CONNECTOR=${CONNECTOR:-canonical-decode-bench}
GPUS=${GPUS:?set GPUS, e.g. 2,3}; PORT=${PORT:-6310}
SRV=${SRV:-yihou-inproc-v3srv}
IMAGE=${IMAGE:-infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2}
MODEL_DIR=${MODEL_DIR:-/data/yihou/dsv41_vllm_pd_20261119/models/DeepSeek-V4.1-Flash}
CACHE_DIR=${CACHE_DIR:-/data/yihou/inproc_bench_20261125/cache_v3}
[[ "$SRV" =~ ^yihou-inproc-v3[A-Za-z0-9_-]*$ ]] || v3_die "server container must be yihou-inproc-v3*"
OUT="$WORK_DIR/iterations/$NAME"
[[ ! -e "$OUT" ]] || v3_die "Refusing existing iteration: $OUT"
docker inspect "$SRV" >/dev/null 2>&1 && v3_die "$SRV already exists"
[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" = "$PINNED_DIGEST" ] || v3_die "image is not the pinned digest"
mkdir -p "$OUT"
ENVS=(PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 HOME=/cache/home TMPDIR=/cache/tmp HF_HOME=/cache/hf XDG_CACHE_HOME=/cache/xdg
  TRITON_CACHE_DIR=/cache/triton AITER_JIT_DIR=/cache/aiter_jit TORCHINDUCTOR_CACHE_DIR=/cache/inductor
  VLLM_CACHE_ROOT=/cache/vllm ROCR_VISIBLE_DEVICES=$GPUS)
while IFS= read -r line; do ENVS+=("$line"); done < <(grep -hv '^\s*#' "$KIT_DIR/vllm_inproc/tool3_spec_step/presets/dsv41_decode.env" | grep -v '^\s*$')
D=(docker run -d --rm --name "$SRV" --label owner=yihou --label kit=inproc_bench --network host --ipc host --pid host
   --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined
   --shm-size 64g --ulimit nofile=1048576:1048576 --ulimit memlock=-1
   -v "$MODEL_DIR:/models/DeepSeek-V4.1-Flash:ro" -v "$CACHE_DIR:/cache" -v "$WORK_DIR:/out" -v "$KIT_DIR:/kit:ro"
   -e PYTHONPATH=/kit/vllm_inproc/tool3_spec_step/connector)
for e in "${ENVS[@]}"; do D+=(-e "$e"); done
mapfile -t PRESET < <(v3_expand_args "$KIT_DIR/vllm_inproc/tool3_spec_step/presets/dsv41_llying_decode.args")
SPEC=$(printf '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","enable_adaptive_verification":false,"synthetic_acceptance_length":%s}' "$ACCEPT")
case "$CONNECTOR" in
  canonical-decode-bench) KVT=$(printf '{"kv_connector":"CanonicalDecodeBenchConnector","kv_connector_module_path":"canonical_decode_bench","kv_role":"kv_both","kv_connector_extra_config":%s}' "$FILL");;
  decode-bench) KVT=$(printf '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_connector_extra_config":%s}' "$FILL");;
  none) KVT="";;
  *) v3_die "CONNECTOR must be canonical-decode-bench | decode-bench | none";;
esac
SERVER=(vllm serve /models/DeepSeek-V4.1-Flash --served-model-name deepseek-ai/DeepSeek-V4.1-Flash --host 127.0.0.1 --port "$PORT"
        --trust-remote-code --tensor-parallel-size "$TP_SIZE" --data-parallel-size "$DP_SIZE" --seed 1234
        --speculative-config "$SPEC" ${KVT:+--kv-transfer-config "$KVT"} "${PRESET[@]}"
        --api-server-count 1 --enable-logging-iteration-details)
# --api-server-count 1: the multi-API-server default for DP2 printed no stat-logger lines (v3_srv2_shapeA_u173k);
# --enable-logging-iteration-details: one log line per engine iteration with its elapsed time (server step time).
printf -v SERVER_CMD '%q ' "${SERVER[@]}"
printf '%s\n' "$SERVER_CMD" > "$OUT/server_command.txt"
printf '{"started_at":"%s","conc":%s,"isl":%s,"osl":%s,"num":%s,"warmup":%s,"gpus":"%s","connector":"%s"}\n' "$(date -Is)" "$CONC" "$ISL" "$OSL" \
    "$NUM" "$WARMUP_REQ" "$GPUS" "$CONNECTOR" > "$OUT/launch_started.json"
"${D[@]}" --entrypoint bash "$IMAGE" -c "$SERVER_CMD > /out/iterations/$NAME/server.log 2>&1" > "$OUT/container_id.txt"
for _ in $(seq 1 240); do
    curl -s -o /dev/null -f "http://127.0.0.1:$PORT/health" && break
    docker inspect "$SRV" >/dev/null 2>&1 || v3_die "server container exited; see $OUT/server.log"
    sleep 15
done
curl -s -o /dev/null -f "http://127.0.0.1:$PORT/health" || { docker stop -t 60 "$SRV" >/dev/null; v3_die "server never healthy"; }
CLIENT=(vllm bench serve --backend vllm --host 127.0.0.1 --port "$PORT" --model deepseek-ai/DeepSeek-V4.1-Flash
        --tokenizer /models/DeepSeek-V4.1-Flash --trust-remote-code
        --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 0
        --ignore-eos --num-prompts "$NUM" --max-concurrency "$CONC" --request-rate inf --num-warmups "$WARMUP_REQ" --seed 1234
        --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir "/out/iterations/$NAME" --result-filename client.json)
printf '%q ' "${CLIENT[@]}" > "$OUT/client_command.txt"
set +e
docker exec -w / "$SRV" "${CLIENT[@]}" > "$OUT/client.log" 2>&1
RC=$?
set -e
docker stop -t 60 "$SRV" > /dev/null || true
python3 "$HERE/server_tpot.py" "$OUT" --local-batch $(( CONC / DP_SIZE )) > "$OUT/server_reference.json" || true
printf '{"exit_code":%s}\n' "$RC" > "$OUT/launch_status.json"
cat "$OUT/server_reference.json"
exit "$RC"
