#!/usr/bin/env bash
# Real-server side of the equivalence check: the InferenceX #5034 SGLang server (scripts/sgl/s2_ix5034_c256.sh argv and
# env, router dropped, overlap schedule ON as served) on the bench GPUs, then a fixed-length, ignore_eos client
# (sglang.benchmark.serving, random dataset, --random-range-ratio 1, --tokenize-prompt so ISL is exact) at one
# concurrency. Writes $WORK_DIR/iterations/NAME/{server.log,client.jsonl,client.log,server_reference.json}.
# Port of the GLM-5.2 reverse validation (which used a fake-input decode server; tool 1 is waived for DSv4.1).
# The bench container must be idle: this uses the same GPUs. Stops (and --rm removes) only its own container.
# Usage: CONC=64 ISL=8192 OSL=1024 NUM=256 WARMUP_REQ=64 bash server_reference.sh NAME
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/lib_t3.sh"
NAME=${1:?usage: server_reference.sh NAME}
[[ "$NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || t3_die "bad NAME"
CONC=${CONC:-64}; ISL=${ISL:-8192}; OSL=${OSL:-1024}; NUM=${NUM:-256}; WARMUP_REQ=${WARMUP_REQ:-64}
GPUS=${GPUS:-0,1,2,3}; PORT=${PORT:-6300}; NCCL_PORT=${NCCL_PORT:-17600}
SRV=${SRV:-yihou-inproc-t3srv}
IMAGE=${IMAGE:-lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111}
MODEL_DIR=${MODEL_DIR:-/data/yihou/dsv41_vllm_pd_20261119/models/DeepSeek-V4.1-Flash}
CACHE_DIR=${CACHE_DIR:-/data/yihou/inproc_bench_20261125/cache_t3}
[[ "$SRV" =~ ^yihou-inproc-[A-Za-z0-9_-]+$ ]] || t3_die "server container must be yihou-inproc-*"
OUT="$WORK_DIR/iterations/$NAME"
[[ ! -e "$OUT" ]] || t3_die "Refusing existing iteration: $OUT"
docker inspect "$SRV" >/dev/null 2>&1 && t3_die "$SRV already exists"
[ "$(docker image inspect -f '{{.Id}}' "$IMAGE")" = "$PINNED_DIGEST" ] || t3_die "image is not the pinned digest"
mkdir -p "$OUT"
ENVS=(PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SGLANG_TIMEOUT_KEEP_ALIVE=900 SGLANG_DEFAULT_THINKING=1 SGLANG_DSV41_REASONING_EFFORT=high
  HOME=/cache/home TMPDIR=/cache/tmp HF_HOME=/cache/hf XDG_CACHE_HOME=/cache/xdg TRITON_CACHE_DIR=/cache/triton
  AITER_JIT_DIR=/cache/aiter_jit TORCHINDUCTOR_CACHE_DIR=/cache/inductor ROCR_VISIBLE_DEVICES=$GPUS)
while IFS= read -r line; do ENVS+=("$line"); done < <(grep -hv '^\s*#' "$KIT_DIR/sglang_inproc/presets/dsv41.env" \
    "$KIT_DIR/sglang_inproc/presets/dsv41_dspark_sim351.env" | grep -v '^\s*$')
D=(docker run -d --rm --name "$SRV" --label owner=yihou --label kit=inproc_bench --network host --ipc host --pid host
   --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined
   --shm-size 64g --ulimit nofile=1048576:1048576 --ulimit memlock=-1
   -v "$MODEL_DIR:/models/DeepSeek-V4.1-Flash:ro" -v "$CACHE_DIR:/cache" -v "$WORK_DIR:/out")
for e in "${ENVS[@]}"; do D+=(-e "$e"); done
SERVER="python3 -m sglang.launch_server --model-path /models/DeepSeek-V4.1-Flash --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
 --host 127.0.0.1 --port $PORT --nccl-port $NCCL_PORT --chunked-prefill-size 65536 --cuda-graph-backend-prefill disabled --cuda-graph-max-bs-decode 128 \
 --data-parallel-size 4 --enable-decoder-swa-bounded-replay --enable-dp-attention --enable-dp-attention-local-control-broadcast --enable-dp-lm-head \
 --enable-metrics --enable-prefill-delayer --enforce-shared-experts-fusion --expert-parallel-size 4 --fp8-gemm-backend aiter \
 --image-processor-backend pil --load-balance-method total_requests --max-running-requests 256 --mem-fraction-static 0.85 \
 --prefill-decode-interval 40 --prefill-delayer-token-usage-low-watermark 0.85 --reasoning-parser auto --schedule-policy shortest-prefill-first \
 --speculative-algorithm DSPARK --speculative-dspark-block-size 5 --swa-prefix-tails 12288 --tensor-parallel-size 4 --tokenizer-backend fastokens \
 --tokenizer-worker-num 2 --tool-call-parser auto --trust-remote-code --watchdog-timeout 3600"
printf '%s\n' "$SERVER" > "$OUT/server_command.txt"
printf '{"started_at":"%s","conc":%s,"isl":%s,"osl":%s,"num":%s,"warmup":%s,"gpus":"%s"}\n' "$(date -Is)" "$CONC" "$ISL" "$OSL" \
    "$NUM" "$WARMUP_REQ" "$GPUS" > "$OUT/launch_started.json"
"${D[@]}" --entrypoint bash "$IMAGE" -c "$SERVER > /out/iterations/$NAME/server.log 2>&1" > "$OUT/container_id.txt"
for _ in $(seq 1 240); do
    curl -s -o /dev/null -f "http://127.0.0.1:$PORT/health" && break
    docker inspect "$SRV" >/dev/null 2>&1 || t3_die "server container exited; see $OUT/server.log"
    sleep 15
done
curl -s -o /dev/null -f "http://127.0.0.1:$PORT/health" || { docker stop -t 60 "$SRV" >/dev/null; t3_die "server never healthy"; }
CLIENT=(python3 -m sglang.benchmark.serving --backend sglang --host 127.0.0.1 --port "$PORT"
        --model /models/DeepSeek-V4.1-Flash --tokenizer /models/DeepSeek-V4.1-Flash
        --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1.0 --tokenize-prompt
        --num-prompts "$NUM" --max-concurrency "$CONC" --request-rate inf --warmup-requests "$WARMUP_REQ" --seed 1234
        --output-file "/out/iterations/$NAME/client.jsonl")
printf '%q ' "${CLIENT[@]}" > "$OUT/client_command.txt"
set +e
docker exec -w / "$SRV" "${CLIENT[@]}" > "$OUT/client.log" 2>&1
RC=$?
set -e
docker stop -t 60 "$SRV" > /dev/null || true
python3 "$HERE/server_tpot.py" "$OUT" > "$OUT/server_reference.json" || true
printf '{"exit_code":%s}\n' "$RC" > "$OUT/launch_status.json"
cat "$OUT/server_reference.json"
exit "$RC"
