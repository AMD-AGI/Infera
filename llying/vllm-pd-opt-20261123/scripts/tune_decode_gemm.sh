#!/usr/bin/env bash
# Tune aiter BF16 GEMMs for the GLM-5.2 decode shapes with the vLLM image's own aiter and hipBLASLt (a tuned table is
# tied to those versions, so the sglang image's tables do not carry over). Runs on the node that owns the GPUs, as a step
# of a job. The JIT volume is the one vllm_up.sh mounts, so kernels the tuner builds are not rebuilt while serving.
#   tune_decode_gemm.sh SHAPES_TXT OUT_DIR [GPUS=0,1,2,3,4,5,6,7] [TIMEOUT_S=4500]
# Env: TUNE_BATCH (shapes per write of the table, default 16; the tuner's own default of 100 writes nothing for a long
# time), TUNE_HIPB (1 = also tune hipBLASLt solutions, about 43 s per shape; default 1).
# Writes OUT_DIR/glm52_decode_bf16_tuned_gemm.csv (+ prof_*.csv, tune.log, help.txt). Exit 0 if the table has rows.
# The container runs with --init: without it bash is PID 1, ignores the SIGTERM docker forwards on timeout and the
# tuner keeps running (a4 ran 25 min past its timeout).
set -uo pipefail
shapes=$1 out=$2 gpus=${3:-0,1,2,3,4,5,6,7} tmo=${4:-4500}
batch=${TUNE_BATCH:-16} hipb=${TUNE_HIPB:-1}
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
IMAGE=infera-vllm:v0340-0c5e5e89
image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
jit=vpd-aiter-jit-${image_id:7:12}
mkdir -p "$out" && chmod 777 "$out"
cp "$shapes" "$out/shapes.txt"
n=$(awk -F, '{print NF}' <<<"$gpus")
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
run() {
    timeout -k 60 "$tmo" docker run --rm --init --name vpd-gemmtune --network none --device /dev/kfd --device /dev/dri "${gids[@]}" \
        -e HIP_VISIBLE_DEVICES="$gpus" -v "$jit:/usr/local/lib/python3.12/dist-packages/aiter/jit" -v "$out:/w" -v "$B:$B:ro" \
        --entrypoint bash "$IMAGE" -c "$1"
    local rc=$?
    docker rm -f vpd-gemmtune >/dev/null 2>&1
    return $rc
}
A=/usr/local/lib/python3.12/dist-packages/aiter
M=/usr/local/lib/python3.12/dist-packages/aiter_meta
T=csrc/gemm_a16w16/gemm_a16w16_tune.py
run "cd $M && python3 $T --help" > "$out/help.txt" 2>&1
run "python3 $B/scripts/make_untuned_csv.py /w/shapes.txt $A/configs/bf16_untuned_gemm.csv /w/untuned.csv" > "$out/make-untuned.log" 2>&1
tuned=/w/glm52_decode_bf16_tuned_gemm.csv
hb=""; [[ "$hipb" == 1 ]] && hb="--with-hipblaslt"
t0=$(date +%s)
run "cd $M && python3 $T --input_file /w/untuned.csv --tuned_file $tuned -o2 /w/prof_glm52_decode_bf16_tuned_gemm.csv --mp $n --batch $batch --libtype all $hb" \
    > "$out/tune.log" 2>&1
rc=$?
rows=$(( $(wc -l < "$out/glm52_decode_bf16_tuned_gemm.csv" 2>/dev/null || echo 1) - 1 ))
echo "$(date -u +%T) tuned rows: $rows of $(( $(wc -l < "$out/untuned.csv") - 1 )) shapes in $(( $(date +%s) - t0 )) s (rc=$rc, batch $batch, hipblaslt $hipb)" | tee "$out/tune-summary.txt"
(( rows > 0 ))
