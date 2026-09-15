#!/bin/bash
# Start the container and the server, wait until it answers, leave it up.
#
#   ./glm52/launch_sglang.sh TP=8 DP=1 EP=1 CONC=8
#   ./glm52/launch_sglang.sh CONC=18 ENABLE_HICACHE=0
#   ./glm52/launch_sglang.sh TP=4 EP=4 HIP_VISIBLE_DEVICES=0,1,2,3 PORT=8888 NAME=glm52-a
#   ./glm52/launch_sglang.sh IMAGE=lmsysorg/sglang:v0.5.18-rocm720-mi35x
#   docker stop -t 300 glm52-server / docker logs -f glm52-server
#
# Mirrors SGLANG_CMD in third_party/InferenceX/benchmarks/single_node/agentic/
# glm5.2_fp4_mi355x_sglang_mtp.sh; re-diff when that submodule moves.
set -euo pipefail
for a in "$@"; do
    [[ "$a" == [A-Za-z_]*=* ]] || { echo "expected VAR=value, got '$a'" >&2; exit 2; }
    export "$a"
done

REPO=$(cd "$(dirname "$0")/.." && pwd)
IMAGE="${IMAGE:-rocm-llm-bench:latest}"
NAME="${NAME:-glm52-server}"
: "${MODEL_PATH:?set MODEL_PATH=/path/to/weights}"
PORT="${PORT:-8888}"
TP="${TP:-8}" EP="${EP:-1}" DP="${DP:-1}" CONC="${CONC:-512}"
ENABLE_HICACHE="${ENABLE_HICACHE:-1}"
for v in TP EP DP CONC; do
    [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "$v must be an integer, got '${!v}'" >&2; exit 2; }
done


# steps/draft/acc_len are calibrated as a set: 5/6/3.61 = MI355X arm,
# 3/4/2.99 = B200/B300 arm. ACC_LEN= turns simulation off (correctness runs).
SPEC_STEPS="${SPEC_STEPS:-5}"
SPEC_DRAFT="${SPEC_DRAFT:-6}"
ACC_LEN="${ACC_LEN-3.61}"     # no colon, so ACC_LEN= stays empty

# aiter JIT-compiles on first use and the products die with the container.
# Measured on a TP=8 start: 778 s across 16 modules, of which module_norm is
# 445 s compiled by one rank while the other seven sit on its lock -- 57% of a
# 1373 s startup. Local disk, not $REPO: this repo can sit on a network
# filesystem, and eight ranks compiling onto one is not the place.
#
# Keyed by image id, and it has to be: aiter reuses a cached .so whenever the
# file exists and carries code for the running arch (jit/core.py, via
# _needs_arch_rebuild) -- it never looks at what aiter built it from. A cache
# shared across images would hand a new AITER_SHA the previous one's kernels
# and say nothing. Files land root-owned; delete a stale one from a container.
IMAGE_KEY=$(docker image inspect -f '{{.Id}}' "$IMAGE" 2>/dev/null | cut -d: -f2 | cut -c1-12)
if [ -n "${AITER_JIT_CACHE:-}" ]; then
    :
elif [ -n "$IMAGE_KEY" ]; then
    AITER_JIT_CACHE="/tmp/aiter-jit-$(id -u)/$IMAGE_KEY"
else
    # Image not pulled yet, so there is no id to key on. Run without the cache
    # rather than key it on a mutable tag: slow is recoverable, stale is not.
    echo "no local image id for $IMAGE; running without the aiter JIT cache" >&2
fi
[ -n "${AITER_JIT_CACHE:-}" ] && mkdir -p "$AITER_JIT_CACHE"
mkdir -p "$REPO/.cache/hf" "$REPO/.cache/aiperf-$NAME"
# SIGTERM with room to finish. Interrupting the HiCache host pool's
# unregistration leaves the memory gone for the better part of a day, and with
# ENABLE_HICACHE that pool is hundreds of GB per rank, so `-t 60` was a timer
# on it -- a measured drain of a 2.5 TB pool took 52 s. `docker stop` returns
# when the container exits, so the ceiling only costs anything if shutdown
# hangs; 300 s is ~6x the measured drain and caps that at five minutes.
docker stop -t 300 "$NAME" >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true

# A node still draining HBM fails RCCL init. Only the GPUs this server will use,
# so a second server on the other half is not blocked by the first one's memory.
for i in $(seq 1 120); do
    vram=$( { rocm-smi --showmemuse 2>/dev/null || true; } |
            awk -v want="${HIP_VISIBLE_DEVICES:-}" -F'): ' '
                /VRAM%/ {
                    match($0, /GPU\[[0-9]+\]/)
                    id = substr($0, RSTART + 4, RLENGTH - 5)
                    if (want != "" && index("," want ",", "," id ",") == 0) next
                    n++; if ($NF + 0 > m) m = $NF + 0
                }
                END { print (n ? m + 0 : "") }') || vram=
    [ -n "$vram" ] || { echo "rocm-smi reported no VRAM%; refusing to skip the drain check" >&2; exit 1; }
    # 2, not 10. A card idles at 0 here, and 10% let 13 GB of a previous
    # server's undrained memory through -- TP=8 then hung at startup, twice,
    # with the same image and arguments that came up cleanly at 0.
    [ "$vram" -le 2 ] && break
    [ "$i" = 120 ] && { echo "GPUs still busy after 2h (vram%max=$vram)" >&2; exit 1; }
    echo "GPUs busy (vram%max=$vram), waiting"; sleep 60
done

PARALLEL=(--tp "$TP" --ep-size "$EP")
[ "$DP" -gt 1 ] && PARALLEL+=(--dp "$DP" --enable-dp-attention)

HICACHE_CFG=()
[ "$ENABLE_HICACHE" = 1 ] && HICACHE_CFG=(
    --enable-hierarchical-cache
    --hicache-ratio "${HICACHE_RATIO:-1.5}"
    --hicache-write-policy write_through
    --hicache-io-backend "${HICACHE_IO_BACKEND:-kernel}"
    --hicache-mem-layout "${HICACHE_MEM_LAYOUT:-page_first}")

# -w / : the image's WORKDIR is /sgl-workspace, where cwd shadows sglang.
docker run -d --name "$NAME" -w / \
    --network host --ipc=host --shm-size 32g \
    --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
    --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
    -v "$REPO:$REPO" -v "$MODEL_PATH:$MODEL_PATH:ro" \
    -v "$REPO/.cache/hf:/hf_cache" -v "$REPO/.cache/aiperf-$NAME:/tmp/inferencex-agentic" \
    -e HF_HOME=/hf_cache -e HF_TOKEN="${HF_TOKEN:-}" \
    -e SGLANG_OPT_USE_TOPK_V2=false \
    -e SGLANG_TIMEOUT_KEEP_ALIVE=900 \
    -e PYTHONNOUSERSITE=1 \
    ${AITER_JIT_CACHE:+-v "$AITER_JIT_CACHE:/jit-cache"} \
    ${AITER_JIT_CACHE:+-e AITER_JIT_DIR=/jit-cache} \
    -e AITER_USE_FLYDSL_MOE_SORTING=1 \
    -e SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1 \
    ${HIP_VISIBLE_DEVICES:+-e HIP_VISIBLE_DEVICES} \
    ${ACC_LEN:+-e SGLANG_SIMULATE_ACC_LEN="$ACC_LEN"} \
    ${ACC_LEN:+-e SGLANG_SIMULATE_ACC_METHOD=match-expected} \
    ${ACC_LEN:+-e SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token} \
    "$IMAGE" \
    python3 -m sglang.launch_server \
      --model-path "$MODEL_PATH" --served-model-name "$MODEL_PATH" \
      --host 0.0.0.0 --port "$PORT" --trust-remote-code \
      "${PARALLEL[@]}" \
      --kv-cache-dtype fp8_e4m3 \
      --dsa-prefill-backend flydsl \
      --dsa-decode-backend flydsl \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --chunked-prefill-size 32768 \
      --mem-fraction-static 0.85 \
      --max-running-requests "$CONC" \
      --cuda-graph-max-bs "$CONC" \
      --speculative-algorithm EAGLE \
      --speculative-num-steps "$SPEC_STEPS" \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens "$SPEC_DRAFT" \
      --dsa-topk-backend aiter \
      --enable-aiter-allreduce-fusion \
      --enable-fused-qk-norm-rope \
      "${HICACHE_CFG[@]}" \
      --watchdog-timeout 1800 \
      --enable-metrics >/dev/null

# Follow the container's output to a file for as long as it lives. A server
# that dies takes its logs with it -- the caller removes the container on exit,
# and `docker logs` then has nothing to show, which is how two crashes were
# lost. The follower ends by itself when the container does.
SERVER_LOG="${SERVER_LOG:-$REPO/.cache/server-$NAME.log}"
mkdir -p "$(dirname "$SERVER_LOG")"
docker logs -f "$NAME" > "$SERVER_LOG" 2>&1 &

echo "$NAME: TP=$TP EP=$EP DP=$DP CONC=$CONC hicache=$ENABLE_HICACHE steps=$SPEC_STEPS draft=$SPEC_DRAFT acc_len=${ACC_LEN:-off}"
echo "  log -> $SERVER_LOG"
for i in $(seq 1 180); do
    curl -fsS -m 30 "http://localhost:$PORT/server_info" >/dev/null 2>&1 && {
        echo "ready after $((i*10))s"
        # A backend that declines does so silently: the server comes up, serves
        # normally, and produces a plausible number with the kernels never
        # entered. Only the two below are printed by the time the server is
        # ready -- the indexer at load, the decode path during graph capture.
        # "FlyDSL sparse MLA prefill engaged" waits for the first real prefill,
        # which is minutes past this point, so it cannot be checked here.
        # Read the log once into a variable rather than piping per pattern:
        # under `set -o pipefail`, grep -q exits at the first match, docker logs
        # takes SIGPIPE, and the pipeline reports 141 -- so every check read as
        # a miss even with the line right there.
        log=$(docker logs "$NAME" 2>&1 || true)
        miss=()
        for want in "FlyDSL sparse MLA decode engaged" \
                    "gfx950 fused DSA indexer enabled"; do
            case $log in *"$want"*) ;; *) miss+=("$want") ;; esac
        done
        if [ ${#miss[@]} -gt 0 ]; then
            echo "!! did not engage:" >&2
            printf '   %s\n' "${miss[@]}" >&2
            echo "   docker logs $NAME | grep -iE 'declin|disabl|fall'" >&2
        fi
        exit 0
    }
    docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true || {
        echo "server exited -- docker logs $NAME" >&2; exit 1; }
    sleep 10
done
echo "not ready after 30min -- docker logs $NAME" >&2; exit 1
