#!/usr/bin/env bash
# Background prep for a plan job, only what PREP (comma list) asks for; each item writes $NL/ready/<item> when done
# and steps wait for it with need_ready (jobs/lib.sh):
#   v034 / nightly   pull the stock vLLM ROCm image (tests mount the branch's files over it; nothing is built)
#   rust             rust:1-bookworm + cmake / clang (image up-rust:1) for the Infera router CI checks
#   model            GLM-5.2-MXFP4 on the node's NVMe (38-109 s from /shared_nfs on earlier nodes)
#   p4d4             p4d4/scripts/node_prep.sh: model + the Infera experiment image infera-vllm:v0340-0c5e5e89
#                    (built from the branch at 0c5e5e89; ~5 min on a fresh node, 11-19 v3-207)
#   aiterjit         restore the aiter JIT cache volume for the v0.34.0 image from $U/cache (saved by save_caches)
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
mkdir -p "$NL/ready"
rm -f "$NL/ready/"*
pull() {
    local tag=$1 img=$2 t0
    t0=$(date +%s)
    if docker image inspect "$img" > /dev/null 2>&1 || docker pull -q "$img" > /dev/null; then
        echo "$(date -u +%T) $tag ready ($(( $(date +%s) - t0 )) s)"; echo "$img" > "$NL/ready/$tag"
    else
        echo "$(date -u +%T) $tag pull FAILED"
    fi
}
has() { local p="+${PREP//,/+}+"; [[ "$p" == *"+$1+"* ]]; }
has v034 && pull v034 "$IMG_V034"
has nightly && pull nightly "$IMG_NIGHTLY" &
has py310 && pull py310 python:3.10-slim-bookworm
if has rust; then
    pull rust-base rust:1-bookworm
    if ! docker image inspect up-rust:1 > /dev/null 2>&1; then
        printf 'FROM rust:1-bookworm\nRUN apt-get update && apt-get install -y --no-install-recommends cmake clang libclang-dev python3 python3-venv && rm -rf /var/lib/apt/lists/*\n' \
            | docker build -q -t up-rust:1 - > /dev/null && echo "$(date -u +%T) up-rust:1 built"
    fi
    docker image inspect up-rust:1 > /dev/null 2>&1 && echo up-rust:1 > "$NL/ready/rust"
fi
if has p4d4; then
    R=$U/p4d4 bash "$U/p4d4/scripts/node_prep.sh" > "$U/logs/p4d4-prep-$JOB.log" 2>&1
    tail -1 "$U/logs/p4d4-prep-$JOB.log"
    grep -q "prep ok" "$U/logs/p4d4-prep-$JOB.log" && touch "$NL/ready/p4d4" "$NL/ready/model"
elif has model; then
    if model_dir > /dev/null || bash "$U/p4d4/scripts/stage_model.sh" /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 "$MODEL_DIR"; then
        touch "$NL/ready/model"; echo "$(date -u +%T) model ready at $(model_dir)"
    fi
fi
has aiterjit && restore_caches
wait
echo "$(date -u +%T) prep done: $(ls "$NL/ready" | tr '\n' ' ')"
