#!/usr/bin/env bash
# Prepare a fresh crusoe node: stage the weights to the NVMe, pull the pinned base images, get the engine image
# (local -> saved tar -> build from the kit checkout), convert GSM8K for the accuracy probe, run the kit's RDMA
# preflight. The last line is "prep ok on <host> image=<tag>" or "prep FAILED on <host>: <why>".
set -uo pipefail
source "$(dirname "$0")/env.sh"
h=$(hostname -s)
fail() { echo "prep FAILED on $h: $*"; exit 0; }
mkdir -p "$NODE_ROOT" "$MODEL_MOUNT"
free_gb=$(df -BG --output=avail "$NODE_ROOT" | tail -1 | tr -dc 0-9)
echo "$(ts) NVMe free ${free_gb} GB"
[[ -f "$MODEL/.complete" ]] || (( free_gb >= 560 )) || fail "NVMe has only ${free_gb} GB free for a 476 GB model"
bash "$W/scripts/stage_model.sh" > "$W/logs/stage-$h.log" 2>&1 &
st=$!

for img in "$BASE_IMAGE" "$RUST_IMAGE" "$ETCD_IMAGE"; do
    pull_image "$img" > /dev/null 2>&1 || fail "pull of $img failed"
done
echo "$(ts) base images present"
if ! docker image inspect "$IMAGE" > /dev/null 2>&1 && [[ -s "$IMAGE_TAR" ]]; then
    timeout 1800 docker load -i "$IMAGE_TAR" > /dev/null 2>&1 && echo "$(ts) loaded $IMAGE from $IMAGE_TAR"
fi
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "$(ts) building $IMAGE"
    # The GPU self-check runs on one of our GPUs, not on a co-tenant's.
    HIP_VISIBLE_DEVICES=${PREFILL_GPUS:-0} bash "$KIT/build_image.sh" "$IMAGE" > "$W/logs/build-$h.log" 2>&1 \
        || fail "image build failed: $(tail -3 "$W/logs/build-$h.log" | tr '\n' ' ')"
    echo "$(ts) built $IMAGE: $(tail -2 "$W/logs/build-$h.log" | head -1)"
fi
# Save the image right away, so the next node can load it instead of rebuilding (~20 min).
if [[ ! -s "$IMAGE_TAR" ]]; then
    mkdir -p "$(dirname "$IMAGE_TAR")"
    timeout 1800 docker save -o "$IMAGE_TAR.part" "$IMAGE" && mv "$IMAGE_TAR.part" "$IMAGE_TAR" \
        && echo "$(ts) saved $IMAGE_TAR ($(du -h "$IMAGE_TAR" | cut -f1))"
    rm -f "$IMAGE_TAR.part"
fi
docker run --rm --entrypoint cat "$IMAGE" /opt/infera/BUILD_INFO | sed 's/^/  /'

mkdir -p "$W/data" && chmod 777 "$W/data"
if [[ ! -s "$W/data/gsm8k_test.jsonl" ]]; then
    # The HF cache snapshot holds a symlink into blobs/: mount the file it points to.
    docker run --rm --network none -v "$(realpath "$GSM8K_PARQUET"):/in/test.parquet:ro" -v "$W/data:/out" --entrypoint python3 "$IMAGE" -c '
import json, pyarrow.parquet as pq
rows = pq.read_table("/in/test.parquet").to_pylist()
with open("/out/gsm8k_test.jsonl", "w") as f:
    for r in rows: f.write(json.dumps({"question": r["question"], "answer": r["answer"]}) + "\n")
print(len(rows), "rows")' || fail "GSM8K conversion failed"
fi
echo "$(ts) GSM8K: $(wc -l < "$W/data/gsm8k_test.jsonl") rows"

IMAGE="$IMAGE" bash "$KIT/preflight_rdma.sh" 2>&1 | sed 's/^/  /'

wait $st; st_rc=$?
tail -1 "$W/logs/stage-$h.log"
(( st_rc == 0 )) || fail "model staging failed"
echo "prep ok on $h image=$IMAGE"
