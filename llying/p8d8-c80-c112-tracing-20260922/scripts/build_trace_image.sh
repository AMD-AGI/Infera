#!/usr/bin/env bash
# Build on 138, transfer to 136, and pin the derived image digest.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/config.trace.p8d8.sh"

BASE_IMAGE="infera-sglang:v0519-yihou-0917-nextnfix-hicache"
BASE_ID="sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35"

for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    actual="$(
        ssh -o BatchMode=yes "$node" \
            docker image inspect "$BASE_IMAGE" --format '{{.Id}}'
    )"
    [[ "$actual" == "$BASE_ID" ]] || {
        echo "$node base image mismatch: $actual expected=$BASE_ID" >&2
        exit 1
    }
done

ssh -o BatchMode=yes "$PREFILL_NODE" \
    docker build --network host \
    -f "$ROOT/docker/Dockerfile" -t "$IMAGE" "$ROOT"

ssh -o BatchMode=yes "$PREFILL_NODE" \
    "docker save '$IMAGE'" |
    ssh -o BatchMode=yes "$DECODE_NODE" docker load

prefill_id="$(
    ssh -o BatchMode=yes "$PREFILL_NODE" \
        docker image inspect "$IMAGE" --format '{{.Id}}'
)"
decode_id="$(
    ssh -o BatchMode=yes "$DECODE_NODE" \
        docker image inspect "$IMAGE" --format '{{.Id}}'
)"
[[ "$prefill_id" == "$decode_id" ]] || {
    echo "derived image differs: prefill=$prefill_id decode=$decode_id" >&2
    exit 1
}
printf '%s\n' "$prefill_id" >"$ROOT/image-id.txt"

for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    count="$(
        ssh -o BatchMode=yes "$node" \
            docker run --rm --entrypoint bash "$IMAGE" -lc \
            "python3 - <<'PY'
from pathlib import Path
p=Path('/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py')
print(p.read_text().count('INFERA_C80_C112_REQUEST_TRACE_V1'))
PY"
    )"
    [[ "$count" == 2 ]] || {
        echo "$node trace marker count=$count, expected=2" >&2
        exit 1
    }
done
echo "trace image ready: $IMAGE -> $prefill_id"
