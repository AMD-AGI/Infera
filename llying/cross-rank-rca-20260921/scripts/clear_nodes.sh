#!/usr/bin/env bash
# Gracefully stop user inference/benchmark workloads while preserving platform
# monitoring containers and all stopped containers/images for forensics.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/config/config.rca.p8d8.sh"
OUT="${1:?usage: $0 OUT_DIR}"
mkdir -p "$OUT"

for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    node_out="$OUT/$node"
    mkdir -p "$node_out"
    ssh -o BatchMode=yes "$node" docker ps -a --no-trunc \
        >"$node_out/docker-before.txt"
    ssh -o BatchMode=yes "$node" python3 - <<'PY' >"$node_out/selected.txt"
import json
import subprocess

ids = subprocess.run(
    ["docker", "ps", "-aq"], check=True, capture_output=True, text=True
).stdout.split()
if not ids:
    raise SystemExit
containers = json.loads(
    subprocess.run(
        ["docker", "inspect", *ids], check=True, capture_output=True, text=True
    ).stdout
)
system_prefixes = (
    "crusoe-vector",
    "crusoe-amd-",
)
for container in containers:
    name = container.get("Name", "").lstrip("/")
    image = (container.get("Config") or {}).get("Image", "")
    devices = (container.get("HostConfig") or {}).get("Devices") or []
    has_kfd = any(item.get("PathOnHost") == "/dev/kfd" for item in devices)
    inference = (
        name.startswith("glm52-pd-")
        or "sglang" in image.lower()
        or has_kfd
    )
    if inference and not name.startswith(system_prefixes):
        print(name)
PY

    mapfile -t selected <"$node_out/selected.txt"
    if (( ${#selected[@]} )); then
        printf 'stopping on %s: %s\n' "$node" "${selected[*]}"
        ssh -o BatchMode=yes "$node" \
            docker stop --time 300 "${selected[@]}" \
            >"$node_out/docker-stop.txt" 2>&1
    else
        echo "no user inference containers selected on $node" \
            >"$node_out/docker-stop.txt"
    fi
    ssh -o BatchMode=yes "$node" docker ps -a --no-trunc \
        >"$node_out/docker-after-stop.txt"
done

TIMEOUT_S=3600 INTERVAL_S=30 THRESHOLD_PCT=1 \
    bash "$PACKUP_ROOT/scripts/wait_gpus_free.yihou.sh" \
    "$PREFILL_NODE" "$DECODE_NODE" | tee "$OUT/wait-gpus-free.log"

date -u --iso-8601=ns >"$OUT/completed-at.txt"
