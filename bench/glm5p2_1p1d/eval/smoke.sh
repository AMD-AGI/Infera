#!/usr/bin/env bash
# Verify worker pairing, normal chat, agentic tool calls, RDMA, DPA and MTP.
# Usage: bash eval/smoke.sh  # requires a running stack
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DIR/config.sh"
acquire_bench_lock

URL="http://$PREFILL_IP:$ROUTER_PORT"
OUT="${SMOKE_OUT_DIR:-$RUN_ROOT/correctness/smoke}"
rm -rf "$OUT"
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1

echo "===== 1. router and PD workers ====="
curl -fsS "$URL/health" | tee "$OUT/health.json"
echo
curl -fsS "$URL/v1/workers" | tee "$OUT/workers.json"
echo
python3 - "$OUT/workers.json" <<'PY'
import json
import os
import sys

workers = json.load(open(sys.argv[1]))["workers"]
roles = [worker.get("disagg_mode") for worker in workers]
assert len(workers) == 2, f"expected 2 workers, got {len(workers)}: {workers}"
assert roles.count("prefill") == 1 and roles.count("decode") == 1, roles
print("workers: PASS (one prefill + one decode)")
kv_aware = os.environ["ENABLE_KV_AWARE"] == "1"
for worker in workers:
    endpoint = worker.get("kv_events_endpoint")
    if kv_aware and worker["disagg_mode"] == "prefill":
        assert endpoint, f"kv-aware prefill has no event endpoint: {worker}"
        assert (worker.get("kv_block_size") or 0) > 0, worker
    else:
        assert not endpoint, (
            f"unexpected KV-event endpoint with kv-aware={kv_aware}: {worker}"
        )
    print(
        f"  {worker['disagg_mode']}: {worker['url']} "
        f"dp={worker.get('dp_size')} block={worker.get('kv_block_size')} "
        f"events={endpoint or 'off'}"
    )
PY

echo "===== 2. deterministic GLM-5.2 factual completion ====="
jupiter_body="$(
    SERVED_MODEL="$SERVED_MODEL" python3 - <<'PY'
import json
import os

print(json.dumps({
    "model": os.environ["SERVED_MODEL"],
    "messages": [{
        "role": "user",
        "content": "Complete this fact with only the planet name: The largest planet in the Solar System is",
    }],
    # glm45 reasoning parser bills the chain of thought against the same budget,
    # so a small cap returns empty content with finish_reason=length.
    "max_tokens": 512,
    "temperature": 0,
}))
PY
)"
curl -fsS --max-time 180 "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$jupiter_body" \
    | tee "$OUT/jupiter.json"
echo
python3 - "$OUT/jupiter.json" <<'PY'
import json
import sys

response = json.load(open(sys.argv[1]))
message = response["choices"][0]["message"]
text = (message.get("content") or "") + (message.get("reasoning_content") or "")
assert "jupiter" in text.lower(), message
assert "�" not in text, message
print("Jupiter: PASS")
PY

echo "===== 3. OpenAI tool-call parser ====="
tool_body="$(
    SERVED_MODEL="$SERVED_MODEL" python3 - <<'PY'
import json
import os

print(json.dumps({
    "model": os.environ["SERVED_MODEL"],
    "messages": [{
        "role": "user",
        "content": "Use the get_weather tool to check the weather in Paris.",
    }],
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }],
    "tool_choice": "required",
    "max_tokens": 512,
}))
PY
)"
curl -fsS --max-time 180 "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$tool_body" \
    | tee "$OUT/tool_call.json"
echo
python3 - "$OUT/tool_call.json" <<'PY'
import json
import sys

response = json.load(open(sys.argv[1]))
message = response["choices"][0]["message"]
calls = message.get("tool_calls") or []
assert calls, message
assert calls[0]["function"]["name"] == "get_weather", calls
arguments = json.loads(calls[0]["function"]["arguments"])
assert arguments.get("city", "").lower() == "paris", arguments
print("tool call: PASS", calls[0]["function"])
PY

if [[ "$ENABLE_MTP" == "1" ]]; then
    echo "===== 4. short concurrent burst for MTP metrics ====="
    URL="$URL" SERVED_MODEL="$SERVED_MODEL" python3 - <<'PY'
import concurrent.futures
import json
import os
import urllib.request

url = f"{os.environ['URL']}/v1/chat/completions"
payload = json.dumps({
    "model": os.environ["SERVED_MODEL"],
    "messages": [{"role": "user", "content": "Name one prime number larger than 100."}],
    "max_tokens": 96,
}).encode()

def one(_):
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
    assert result.get("choices"), result

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    list(pool.map(one, range(24)))
print("MTP burst: PASS (24 requests, concurrency 8)")
PY
fi

echo "===== 5. engine evidence ====="
ssh $SSH_OPTS "$PREFILL_NODE" docker logs glm52-pd-prefill \
    >"$OUT/prefill.log" 2>&1
ssh $SSH_OPTS "$DECODE_NODE" docker logs glm52-pd-decode \
    >"$OUT/decode.log" 2>&1
ssh $SSH_OPTS "$PREFILL_NODE" rocm-smi --showuse --showmemuse \
    >"$OUT/gpu_prefill.log" 2>&1
ssh $SSH_OPTS "$DECODE_NODE" rocm-smi --showuse --showmemuse \
    >"$OUT/gpu_decode.log" 2>&1
ssh $SSH_OPTS "$PREFILL_NODE" docker inspect glm52-pd-prefill \
    >"$OUT/prefill_container.json" 2>&1
ssh $SSH_OPTS "$DECODE_NODE" docker inspect glm52-pd-decode \
    >"$OUT/decode_container.json" 2>&1
ssh $SSH_OPTS "$PREFILL_NODE" docker exec glm52-pd-prefill ibv_devices \
    >"$OUT/rdma_prefill.log" 2>&1
ssh $SSH_OPTS "$DECODE_NODE" docker exec glm52-pd-decode ibv_devices \
    >"$OUT/rdma_decode.log" 2>&1
IFS=',' read -r -a rdma_devices <<<"$RDMA_DEVICE"
for log in "$OUT/rdma_prefill.log" "$OUT/rdma_decode.log"; do
    for device in "${rdma_devices[@]}"; do
        grep -qw "$device" "$log" || {
            echo "RDMA device $device is not usable; see $log" >&2
            exit 1
        }
    done
done
python3 - "$OUT/prefill_container.json" "$OUT/decode_container.json" <<'PY'
import json
import os
import sys

expected_env = {
    "MC_TE_FILTERS": os.environ["MC_TE_FILTERS"],
    "MC_GID_INDEX": os.environ["MC_GID_INDEX"],
    "MOONCAKE_DISABLE_HIP_DMABUF": os.environ["MOONCAKE_DISABLE_HIP_DMABUF"],
    "MC_ENABLE_DEST_DEVICE_AFFINITY": os.environ["MC_ENABLE_DEST_DEVICE_AFFINITY"],
}
for path in sys.argv[1:]:
    config = json.load(open(path))[0]["Config"]
    env = dict(item.split("=", 1) for item in config["Env"] if "=" in item)
    assert {key: env.get(key) for key in expected_env} == expected_env, path
    cmd = config["Cmd"]
    index = cmd.index("--disaggregation-ib-device")
    assert cmd[index + 1] == os.environ["RDMA_DEVICE"], path
print("RDMA launch contract: PASS")
PY
ENABLE_MTP="$ENABLE_MTP" python3 - "$OUT/prefill.log" "$OUT/decode.log" <<'PY'
import os
import re
import statistics
import sys

fatal = (
    "MC_FORCE_TCP",
    "GID is NULL",
    "KVTransferError",
    "Expected lengths.size",
    "Traceback (most recent call last)",
    "Memory access fault",
)
failed = False
for path in sys.argv[1:]:
    text = open(path, errors="replace").read()
    print(f"{os.path.basename(path)}:")
    for pattern in fatal:
        count = text.count(pattern)
        print(f"  {pattern}: {count}")
        failed |= count > 0
    dp = re.search(r"dp_size=(\d+)", text)
    dpa = re.search(r"enable_dp_attention=(True|False)", text)
    print("  resolved:", dp.group(0) if dp else "dp_size=?", dpa.group(0) if dpa else "dpa=?")

decode = open(sys.argv[2], errors="replace").read()
accept = [float(x) for x in re.findall(r"accept len: ([0-9.]+)", decode)]
if os.environ.get("ENABLE_MTP") == "1":
    assert accept, "MTP enabled but no accept-len evidence found"
    print(
        f"  MTP accept length: n={len(accept)} "
        f"median={statistics.median(accept):.2f} min={min(accept):.2f} max={max(accept):.2f}"
    )
if failed:
    raise SystemExit("fatal pattern found in engine logs")
print("engine evidence: PASS")
PY

echo "[smoke] PASS; artifacts: $OUT"
