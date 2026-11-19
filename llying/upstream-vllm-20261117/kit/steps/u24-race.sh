#!/usr/bin/env bash
# U24 mechanism check (no GPU): a uvicorn app with keep-alive K s (vLLM's API server is uvicorn, default K = 5) and a
# reqwest client built like the Infera router's upstream client, requests spaced around K s. Old router: pool idle
# 90 s (reqwest's default); new: below the keep-alive (0.8 s for K = 1, the PR's default 4 s for K = 5).
# Expect "error sending request" with the old setting and none with the new one.
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/checks/u24_race/src/main.rs /home/liyingli/bench_agentx/upstream-20261119/steps/u24-race.sh
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready rust py312 || exit 1
mkdir -p "$NL/cargo" "$NL/u24-target"
docker rm -f up-u24-server up-u24-client > /dev/null 2>&1
cat > "$OUT/app.py" <<'EOF'
async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    while (await receive()).get("more_body"):
        pass
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"application/json")]})
    await send({"type": "http.response.body", "body": b'{"ok":true}'})
EOF
build=$(docker run --rm --label up.job=$JOB --name up-u24-client --network host -v "$U/checks/u24_race:/src:ro" -v "$NL/cargo:/cargo-home" \
    -v "$NL/u24-target:/target" -e CARGO_HOME=/cargo-home -e CARGO_TARGET_DIR=/target --entrypoint bash up-rust:1 \
    -c 'cp -r /src /tmp/w && cd /tmp/w && cargo build --release -q 2>&1 | tail -3; ls /target/release/u24-race')
echo "$build"
for case in "1 90000 1000" "1 800 1000" "5 90000 5000" "5 4000 5000"; do
    read -r ka idle gap <<< "$case"
    docker rm -f up-u24-server > /dev/null 2>&1
    docker run -d --rm --label up.job=$JOB --name up-u24-server --network host -v "$OUT:/app:ro" -v "$NL/pip:/root/.cache/pip" \
        --entrypoint sh python:3.12-slim-bookworm -c \
        "pip install -q uvicorn > /dev/null 2>&1; cd /app && exec uvicorn app:app --host 127.0.0.1 --port 28791 --timeout-keep-alive $ka --log-level warning" > /dev/null
    for i in $(seq 60); do curl -sf -m 2 -X POST http://127.0.0.1:28791/ > /dev/null && break; sleep 2; done
    dur=$(( ka == 1 ? 120 : 300 ))
    docker run --rm --label up.job=$JOB --name up-u24-client --network host -v "$NL/u24-target:/target:ro" --entrypoint /target/release/u24-race \
        up-rust:1 http://127.0.0.1:28791/ "$idle" "$gap" "$dur" 64 | sed "s/^{/{\"keep_alive_s\":$ka,/" | tee -a "$OUT/race.jsonl"
done
docker rm -f up-u24-server > /dev/null 2>&1
python3 - "$OUT/race.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
old = [r for r in rows if r["pool_idle_ms"] == 90000]
new = [r for r in rows if r["pool_idle_ms"] < r["keep_alive_s"] * 1000]
print("old (pool idle 90 s):", [(r["keep_alive_s"], r["requests"], r["error_sending_request"]) for r in old])
print("new (idle below keep-alive):", [(r["keep_alive_s"], r["requests"], r["error_sending_request"]) for r in new])
sys.exit(0 if new and all(r["errors"] == 0 for r in new) else 1)
EOF
