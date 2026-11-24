#!/usr/bin/env bash
# kv-aware check against a running 1P1D deployment (run on the node, decode on block acceptance):
#   1. the engine's prompt for a probe body, via the prefill's /tokenize (logged for reference). The router's
#      render probe (infera_router_render_parity) only asks sglang workers, so there is no parity gauge here;
#   2. prefill picks with cache_hits > 0: 8 two-turn conversations whose second turn is sent under a NEW session id,
#      so session affinity cannot place it and only kv-aware can find the rank holding its prefix. A hash that
#      disagrees with the engine's anywhere in the prompt scores 0 here, so this is the parity check that counts.
#   kvaware_check.sh LOG_DIR ROUTER_URL PREFILL_URL MODEL
set -uo pipefail
LOG_DIR=${1:?}; URL=${2:?}; PURL=${3:?}; SERVED=${4:?}
out=$LOG_DIR/kvaware; mkdir -p "$out"
python3 - "$PURL" "$SERVED" > "$out/tokenize.txt" 2>&1 <<'PY'
import json, sys, urllib.request
url, model = sys.argv[1], sys.argv[2]
body = {"model": model, "messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "2+2?"}],
        "add_special_tokens": False}
req = urllib.request.Request(url + "/tokenize", json.dumps(body).encode(), {"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=60)))
PY
echo "engine /tokenize: $(head -c 300 "$out/tokenize.txt")"
before=$(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/router.log" | grep -a 'role=Prefill' | grep -ac 'cache_hits=[1-9]')
total0=$(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/router.log" | grep -ac 'role=Prefill')
python3 - "$URL" "$SERVED" > "$out/twoturn.txt" 2>&1 <<'PY'
import json, random, sys, urllib.request
url, model = sys.argv[1], sys.argv[2]
def chat(msgs, sid):
    body = {"model": model, "messages": msgs, "max_tokens": 32, "temperature": 0}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "X-Session-ID": sid})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return d["choices"][0]["message"].get("content") or "", d["usage"]["prompt_tokens"]
for i in range(8):
    r = random.Random(i)
    words = "river mountain window garden lantern bridge tower valley ocean island".split()
    sys_prompt = " ".join(r.choice(words) for _ in range(3000))
    t1 = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"Question {i}: say ok."}]
    a1, p1 = chat(t1, f"kv-check-{i}-a")
    t2 = t1 + [{"role": "assistant", "content": a1}, {"role": "user", "content": "And again."}]
    a2, p2 = chat(t2, f"kv-check-{i}-b")
    print(i, p1, p2, repr(a2[:20]))
PY
sleep 3
after=$(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/router.log" | grep -a 'role=Prefill' | grep -ac 'cache_hits=[1-9]')
total1=$(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/router.log" | grep -ac 'role=Prefill')
echo "two-turn probe: prefill picks $((total1 - total0)), with cache_hits>0: $((after - before)) (want >= 8)"
sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/router.log" | grep -a 'role=Prefill' | grep -a 'cache_hits=[1-9]' | tail -2 | cut -c1-240
[ $((after - before)) -ge 8 ]
