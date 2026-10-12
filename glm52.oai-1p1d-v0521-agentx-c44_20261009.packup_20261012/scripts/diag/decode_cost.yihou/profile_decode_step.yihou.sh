#!/usr/bin/env bash
# DRAFT (repro, not run): one-shot decode-step profile at controlled per-rank bs, OUTSIDE a scored window.
# Goal: split the ~23-26 ms fixed decode-step cost and the bs 5->6 jump (R041/R042) into kernels.
# Mechanism: SGLang native /start_profile on the decode engine port (torch profiler, kineto/roctracer;
#   ProfileReq fields io_struct.py:2257-2283; route http_server.py:1164). Trace files land in output_dir
#   (WS is bind-mounted at the same path in the containers).
# Load: N identical-length long prompts with ignore_eos + large max_tokens through the router, so all N
#   are in decode together; per-rank bs ~= N/4 (check the per-DP "#running-req" log lines before profiling).
# usage: profile_decode_step.yihou.sh <node> <job> <per_rank_bs> <ctx_tokens> [tag]
set -euo pipefail
node=${1:?node}; job=${2:?job}; BS=${3:?per-rank bs}; CTX=${4:?context tokens}; TAG=${5:-bs$3-ctx$4}
HOST=10.245.147.130; ROUTER=http://$HOST:28100; DEC=http://$HOST:29357; MODEL=glm5.2-mxfp4
WS=/shared_nfs/yihou_glm52/v0521dbg_20261010
OUT=$WS/diag/decode_cost.yihou/prof.$TAG.$(date -u +%Y%m%d-%H%M%S).yihou; mkdir -p "$OUT"
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS="$node:$job"
N=$(( BS * 4 ))
# 1) load: N streams, distinct prompt prefixes (no prefix-cache sharing on decode; DSA ctx = CTX), long decode
python3 - "$ROUTER" "$MODEL" "$N" "$CTX" "$OUT" <<'EOF' &
import json, sys, threading, urllib.request
url, model, n, ctx, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
def one(i):
    # ~1 token per " w<k>" word is NOT guaranteed; check prompt_tokens in usage and adjust the factor.
    words = " ".join(f"w{(i * 7919 + k) % 50000}" for k in range(int(ctx / 2.2)))
    body = {"model": model, "prompt": f"[{i}] " + words, "max_tokens": 6000, "ignore_eos": True,
            "temperature": 0.0}
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    open(f"{out}/usage.{i}.json", "w").write(json.dumps(r.get("usage", {})))
ts = [threading.Thread(target=one, args=(i,)) for i in range(n)]
[t.start() for t in ts]; [t.join() for t in ts]
EOF
LOADPID=$!
# 2) wait until every decode DP reports #running-req == BS (decode log, via docker logs on the node)
for i in $(seq 1 120); do
  s=$(ssh "$node" "docker logs --since 20s glm52-v0521-yihou-102-e2g-decode-0 2>&1 | strings | grep -a 'Decode batch' | tail -8" || true)
  echo "$s" | grep -oE 'DP[0-9] TP[0-9]\] Decode batch, #running-req: [0-9]+' | sort -u | tail -4
  ok=$(echo "$s" | grep -oE 'DP([0-9]) TP[0-9]\] Decode batch, #running-req: '"$BS"',' | grep -oE '^DP[0-9]' | sort -u | wc -l)
  [[ "$ok" == 4 ]] && break; sleep 15
done
# 3) profile 8 forward steps on all decode ranks (GPU kernels + CPU ops; no stacks/shapes to keep overhead low)
curl -s -X POST "$DEC/start_profile" -H 'Content-Type: application/json' -d "{
  \"output_dir\": \"$OUT\", \"num_steps\": 8, \"activities\": [\"CPU\", \"GPU\"],
  \"with_stack\": false, \"record_shapes\": false, \"profile_prefix\": \"$TAG\"}" | tee "$OUT/start_profile.json"
sleep 60   # num_steps auto-stops; trace export of 4 ranks takes a while
ls -la "$OUT"
# 4) release the load (abort is cleaner than waiting 6000 tokens)
kill $LOADPID 2>/dev/null || true
echo "[prof] $OUT  -> analyze: per-kernel sum over one verify+draft cycle; diff bs5 vs bs6 traces"
# Matrix to run (each ~5-10 min incl. prefill+transfer of N*CTX tokens):
#   bs 4/5/6 at ctx 128k  -> isolates the 5->6 jump at fixed ctx
#   bs 5 at ctx 32k / 256k -> context slope (DSA indexer, ∝ 6 rows x ctx per request)
# Fallback if kineto shows only hipGraphLaunch without inner kernels: relaunch the decode leg (debug stack,
#   not scored) under `rocprofv3 --kernel-trace --stats --output-format csv -d $OUT --` with the same args
#   and drive the same load; ROCm 7.2 rocprofv3 traces kernels inside graph replays.
