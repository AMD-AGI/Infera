#!/usr/bin/env bash
# Runs INSIDE the container. The gate for P1 (the Rust bigram decode).
#
# WHY AN A/B AND NOT A SINGLE RUN
# -------------------------------
# A patched router reading nonzero `cache_hits` proves nothing on its own: the
# same number appears if the engine happened to send plain ints. The claim is
# specifically "the Rust decoder used to drop MTP's bigram pairs", so the
# unpatched binary must be shown reading ZERO against the very same traffic.
#
# The observable is policy.rs's per-pick `cache_hits` tracing field, which is
# the number the routing decision actually used -- not a view count that could
# be populated by some other path.
#
# Both legs share the live MTP prefill engine and one prompt sent twice: the
# first request populates the view, the second is the one whose pick can hit.
set -u
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
ETCD=10.2.122.10:2379
PORT=8199   # not 8100 -- leave the python router of the previous gate alone
LOG=/tmp/rustab

mkdir -p "$LOG"

# A ~1.2k-token prompt: long enough to span several 16-token pages, so a repeat
# has real prefix overlap to find.
PROMPT=$(python3 - <<'PY'
print("You are a helpful assistant. Background follows; ignore unless asked.\n" +
      "\n".join(f"Section {i}: component {i} handles subsystem {i%7}, mode {i%3}, "
                f"controller {i%5}, throughput {1000+i*7} units/s." for i in range(120)))
PY
)

drive() { # $1 = tag ; sends the SAME prompt twice through the router on $PORT
  for n in 1 2; do
    python3 - "$PORT" "$n" <<PY > /dev/null 2>&1
import json,sys,urllib.request
port,n=sys.argv[1],sys.argv[2]
body=json.dumps({"model":"glm5.2-mxfp4","messages":[
  {"role":"system","content":'''$PROMPT'''},
  {"role":"user","content":"Reply with the single word: ping"}],
  "max_tokens":16,"temperature":1.0,"top_p":0.95}).encode()
r=urllib.request.Request(f"http://10.2.122.10:{port}/v1/chat/completions",
    data=body,headers={"Content-Type":"application/json"})
try:
    urllib.request.urlopen(r,timeout=300).read()
except Exception as e:
    print("req",n,"err",e)
PY
    sleep 3
  done
}

run_leg() { # $1 = tag, $2 = binary
  echo "=== leg $1 : $2 ==="
  md5sum "$2"
  pkill -f 'infera-router' 2>/dev/null; sleep 2
  RUST_LOG=info nohup "$2" --host 0.0.0.0 --port "$PORT" \
      --etcd-endpoint "$ETCD" --router-policy kv-aware \
      --discovery-backend etcd --request-transport http \
      --kv-tokenizer-path "$MODEL" \
      --kv-overlap-weight 1.0 --kv-prefill-overlap-weight 20.0 \
      --kv-decode-overlap-weight 2.0 \
      > "$LOG/$1.log" 2>&1 &
  # Discovery + the kv-event SUB need time to attach before traffic; a router
  # that has not yet subscribed reads 0 for a trivial reason, which would look
  # exactly like the bug.
  sleep 25
  curl -sf -m5 "http://10.2.122.10:$PORT/health" >/dev/null && echo "  router up" \
    || { echo "  ROUTER DOWN"; tail -20 "$LOG/$1.log"; return 1; }
  drive "$1"
  sleep 3
  echo "  --- cache_hits seen on picks ---"
  grep -o 'cache_hits=[0-9]*' "$LOG/$1.log" | sort | uniq -c
  echo "  --- max ---"
  grep -o 'cache_hits=[0-9]*' "$LOG/$1.log" | sed 's/.*=//' | sort -n | tail -1
  pkill -f 'infera-router' 2>/dev/null; sleep 2
}

run_leg before /usr/local/bin/infera-router
run_leg after  /opt/infera/rust/target/release/infera-router
echo "=== done; logs in $LOG ==="
