#!/usr/bin/env bash
# Cheap, repeatable decode-throughput probe for the CUDA-graph debug loop.
# Not a benchmark — a comparable number across rounds. Fixed output length,
# ignore_eos so every round decodes exactly the same token count, and a warmup
# request first because this engine JIT-compiles a KDA Triton kernel AFTER
# serving starts (1.6-2.5 s a time) and that would land in round 1 only.
set -u
MY_IP="${MY_IP:?MY_IP=IP of this node}"
PORT="${PORT:-8100}"
SERVED="${SERVED:-glm5.3-flash}"
CONC="${CONC:-1}"
OUT_TOKENS="${OUT_TOKENS:-256}"
URL="http://$MY_IP:$PORT/v1/chat/completions"

req() {
  curl -s -m600 "$URL" -H 'Content-Type: application/json' -d "{
    \"model\": \"$SERVED\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Write a detailed description of the water cycle.\"}],
    \"temperature\": 1.0, \"top_p\": 0.95,
    \"max_tokens\": $OUT_TOKENS,
    \"ignore_eos\": true
  }"
}

echo "warmup..."; req >/dev/null

start=$(date +%s.%N)
tmp=$(mktemp -d)
for i in $(seq 1 "$CONC"); do req > "$tmp/r$i.json" & done
wait
end=$(date +%s.%N)

python3 - "$tmp" "$start" "$end" "$CONC" <<'PY'
import json, sys, glob, os
tmp, start, end, conc = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
tot = 0; n = 0
for f in glob.glob(os.path.join(tmp, "*.json")):
    try:
        d = json.load(open(f))
        tot += d["usage"]["completion_tokens"]; n += 1
    except Exception as e:
        print("BAD RESPONSE", f, e)
el = end - start
print(f"concurrency   : {conc} (responses ok: {n})")
print(f"elapsed       : {el:.2f} s")
print(f"output tokens : {tot}")
print(f"decode tok/s  : {tot/el:.2f}   (aggregate)")
print(f"per-user tok/s: {tot/el/max(n,1):.2f}")
PY
rm -rf "$tmp"
