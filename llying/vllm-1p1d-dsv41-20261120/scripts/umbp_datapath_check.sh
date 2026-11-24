#!/usr/bin/env bash
# UMBP data-path check for a GEM prefill: bring up the kit (standalone UMBP over GEM IPC), send one ~40k-token needle
# prompt, drop the prefill GPU prefix cache, send it again. The second answer must still be right and the prefill's
# external (connector) prefix-cache hit counter must move: the KV went GPU -> UMBP DRAM -> GPU through the IPC mapping.
#   INFERA_IMAGE (required), RUN. Runs on the node.
set -uo pipefail
source "$(dirname "$0")/env.sh"
KIT=/perf_apps/liyingli/bench_agentx/upstream/Infera-vllm-1p1d-dsv41/examples/vllm_1p1d_dsv41_flash
: "${INFERA_IMAGE:?INFERA_IMAGE}"
RUN=${RUN:-umbp-datapath-$(date -u +%m%dT%H%M)}
export LOG_DIR=$W/logs/$RUN
mkdir -p "$LOG_DIR" && chmod 777 "$LOG_DIR"
say() { echo "$(ts) $RUN: $*" | tee -a "$W/logs/summary.log"; }
export INFERA_IMAGE PREFILL_NODE=$(hostname -s) PREFILL_IP=$(node_ip)
export MODEL_MOUNT=/perf_apps/data/models MODEL=$MODEL
export RDMA_IB_DEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 MC_GID_INDEX=1
export CTR=${CTR_PREFIX}pd UMBP_DIR=/dev/shm/${CTR_PREFIX}umbp-pd ETCD_PORT=22379 ROUTER_PORT=$ROUTER_PORT
export UMBP_MODE=standalone UMBP_GEM_IPC=1
wrap() { bash "$KIT/cluster/cluster.gem.sh" "$@"; }
trap 'wrap down > "$LOG_DIR/down.log" 2>&1' EXIT
say "start on $PREFILL_NODE: P GEM 0.9 + standalone UMBP over GEM IPC"
wrap down > "$LOG_DIR/down0.log" 2>&1
wrap up > "$LOG_DIR/up.log" 2>&1 || { say "up FAILED: $(grep -a 'ERROR\|died' "$LOG_DIR/up.log" | tail -2 | tr '\n' ' ')"; exit 1; }
ext() { curl -s -m 10 "http://$PREFILL_IP:30000/metrics" | grep -aE '^vllm:(external_prefix_cache_(hits|queries)|prefix_cache_(hits|queries))_total' \
          | awk '{split($1,a,"{"); s[a[1]]+=$2} END {for (k in s) printf "%s=%d ", k, s[k]}'; }
python3 - "$LOG_DIR/prompt.json" <<'PY'
import json, random, sys
r = random.Random(11)
w = "river mountain window garden lantern bridge tower valley ocean island desert canyon teacher farmer sailor".split()
sents = [" ".join(r.choice(w) for _ in range(12)).capitalize() + "." for _ in range(2600)]
sents.insert(1700, "Record Quinn: the access code is K7-4821.")
body = {"model": "deepseek-ai/DeepSeek-V4.1-Flash", "temperature": 0, "max_tokens": 2048,
        "messages": [{"role": "user", "content": " ".join(sents) + "\n\nWhat is the access code of record Quinn? Answer with the code only."}]}
json.dump(body, open(sys.argv[1], "w"))
PY
ask() { curl -s -m 900 "http://$PREFILL_IP:$ROUTER_PORT/v1/chat/completions" -H 'Content-Type: application/json' -H 'X-Session-ID: umbp-check' \
          -d @"$LOG_DIR/prompt.json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["usage"]["prompt_tokens"], repr((d["choices"][0]["message"].get("content") or "").strip()[:40]))' 2>&1; }
m0=$(ext); a1=$(ask); sleep 5; m1=$(ext)
say "first: $a1 | metrics $m1"
r=$(curl -s -o /dev/null -w '%{http_code}' -X POST -m 60 "http://$PREFILL_IP:30000/reset_prefix_cache")
sleep 5
a2=$(ask); sleep 5; m2=$(ext)
say "GPU prefix cache reset HTTP $r; second: $a2 | metrics $m2"
say "UMBP server: $(grep -aE 'registered GPU IPC' "$LOG_DIR/umbp.log" | wc -l) GPU IPC clients; hip_gem_alloc IPC errors: $(grep -ac 'IPC open' "$LOG_DIR/umbp.log")"
say "end"
