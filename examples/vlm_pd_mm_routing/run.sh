#!/bin/bash
# VLM multimodal-affinity routing on PD disaggregation — one-command demo.
#
# Brings up, on a single MI355X node: 2 PREFILL + 2 DECODE SGLang Qwen2.5-VL
# workers (KV cache moved over mooncake_tcp — no RDMA) self-registering into
# etcd, fronted by the Infera router. It then sends 20 requests all carrying the
# SAME image under two policies and reads the routing decision from the router
# log:
#
#   kv-aware  — image affinity co-locates every repeat on ONE prefill worker
#               (it holds that image's warm vision/prefix cache).
#   round-robin (control) — the same workload splits evenly across the pool.
#
# The router is run straight from this repo (`--router-backend python`), so it
# exercises the source in this checkout with no image rebuild. Everything is
# env-driven — you should not need to edit this script.
#
#   bash run.sh              # run the demo, tear down at the end
#   KEEP=1 bash run.sh       # leave the containers up to poke at
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

IMG=${IMG:-rocm/infera:sglang-v0.1.1}
ETCD_IMG=${ETCD_IMG:-quay.io/coreos/etcd:v3.5.14}
MODEL=${MODEL:-/mnt/vast/john/huggingface/Qwen2.5-VL-7B-Instruct}
NAME=${NAME:-qwen-vl}
MOUNT=${MOUNT:-/mnt/vast}                       # shared mount covering $REPO and $MODEL
NODE_IP=${NODE_IP:-$(hostname -I | awk '{print $1}')}   # disagg peers need a routable IP
NREQ=${NREQ:-20}
KEEP=${KEEP:-0}
read -r -a P_GPUS <<<"${P_GPUS:-0 2}"           # one prefill worker per GPU
read -r -a D_GPUS <<<"${D_GPUS:-4 6}"           # one decode worker per GPU
P_PORTS=(30000 30001); D_PORTS=(30002 30003); P_BOOT=(8998 8999)

NAMES=(mm_etcd mm_router mm_p0 mm_p1 mm_d0 mm_d1)
cleanup() { docker rm -f "${NAMES[@]}" >/dev/null 2>&1; }
trap '[ "$KEEP" = 1 ] && echo "(KEEP=1 — containers left up; docker rm -f ${NAMES[*]})" || cleanup' EXIT
strip_ansi() { sed -r 's/\x1b\[[0-9;]*m//g'; }

echo "== [1/5] etcd (shared PD registry) =="
cleanup
docker run -d --name mm_etcd --network host "$ETCD_IMG" \
  /usr/local/bin/etcd --data-dir /tmp/etcd-mm \
  --listen-client-urls http://0.0.0.0:2379 \
  --advertise-client-urls http://127.0.0.1:2379 >/dev/null
sleep 3

launch_worker() {  # name gpu role port [bootstrap_port]
  local name=$1 gpu=$2 role=$3 port=$4 boot=${5:-}
  local rflags="--disaggregation-mode $role"
  [ -n "$boot" ] && rflags+=" --disaggregation-bootstrap-port $boot"
  docker run -d --name "$name" --network host --ipc host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --shm-size 32g --ulimit memlock=-1 \
    -v "$MOUNT":"$MOUNT" -e HF_HOME="$MOUNT/.cache/huggingface" \
    -e HIP_VISIBLE_DEVICES="$gpu" \
    -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200 \
    -e SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200 \
    "$IMG" \
    python -m infera.engine.sglang \
      --model-path "$MODEL" --served-model-name "$NAME" --trust-remote-code \
      --page-size 16 --context-length 8192 --mem-fraction-static 0.5 \
      --host 0.0.0.0 --port "$port" --advertise-host "$NODE_IP" \
      --discovery-backend etcd --etcd-endpoint 127.0.0.1:2379 \
      --request-transport http --kv-event-transport zmq \
      --disaggregation-transfer-backend mooncake_tcp $rflags >/dev/null
  echo "   $name: gpu $gpu role=$role port=$port ${boot:+bootstrap=$boot}"
}

echo "== [2/5] launch 2 prefill + 2 decode Qwen2.5-VL workers =="
for i in 0 1; do launch_worker "mm_p$i" "${P_GPUS[$i]}" prefill "${P_PORTS[$i]}" "${P_BOOT[$i]}"; done
for i in 0 1; do launch_worker "mm_d$i" "${D_GPUS[$i]}" decode  "${D_PORTS[$i]}"; done

echo "== [3/5] wait for workers to load (~3-4 min) =="
for p in "${P_PORTS[@]}" "${D_PORTS[@]}"; do
  for t in $(seq 1 100); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$p/health" 2>/dev/null)" = 200 ] \
      && { echo "   :$p ready"; break; }
    sleep 4
    [ "$t" = 100 ] && { echo "   :$p NEVER READY"; exit 2; }
  done
done

start_router() {  # policy
  docker rm -f mm_router >/dev/null 2>&1
  docker run -d --name mm_router --network host -v "$MOUNT":"$MOUNT" \
    -w "$REPO" -e PYTHONPATH="$REPO" --entrypoint python "$IMG" \
    -m infera.server --router-backend python --router-policy "$1" \
    --discovery-backend etcd --etcd-endpoint 127.0.0.1:2379 \
    --request-transport http --kv-event-transport zmq \
    --router-tokenizer-path "$MODEL" --host 0.0.0.0 --port 8000 >/dev/null
  for t in $(seq 1 40); do
    [ "$(curl -s http://127.0.0.1:8000/v1/workers 2>/dev/null | grep -o "$NAME" | wc -l)" -ge 4 ] \
      && return 0
    sleep 2
  done
  echo "   router[$1] did not discover 4 workers"; return 1
}

# prefill-pool distribution (image affinity lives in prefill: the image is
# processed there). Returns "wA=nA wB=nB".
prefill_dist() {
  docker logs mm_router 2>&1 | strip_ansi | grep 'pick policy=' | grep 'role=prefill' \
    | grep -oE 'picked=[^ ]+' | sort | uniq -c | awk '{printf "%s×%s ", $2, $1}'
}

declare -A RESULT
for policy in kv-aware round-robin; do
  echo "== [4/5] policy=$policy — $NREQ requests, same image =="
  start_router "$policy" || exit 1
  python3 "$HERE/mm_probe.py" --model "$NAME" --n "$NREQ" | sed 's/^/   /'
  RESULT[$policy]=$(prefill_dist)
  echo "   prefill pool: ${RESULT[$policy]}"
done

echo "== [5/5] verdict (prefill-pool routing of $NREQ identical-image requests) =="
printf "   %-14s %s\n" "kv-aware:"    "${RESULT[kv-aware]}   <- affinity co-locates"
printf "   %-14s %s\n" "round-robin:" "${RESULT[round-robin]}   <- control splits evenly"
