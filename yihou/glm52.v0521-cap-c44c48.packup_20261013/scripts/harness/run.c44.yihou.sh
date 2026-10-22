#!/usr/bin/env bash
# Same-node 1P1D P4D4 on crsuse2-m2m-102 with infera-sglang:v0521-yihou-1009.
# Modelled on kvcap_20261008/rounds/r013-1p1d-validate/run.cmd.yihou.sh (proven on 102),
# but every port/name/flag comes from the config, and container names/ports match
# launch.sh rows() so agentx_bench.sh / stop.sh / agentx_env.py see one deployment.
#
# Usage (from the login host):
#   bash run.c44.yihou.sh dryrun                 render etcd/engine/router commands, start nothing
#   bash run.c44.yihou.sh up                     gate -> etcd -> prefill -> decode (3 tries) -> router
#   bash run.c44.yihou.sh retry-decode a4        decode alone again; prefill/etcd/router untouched
#   bash run.c44.yihou.sh router                 (re)start only the router
#   bash run.c44.yihou.sh census|audit|smoke     post-bring-up checks
# Knobs: YIHOU_ROUND (default c44a; scopes container names and the run dir),
#        CONFIG (default ./config.yihou.102-v0521.sh), any KEY=VALUE the config reads.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${CONFIG:=$HERE/config.yihou.102-v0521.sh}"
export PATH="/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH"
export SPUR_SSH_JOBS="${SPUR_SSH_JOBS:-crsuse2-m2m-102:200276}"
set -a; source "$CONFIG"; set +a

NODE="$YIHOU_NODE"
IP="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$NODE")"
RD="$WS/runs/$YIHOU_ROUND"
P_PORT=$ENGINE_PORT_BASE
D_PORT=$((ENGINE_PORT_BASE + ENGINE_PORT_STRIDE))
ETCD_EP="$IP:$ETCD_PORT"
[[ "$PD_DP_RANK_AFFINITY" == 1 ]] && AFF=true || AFF=false   # router clap: true|false only
SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=10
     -o ServerAliveInterval=30 -o ServerAliveCountMax=1000)
mkdir -p "$RD/server-logs"
mark(){ printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" | tee -a "$RD/status.tsv"; }
hc(){ "${SSH[@]}" -n "$NODE" "curl -s -o /dev/null -w '%{http_code}' -m 25 http://$IP:$1/health" 2>/dev/null; }
q(){ local o="" a; for a in "$@"; do printf -v a '%q' "$a"; o+="${o:+ }$a"; done; echo "$o"; }
COMMON=("CONFIG=$CONFIG" "YIHOU_ROUND=$YIHOU_ROUND" "CONTAINER_PREFIX=$CONTAINER_PREFIX")
[[ "${DRY_RUN:-0}" == 1 ]] && COMMON+=(DRY_RUN=1)

etcd_cmd=(docker run -d --init --name "$CONTAINER_PREFIX-etcd" --network host
  quay.io/coreos/etcd:v3.5.14 etcd
  --advertise-client-urls "http://$ETCD_EP" --listen-client-urls "http://0.0.0.0:$ETCD_PORT"
  --listen-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT"
  --initial-advertise-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT"
  --initial-cluster "default=http://127.0.0.1:$ETCD_PEER_PORT")
router_cmd=(docker run -d --init --name "$CONTAINER_PREFIX-router" --network host
  -e "INFERA_PD_DP_RANK_AFFINITY=$AFF"
  -v "$WS:$WS" -v "$MODEL:$MODEL:ro"
  --tmpfs "$WS/tmp/$NODE_KEY/router:rw,exec,size=2g"
  -e "TMPDIR=$WS/tmp/$NODE_KEY/router" -e "HF_HOME=$WS/cache/$NODE_KEY/router"
  -e "XDG_CACHE_HOME=$WS/cache/$NODE_KEY/router" -e PYTHONDONTWRITEBYTECODE=1
  "$IMAGE"
  python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT" --router-backend rust
  --discovery-backend etcd --etcd-endpoint "$ETCD_EP"
  --request-transport http --kv-event-transport zmq
  --router-tokenizer-path "$MODEL" --router-policy kv-aware
  --kv-prefill-overlap-weight "$KV_PREFILL_OVERLAP_WEIGHT"
  --kv-decode-overlap-weight "$KV_DECODE_OVERLAP_WEIGHT")

engine(){  # role attempt-tag
  local role="$1" tag="$2" gpus port boot kv snap
  if [[ "$role" == prefill ]]; then
    gpus=$PREFILL_GPU_DEVICES; port=$P_PORT; boot=$BOOTSTRAP_PORT_BASE
    kv=$KV_EVENT_PORT_BASE; snap=$SNAPSHOT_PORT_BASE
  else
    gpus=$DECODE_GPU_DEVICES; port=$D_PORT; boot=$((BOOTSTRAP_PORT_BASE + 1))
    kv=$((KV_EVENT_PORT_BASE + 1)); snap=$((SNAPSHOT_PORT_BASE + 1))
  fi
  "${SSH[@]}" -n "$NODE" bash "$BENCH_DIR/engine.sh" \
    "$role" "$role-0" "$IP" "$gpus" "$port" "$boot" "$kv" "$snap" \
    "$CONTAINER_PREFIX-$role-0" "$ETCD_EP" "${COMMON[@]}" \
    "SERVER_LOG=$RD/server-logs/$role-0$tag.log"
}

gpu_idle_gate(){
  local i mb
  for i in $(seq 1 80); do
    mb=$("${SSH[@]}" -n "$NODE" 'rocm-smi --showmeminfo vram --json 2>/dev/null' |
      python3 -c 'import sys,json;d=json.load(sys.stdin);print(max(int(v["VRAM Total Used Memory (B)"]) for v in d.values())//1048576)' 2>/dev/null)
    [[ -n "$mb" && "$mb" -lt 400 ]] && { mark gpuidle "ok:max=${mb}MiB"; return 0; }
    echo "GPUs not idle yet (max ${mb:-?} MiB); HiCache/VRAM release is slow by design"
    sleep 30
  done
  mark gpuidle "FAIL:max=${mb:-?}"; return 1
}

start_decode(){  # attempt tag; renames a dead decode aside, never removes it
  local att="$1" i c
  "${SSH[@]}" -n "$NODE" "s=\$(docker inspect -f '{{.State.Status}}' $CONTAINER_PREFIX-decode-0 2>/dev/null) || exit 0; \
    [ \"\$s\" = running ] && docker stop -t 180 $CONTAINER_PREFIX-decode-0 >/dev/null; \
    docker rename $CONTAINER_PREFIX-decode-0 $CONTAINER_PREFIX-decode-dead-$att" || return 1
  engine decode ".$att" > "$RD/engine-decode.$att.log" 2>&1 || { mark "decode-$att" "FAIL:engine.sh"; return 1; }
  for i in $(seq 1 120); do   # cold AITER JIT + CUDA-graph capture: allow 30 min
    c=$(hc "$D_PORT"); [[ "$c" == 200 ]] && { mark "decode-$att" ok; return 0; }
    "${SSH[@]}" -n "$NODE" "docker inspect -f '{{.State.Running}}' $CONTAINER_PREFIX-decode-0" 2>/dev/null |
      grep -q true || { mark "decode-$att" "FAIL:exited"; return 1; }
    sleep 15
  done
  mark "decode-$att" "FAIL:health=$c"; return 1
}

start_router(){
  "${SSH[@]}" -n "$NODE" "$(q "${router_cmd[@]}")" || { mark router FAIL; return 1; }
  local i c
  for i in $(seq 1 60); do c=$(hc "$ROUTER_PORT"); [[ "$c" == 200 ]] && break; sleep 5; done
  [[ "$c" == 200 ]] || { mark router "FAIL:health=$c"; return 1; }
  "${SSH[@]}" -n "$NODE" "curl -s -m 30 http://$IP:$ROUTER_PORT/v1/workers" > "$RD/workers.json"
  python3 - "$RD/workers.json" "$IP" "$P_PORT" "$D_PORT" <<'PY' || { mark router "FAIL:workers"; return 1; }
import json, sys
w = json.load(open(sys.argv[1]))
w = w if isinstance(w, list) else (w.get("workers") or w.get("data") or [])
urls = sorted(str(x.get("base_url") or x.get("url")) for x in w)
want = sorted(f"http://{sys.argv[2]}:{p}" for p in sys.argv[3:])
print("router workers:", urls)
sys.exit(0 if urls == want else 1)
PY
  mark router ok
}

case "${1:-}" in
dryrun)
  export DRY_RUN=1; COMMON+=(DRY_RUN=1)
  echo "# config=$CONFIG round=$YIHOU_ROUND node=$NODE ip=$IP"
  echo "# ports: etcd=$ETCD_PORT router=$ROUTER_PORT prefill=$P_PORT decode=$D_PORT"
  echo "## etcd"; q "${etcd_cmd[@]}"
  echo "## prefill-0 (engine.sh DRY_RUN=1 on $NODE)"; engine prefill ""
  echo "## decode-0 (engine.sh DRY_RUN=1 on $NODE)"; engine decode ".a1"
  echo "## router"; q "${router_cmd[@]}"
  ;;
up)
  mark start "round=$YIHOU_ROUND image=$IMAGE"
  "${SSH[@]}" -n "$NODE" "docker ps -a --format '{{.Names}}' | grep -q '^$CONTAINER_PREFIX-'" &&
    { echo "containers with prefix $CONTAINER_PREFIX exist; use a new YIHOU_ROUND" >&2; exit 1; }
  gpu_idle_gate || exit 1
  "${SSH[@]}" -n "$NODE" "$(q "${etcd_cmd[@]}")" || { mark etcd FAIL; exit 1; }
  for i in $(seq 1 60); do
    "${SSH[@]}" -n "$NODE" docker exec "$CONTAINER_PREFIX-etcd" etcdctl \
      --endpoints "http://127.0.0.1:$ETCD_PORT" endpoint health >/dev/null 2>&1 && break
    [[ $i == 60 ]] && { mark etcd "FAIL:health"; exit 1; }; sleep 1
  done
  mark etcd ok
  engine prefill "" > "$RD/engine-prefill.log" 2>&1 || { mark prefill "FAIL:engine.sh"; exit 1; }
  for i in $(seq 1 180); do   # cold: ~36 min (450 GB host pool + 4x103 GB indexer pools + JIT)
    c=$(hc "$P_PORT"); [[ "$c" == 200 ]] && { mark prefill "ok:${i}x20s"; break; }
    [[ $i == 180 ]] && { mark prefill "FAIL:health=$c"; exit 1; }; sleep 20
  done
  # Same-node RCCL-init race: decode may die in ncclCommInitRank. Retry decode
  # ONLY; prefill is never restarted. Settle before the first try (C40-full ref).
  sleep "${DECODE_SETTLE_S:-120}"
  ok=0
  for att in a1 a2 a3; do
    start_decode "$att" && { ok=1; break; }
    echo "decode $att failed; retrying decode only (prefill untouched)"; sleep 60
  done
  [[ $ok == 1 ]] || { mark decode "FAIL:all-attempts"; exit 1; }
  mark bothlegs ok
  start_router || exit 1
  ;;
retry-decode)
  [[ "$(hc "$D_PORT")" == 200 ]] && { echo "decode is healthy; refusing to restart it" >&2; exit 1; }
  start_decode "${2:?attempt tag, e.g. a4}" ;;
router) start_router ;;
census)
  for f in "$RD"/server-logs/prefill-0.log $(ls -t "$RD"/server-logs/decode-0.*.log 2>/dev/null | head -1); do
    echo "== $f"
    echo "  Found N HCAs   : $(strings "$f" | grep -ao 'Found [0-9]* HCAs' | sort | uniq -c | tr '\n' ' ')  (want: 4 x Found 0 HCAs)"
    echo "  HIP installed  : $(strings "$f" | grep -c 'HIP transport installed')  (want 4)"
    echo "  HIP inst FAIL  : $(strings "$f" | grep -c 'Failed to install HIP transport')  (want 0)"
    echo "  isHipReachable : $(strings "$f" | grep -c 'isHipReachable')  (r013 baseline: 0 per leg; informational)"
    echo "  rdma installs  : $(strings "$f" | grep -c 'installTransport, type=rdma')  (want 0)"
    echo "  Invalid arg    : $(strings "$f" | grep -c 'Invalid argument')  (want 0)"
    echo "  hipIpc failed  : $(strings "$f" | grep -c 'hipIpc[A-Za-z]* failed')  (want 0)"
    echo "  OOM/Traceback  : $(strings "$f" | grep -ci 'out of memory')/$(strings "$f" | grep -c Traceback)  (want 0/0)"
    strings "$f" | grep -ao 'max_total_num_tokens=[0-9]*.*available_gpu_mem=[0-9.]* GB' | tail -1
    strings "$f" | grep -ao 'Allocating .*host.*GB' | sort | uniq -c | head -4
  done
  ;;
audit)
  python3 "$HERE/tools/audit_live.yihou.py" --config "$CONFIG" --node "$NODE" \
    --prefill "http://$IP:$P_PORT" --decode "http://$IP:$D_PORT" \
    --prefix "$CONTAINER_PREFIX" --out "$RD/audit" ;;
smoke)
  "${SSH[@]}" -n "$NODE" "curl -s -m 300 http://$IP:$ROUTER_PORT/v1/completions -H 'Content-Type: application/json' \
    -d '{\"model\":\"$SERVED_MODEL\",\"prompt\":\"List the first five prime numbers:\",\"max_tokens\":40,\"temperature\":0}'" \
    | tee "$RD/smoke-response.json"; echo
  grep -q '"text"' "$RD/smoke-response.json" && mark smoke ok || { mark smoke FAIL; exit 1; }
  echo "NOTE: simulated acceptance is ON -> the text is expected to be garbled; this only proves KV moved."
  ;;
*) sed -n '2,15p' "$0"; exit 2 ;;
esac
