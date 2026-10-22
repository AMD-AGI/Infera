#!/usr/bin/env bash
# run.v3x (task g4tune xnode, 2026-10-21): CROSS-NODE copy of run.v3.yihou.sh. Prefill, etcd, router and the
# AgentX client on YIHOU_NODE/YIHOU_IP; decode on XNODE_DECODE_NODE/XNODE_DECODE_IP (config.yihou.xnode.sh,
# XNODE_* passed in YIHOU_OVERRIDES). Same stages and checks as run.v3; differences are marked "xnode".
# Original header (run.v3):
# Node-agnostic same-node 1P1D P4D4 driver for infera-sglang:v0521-yihou-1009 (v0521dbg).
# Adapted from agentx44_20261009/harness/run.c44.yihou.sh (same RCCL-race handling,
# GPU-idle gate, router with INFERA_PD_DP_RANK_AFFINITY), plus census extensions and bench/stop.
#
# Usage (from the login host):
#   CONFIG=.../config.yihou.E1-relaxed.sh YIHOU_NODE=crsuse2-m2m-NNN YIHOU_IP=a.b.c.d \
#   YIHOU_JOB=NNNNNN YIHOU_ROUND=e1a bash run.v2.yihou.sh MODE
# MODE:
#   dryrun              render etcd/engine/router argv locally; NO ssh, NO docker (stubbed)
#   up                  gate -> etcd -> prefill -> decode (3 tries) -> router
#   retry-decode aN     decode alone again; prefill/etcd/router untouched
#   router              (re)start only the router
#   census|audit|smoke  post-bring-up checks
#   bench               AgentX (CONC from the config) -> runs/<round>/agentx-c<CONC>
#   stop                bench-harness/stop.sh for this round (stop, never rm)
# YIHOU_JOB= (empty) -> plain ssh to YIHOU_NODE; non-empty -> spur exec shim (SPUR_SSH_JOBS).
# Knob overrides that must reach the node-side engine.sh and agentx_bench.sh too:
#   YIHOU_OVERRIDES="DECODE_MEM_FRACTION=0.92 ..." (space-separated KEY=VALUE).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${CONFIG:?set CONFIG to an overlay, e.g. $HERE/config.yihou.E1-relaxed.sh}"
: "${YIHOU_OVERRIDES:=}"
read -r -a OVR <<< "$YIHOU_OVERRIDES"
for kv in "${OVR[@]}"; do
  [[ "$kv" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "bad YIHOU_OVERRIDES item: $kv" >&2; exit 2; }
  export "$kv"
done
set -a; source "$CONFIG" || exit 2; set +a
# Overlay chains swallow a nested `return 2`; re-check the hard guards here.
[[ -n "${ENGINE_PORT_BASE:-}" ]] || { echo "config did not load completely (see errors above)" >&2; exit 2; }
[[ "${ENGINE_PID_HOST:-0}" == 1 && -n "$YIHOU_JOB" ]] && { echo "ENGINE_PID_HOST=1 refused on spur node (spur-authz denies --pid=host)" >&2; exit 2; }

NODE="$YIHOU_NODE"
IP="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$NODE")" || exit 2
[[ "$IP" == "$YIHOU_IP" ]] || { echo "topology IP $IP != YIHOU_IP $YIHOU_IP" >&2; exit 2; }
# xnode: the decode leg's host
: "${XNODE_DECODE_NODE:?xnode: set XNODE_DECODE_NODE (in YIHOU_OVERRIDES)}"
DNODE="$XNODE_DECODE_NODE"
DIP="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$DNODE")" || exit 2
[[ "$DIP" == "$XNODE_DECODE_IP" && "$DNODE" != "$NODE" ]] || { echo "xnode: topology decode $DNODE/$DIP mismatch" >&2; exit 2; }
RD="$WS/runs/$YIHOU_ROUND"
P_PORT=$ENGINE_PORT_BASE
D_PORT=$((ENGINE_PORT_BASE + ENGINE_PORT_STRIDE))
ETCD_EP="$IP:$ETCD_PORT"
[[ "$PD_DP_RANK_AFFINITY" == 1 ]] && AFF=true || AFF=false   # router clap: true|false only
SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes -o ConnectTimeout=10
     -o ServerAliveInterval=30 -o ServerAliveCountMax=1000)
mark(){ printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" | tee -a "$RD/status.tsv"; }
hc(){ local n="${2:-$NODE}" i="${3:-$IP}"   # xnode: port [node ip]
  "${SSH[@]}" -n "$n" "curl -s -o /dev/null -w '%{http_code}' -m 25 http://$i:$1/health" 2>/dev/null; }
q(){ local o="" a; for a in "$@"; do printf -v a '%q' "$a"; o+="${o:+ }$a"; done; echo "$o"; }
# Everything engine.sh / agentx_bench.sh / stop.sh need to re-source the SAME config.
COMMON=("CONFIG=$CONFIG" "YIHOU_NODE=$YIHOU_NODE" "YIHOU_IP=$YIHOU_IP" "YIHOU_JOB=$YIHOU_JOB"
        "YIHOU_ROUND=$YIHOU_ROUND" "CONTAINER_PREFIX=$CONTAINER_PREFIX"
        "IMAGE=$IMAGE" "EXPECTED_IMAGE_ID=$EXPECTED_IMAGE_ID" "${OVR[@]}")
[[ -n "${YIHOU_NIC+x}" ]] && COMMON+=("YIHOU_NIC=$YIHOU_NIC")
case "${1:-}" in
  up|retry-decode) [[ "$EXPECTED_IMAGE_ID" == sha256:* ]] ||
    { echo "set EXPECTED_IMAGE_ID (docker image inspect on $NODE)" >&2; exit 2; } ;;
esac
case "${1:-}" in up|retry-decode|router|audit|smoke|bench) mkdir -p "$RD/server-logs" ;; esac

etcd_cmd=(docker run -d --init --name "$CONTAINER_PREFIX-etcd" --network host
  quay.io/coreos/etcd:v3.5.14 etcd
  --advertise-client-urls "http://$ETCD_EP" --listen-client-urls "http://0.0.0.0:$ETCD_PORT"
  --listen-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT"
  --initial-advertise-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT"
  --initial-cluster "default=http://127.0.0.1:$ETCD_PEER_PORT")
router_cmd=(docker run -d --init --name "$CONTAINER_PREFIX-router" --network host
  -e "INFERA_PD_DP_RANK_AFFINITY=$AFF"
  -v "$WS:$WS${NODE_RW_ROOT:+:ro}" -v "$MODEL:$MODEL:ro"
  ${NODE_RW_ROOT:+-v "$NODE_RW_ROOT:$NODE_RW_ROOT"}
  --tmpfs "$NODE_RW/tmp/$NODE_KEY/router:rw,exec,size=2g"
  -e "TMPDIR=$NODE_RW/tmp/$NODE_KEY/router" -e "HF_HOME=$NODE_RW/cache/$NODE_KEY/router"
  -e "XDG_CACHE_HOME=$NODE_RW/cache/$NODE_KEY/router" -e PYTHONDONTWRITEBYTECODE=1
  "$IMAGE"
  python3 -m infera.server --host 0.0.0.0 --port "$ROUTER_PORT" --router-backend rust
  --discovery-backend etcd --etcd-endpoint "$ETCD_EP"
  --request-transport http --kv-event-transport zmq
  --router-tokenizer-path "$MODEL" --router-policy kv-aware
  --kv-prefill-overlap-weight "$KV_PREFILL_OVERLAP_WEIGHT"
  --kv-decode-overlap-weight "$KV_DECODE_OVERLAP_WEIGHT")

engine_argv(){  # role attempt-tag -> the engine.sh argv (without the transport)
  local role="$1" tag="$2" gpus port boot kv snap
  if [[ "$role" == prefill ]]; then
    gpus=$PREFILL_GPU_DEVICES; port=$P_PORT; boot=$BOOTSTRAP_PORT_BASE
    kv=$KV_EVENT_PORT_BASE; snap=$SNAPSHOT_PORT_BASE
  else
    gpus=$DECODE_GPU_DEVICES; port=$D_PORT; boot=$((BOOTSTRAP_PORT_BASE + 1))
    kv=$((KV_EVENT_PORT_BASE + 1)); snap=$((SNAPSHOT_PORT_BASE + 1))
  fi
  local lip=$IP; [[ "$role" == decode ]] && lip=$DIP   # xnode
  EARGV=(bash "$BENCH_DIR/engine.sh" "$role" "$role-0" "$lip" "$gpus" "$port" "$boot" "$kv" "$snap"
    "$CONTAINER_PREFIX-$role-0" "$ETCD_EP" "${COMMON[@]}" "SERVER_LOG=$NODE_RD/server-logs/$role-0$tag.log")
}
engine(){ local n=$NODE; [[ "$1" == decode ]] && n=$DNODE   # xnode
  engine_argv "$1" "$2"; "${SSH[@]}" -n "$n" "$(q "${EARGV[@]}")"; }

# NODE_RW_ROOT set: the node writes under $NODE_RD (node-local); pull it to $RD on the login
# host (caches are not copied). base64: the spur shim mangles binary stdout.
sync_back(){  # [subpath ...]   (default: everything under the node run dir)
  [[ -n "$NODE_RW_ROOT" ]] || return 0
  echo "xnode: NODE_RW_ROOT must be empty (logs of the decode node would not be copied)" >&2; return 1
  local what=("${@:-.}")
  local b64
  mkdir -p "$RD"
  b64=$("${SSH[@]}" -n "$NODE" "cd $(q "$NODE_RD") 2>/dev/null || exit 0; \
    tar --warning=no-file-changed -cf - $(q "${what[@]}") 2>/dev/null | base64 -w0") || return 1
  [[ -n "$b64" ]] || { echo "sync_back: nothing under $NODE_RD on $NODE" >&2; return 0; }
  base64 -d <<< "$b64" | tar -C "$RD" -xf -
}

# Fixed ports a leg binds: engine port, sglang's derived block (port+233 .. +233+6+DP, rust
# path: server_args.py:699,806), bootstrap, kv-event block (+attn_dp_rank), snapshot.
leg_ports(){  # role
  local p dp
  if [[ "$1" == prefill ]]; then p=$P_PORT; dp=$PREFILL_DP
    echo "$p $(seq -s ' ' $((p + 233)) $((p + 233 + 6 + dp))) $BOOTSTRAP_PORT_BASE $(seq -s ' ' $KV_EVENT_PORT_BASE $((KV_EVENT_PORT_BASE + dp - 1))) $SNAPSHOT_PORT_BASE"
  else p=$D_PORT; dp=$DECODE_DP
    echo "$p $(seq -s ' ' $((p + 233)) $((p + 233 + 6 + dp)))"
  fi
}
port_preflight(){  # what: all|decode|router.  Every fixed port < 30000 and not listening.
  local ports="" x bad=0 busy
  case "$1" in
    all)    port_preflight decode || return 1   # xnode: decode block on the decode node, the rest here
            ports="$ETCD_PORT $ETCD_PEER_PORT $ROUTER_PORT $(leg_ports prefill)" ;;
    decode) ports="$(leg_ports decode)" ;;
    router) ports="$ROUTER_PORT" ;;
  esac
  local pn=$NODE; [[ "$1" == decode ]] && pn=$DNODE   # xnode
  for x in $ports; do (( x < 30000 )) || { echo "port_preflight: $x >= 30000 (ephemeral/NodePort hazard)" >&2; bad=1; }; done
  # IPv4/IPv6 listeners on any address
  busy=$("${SSH[@]}" -n "$pn" "ss -Htln | awk '{print \$4}' | sed 's/.*://' | sort -un" 2>/dev/null |
    grep -xF -f <(tr ' ' '\n' <<< "$ports" | grep .) | tr '\n' ' ')
  [[ -z "$busy" ]] || { echo "port_preflight: already listening on $pn: $busy" >&2; bad=1; }
  [[ $bad == 0 ]] && { mark "ports-$1" "ok:$(wc -w <<< "$ports") ports <30000 and free"; return 0; }
  mark "ports-$1" "FAIL"; return 1
}

gpu_idle_gate(){   # xnode: max over BOTH nodes
  local i mb idle_hits=0
  for i in $(seq 1 80); do
    mb=$( { "${SSH[@]}" -n "$NODE" 'rocm-smi --showmeminfo vram --json 2>/dev/null' |
      python3 -c 'import sys,json;d=json.load(sys.stdin);print(max(int(v["VRAM Total Used Memory (B)"]) for v in d.values())//1048576)' 2>/dev/null
      "${SSH[@]}" -n "$DNODE" 'rocm-smi --showmeminfo vram --json 2>/dev/null' |
      python3 -c 'import sys,json;d=json.load(sys.stdin);print(max(int(v["VRAM Total Used Memory (B)"]) for v in d.values())//1048576)' 2>/dev/null
      } | sort -n | awk 'NR==2{print $1} END{if(NR<2) print ""}')
    # Idle must hold on 2 consecutive checks >= 60 s apart: a foreign looping job (136 dev_xb)
    # has short gaps that a single check can hit.
    if [[ -n "$mb" && "$mb" -lt 400 ]]; then
      (( ${idle_hits:=0} >= 1 )) && { mark gpuidle "ok:max=${mb}MiB (2 consecutive)"; return 0; }
      idle_hits=1; echo "GPUs idle (max ${mb} MiB); confirming in 60 s"; sleep 60; continue
    fi
    idle_hits=0
    echo "GPUs not idle yet (max ${mb:-?} MiB); HiCache/VRAM release is slow by design"
    sleep 30
  done
  mark gpuidle "FAIL:max=${mb:-?}"; return 1
}

ipc_guard(){  # the live prefill must match the config on both IPC knobs (never one leg only)
  local live pm lg
  live=$("${SSH[@]}" -n "$NODE" "docker inspect -f '{{.HostConfig.PidMode}}|{{range .Config.Env}}{{println .}}{{end}}' $CONTAINER_PREFIX-prefill-0" 2>/dev/null) ||
    { echo "ipc_guard: cannot inspect $CONTAINER_PREFIX-prefill-0" >&2; return 1; }
  pm=0; [[ "${live%%|*}" == host ]] && pm=1
  lg=0; grep -qx 'HSA_ENABLE_IPC_MODE_LEGACY=1' <<< "${live#*|}" && lg=1
  [[ "$pm" == "$ENGINE_PID_HOST" && "$lg" == "$IPC_MODE_LEGACY" ]] && return 0
  echo "ipc_guard: prefill has pid_host=$pm legacy_ipc=$lg, config wants $ENGINE_PID_HOST/$IPC_MODE_LEGACY; refusing decode" >&2
  return 1
}

start_decode(){  # attempt tag; renames a dead decode aside, never removes it
  local att="$1" i c st
  ipc_guard || { mark "decode-$att" "FAIL:ipc-knob-mismatch"; return 1; }
  "${SSH[@]}" -n "$DNODE" "s=\$(docker inspect -f '{{.State.Status}}' $CONTAINER_PREFIX-decode-0 2>/dev/null) || exit 0; \
    [ \"\$s\" = running ] && docker stop -t 180 $CONTAINER_PREFIX-decode-0 >/dev/null; \
    docker rename $CONTAINER_PREFIX-decode-0 $CONTAINER_PREFIX-decode-dead-$att" || return 1
  engine decode ".$att" > "$RD/engine-decode.$att.log" 2>&1 || { mark "decode-$att" "FAIL:engine.sh"; return 1; }
  for i in $(seq 1 120); do   # cold AITER JIT + CUDA-graph capture: allow 30 min
    c=$(hc "$D_PORT" "$DNODE" "$DIP"); [[ "$c" == 200 ]] && { mark "decode-$att" ok; return 0; }
    # Only a definite "false" counts as exited; an ssh/spur-exec hiccup (empty output) does not.
    # (e2b a4: a transient failure here marked a healthy decode FAIL:exited.)
    st=$("${SSH[@]}" -n "$DNODE" "docker inspect -f '{{.State.Running}}' $CONTAINER_PREFIX-decode-0" 2>/dev/null)
    [[ "$st" == false ]] && { mark "decode-$att" "FAIL:exited"; return 1; }
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
  python3 - "$RD/workers.json" "$IP" "$P_PORT" "$DIP" "$D_PORT" <<'PY' || { mark router "FAIL:workers"; return 1; }
import json, sys
w = json.load(open(sys.argv[1]))
w = w if isinstance(w, list) else (w.get("workers") or w.get("data") or [])
urls = sorted(str(x.get("base_url") or x.get("url")) for x in w)
want = sorted([f"http://{sys.argv[2]}:{sys.argv[3]}", f"http://{sys.argv[4]}:{sys.argv[5]}"])  # xnode
print("router workers:", urls)
sys.exit(0 if urls == want else 1)
PY
  mark router ok
}

case "${1:-}" in
dryrun)
  # engine.sh runs LOCALLY with stubbed docker/ssh: nothing reaches the node.
  stub="$(mktemp -d "${TMPDIR:-/tmp}/yihou-dryrun-stub.XXXXXX")"
  printf '#!/bin/sh\necho "dryrun stub: docker $*" >&2\nexit 1\n' > "$stub/docker"
  printf '#!/bin/sh\necho "dryrun stub: ssh refused ($*)" >&2\nexit 255\n' > "$stub/ssh"
  chmod +x "$stub/docker" "$stub/ssh"
  echo "# exp=${YIHOU_EXP:-?} config=$CONFIG round=$YIHOU_ROUND node=$NODE ip=$IP job=$YIHOU_JOB decode_node=$DNODE decode_ip=$DIP"
  echo "# ports: etcd=$ETCD_PORT router=$ROUTER_PORT prefill=$P_PORT decode=$D_PORT  CONC=$CONC"
  echo "# topology=$TOPOLOGY"; sed 's/^/#   /' "$TOPOLOGY"
  echo "# NODE_RW_ROOT=${NODE_RW_ROOT:-<empty: node writes to \$WS on NFS>}  node run dir=$NODE_RD  login run dir=$RD"
  echo "# fixed ports prefill: $(leg_ports prefill)"
  echo "# fixed ports decode : $(leg_ports decode)"
  echo "## etcd"; q "${etcd_cmd[@]}"
  # engine.sh checks HOST_RDMA_LIB is readable; it exists on the node, not here. Point it
  # at an empty stub and print the real path back, so the rendered argv is the node's.
  rdma_lib=() ; [[ -r "$HOST_RDMA_LIB" ]] || { : > "$stub/libionic.so"; rdma_lib=("HOST_RDMA_LIB=$stub/libionic.so")
    echo "# HOST_RDMA_LIB $HOST_RDMA_LIB not readable here: rendered with a stub, path restored in the output"; }
  for r in prefill:"" decode:.a1; do
    echo "## ${r%%:*}-0 (engine.sh DRY_RUN=1, run locally, docker/ssh stubbed)"
    engine_argv "${r%%:*}" "${r#*:}"
    echo "# ${EARGV[-1]}"
    PATH="$stub:$PATH" "${EARGV[@]}" "${rdma_lib[@]}" DRY_RUN=1 2>&1 | sed "s#$stub/libionic.so#$HOST_RDMA_LIB#g"
  done
  echo "## router"; q "${router_cmd[@]}"
  rm -f "$stub/docker" "$stub/ssh" "$stub/libionic.so"; rmdir "$stub"
  ;;
up)
  mark start "exp=${YIHOU_EXP:-?} round=$YIHOU_ROUND image=$IMAGE config=$CONFIG"
  for n in "$NODE" "$DNODE"; do   # xnode: both hosts
    "${SSH[@]}" -n "$n" "docker ps -a --format '{{.Names}}' | grep -q '^$CONTAINER_PREFIX-'" &&
      { echo "containers with prefix $CONTAINER_PREFIX exist on $n; use a new YIHOU_ROUND" >&2; exit 1; }
  done
  port_preflight all || exit 1
  gpu_idle_gate || exit 1
  "${SSH[@]}" -n "$NODE" "$(q "${etcd_cmd[@]}")" || { mark etcd FAIL; exit 1; }
  for i in $(seq 1 60); do
    "${SSH[@]}" -n "$NODE" docker exec "$CONTAINER_PREFIX-etcd" etcdctl \
      --endpoints "http://127.0.0.1:$ETCD_PORT" endpoint health >/dev/null 2>&1 && break
    [[ $i == 60 ]] && { mark etcd "FAIL:health"; exit 1; }; sleep 1
  done
  mark etcd ok
  engine prefill "" > "$RD/engine-prefill.log" 2>&1 || { mark prefill "FAIL:engine.sh"; exit 1; }
  for i in $(seq 1 180); do   # cold: up to ~45 min (host pools + indexer pools + JIT)
    c=$(hc "$P_PORT"); [[ "$c" == 200 ]] && { mark prefill "ok:${i}x20s"; break; }
    [[ $i == 180 ]] && { mark prefill "FAIL:health=$c"; exit 1; }; sleep 20
  done
  # Same-node RCCL-init race: decode may die in ncclCommInitRank. Retry decode
  # ONLY; prefill is never restarted. Settle before the first try.
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
  [[ "$(hc "$D_PORT" "$DNODE" "$DIP")" == 200 ]] && { echo "decode is healthy; refusing to restart it" >&2; exit 1; }
  # The dead decode's ports are free once it exited; a live prefill's random sockets may not be.
  port_preflight decode || exit 1
  start_decode "${2:?attempt tag, e.g. a4}" ;;
router) start_router ;;
census)
  sync_back server-logs
  dlog="$(ls -t "$RD"/server-logs/decode-0.*.log 2>/dev/null | head -1)"
  for f in "$RD"/server-logs/prefill-0.log $dlog; do
    echo "== $f"
    echo "  Found N HCAs   : $(strings "$f" | grep -ao 'Found [0-9]* HCAs' | sort | uniq -c | tr '\n' ' ')  (xnode want: 4 x Found 4 HCAs)"
    echo "  HIP installed  : $(strings "$f" | grep -c 'HIP transport installed')  (want 4)"
    echo "  HIP inst FAIL  : $(strings "$f" | grep -c 'Failed to install HIP transport')  (want 0)"
    echo "  isHipReachable : $(strings "$f" | grep -c 'isHipReachable')  (r013 baseline: 0 per leg; informational)"
    echo "  rdma installs  : $(strings "$f" | grep -c 'installTransport, type=rdma')  (xnode want 4)"
    echo "  Invalid arg    : $(strings "$f" | grep -c 'Invalid argument')  (want 0)"
    echo "  hipIpc failed  : $(strings "$f" | grep -c 'hipIpc[A-Za-z]* failed')  (want 0)"
    # Tracebacks whose chain (next 40 lines) names a HEALTH_CHECK request are bring-up health
    # probe aborts ("Request is disconnected ... HEALTH_CHECK_"), not faults (seen on e2b2).
    tb_health=$(strings "$f" | awk '{a[NR]=$0} END{for(i=1;i<=NR;i++) if(a[i]~/^.*Traceback \(most recent call last\)/){h=0; for(j=i+1;j<=i+40&&j<=NR;j++) if(a[j]~/HEALTH_CHECK_/){h=1;break}; n+=h}; print n+0}')
    tb_all=$(strings "$f" | grep -c Traceback)
    echo "  OOM/Traceback  : $(strings "$f" | grep -ci 'out of memory')/$((tb_all - tb_health))  (want 0/0; + $tb_health health-check aborts, benign)"
    want_fi=4; [[ "$DSA_FUSED_INDEXER" == off ]] && want_fi=0
    echo "  fused idx ON   : $(strings "$f" | grep -c 'gfx950 fused DSA indexer enabled')  (want $want_fi: DSA_FUSED_INDEXER=$DSA_FUSED_INDEXER)"
    echo "  fused idx decl : $(strings "$f" | grep -c 'gfx950 fused DSA indexer \(disabled\|declined\|not used\)')  (informational)"
    # repro's REGBASE diagnostic (mounted conn.py; mc4340 wave) + the #4340 boundary warning
    rb=$(strings "$f" | grep -a 'YIHOU-REGBASE.*summary' | grep -ao 'interior=[0-9]*' | sort | uniq -c | tr '\n' ' ')
    echo "  REGBASE summary: ${rb:-none}  'extends past its allocation'=$(strings "$f" | grep -c 'extends past its allocation') (want 0)"
    strings "$f" | grep -ao 'max_total_num_tokens=[0-9]*.*available_gpu_mem=[0-9.]* GB' | tail -1
    strings "$f" | grep -ao 'Allocating .*hierarchical KV host pool: [0-9]* tokens, [0-9.]* GB[^.]*' | sed -E 's/^/  /' | sort | uniq -c
    strings "$f" | grep -ao 'Allocating [0-9.]* GB host memory for DSA indexer[^)]*)' | sort | uniq -c
  done
  hs=$PREFILL_HICACHE_SIZE
  if (( hs > 0 )); then echo "  (host pool want: ${hs}.00 GB per rank, fixed size)"
  else echo "  (host pool want: ratio $PREFILL_HICACHE_RATIO x device pool, ~154 GB per rank expected at prefill mem 0.85)"; fi
  echo "== prefill container env (docker inspect)"
  "${SSH[@]}" -n "$NODE" "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CONTAINER_PREFIX-prefill-0" |
    grep -E '^(SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB|SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER|MC_TE_FILTERS|MC_DISABLE_HIP)=' |
    sed 's/^/  /' || echo "  (none of the watched keys set)"
  echo "  (want SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=${HICACHE_HOST_REGISTER_CHUNK_GB:-<absent>})"
  echo "== ROCm debug agent (DEBUG_AGENT=$DEBUG_AGENT, report goes to stderr = the server logs)"
  if [[ "$DEBUG_AGENT" == 1 ]]; then
    for f in "$RD"/server-logs/*.log; do
      printf '  %-28s kernel_code_entry=%s  stopped,reason=%s  "Memory access fault"=%s\n' "$(basename "$f")" \
        "$(strings "$f" | grep -c 'kernel_code_entry')" "$(strings "$f" | grep -c 'stopped, reason')" \
        "$(strings "$f" | grep -c 'Memory access fault')"
      strings "$f" | grep -a -m3 'kernel_code_entry\|stopped, reason' | cut -c1-220 | sed 's/^/    /'
    done
    echo "  (want 0/0/0; any hit = a GPU fault, and the agent names the kernel next to it)"
  fi
  echo "== same-node IPC namespace: not applicable cross-node (xnode; legs on $NODE and $DNODE)"
  if [[ -n "$dlog" ]]; then
    # Eager-fallback signal: with NCCL_DEBUG=INFO, decode CUDA-graph replays are silent;
    # NCCL INFO lines after capture mean eager collectives. c44c (index_share=true):
    # ~17.8k/rank over the run; c44d/c44e (false): 0 after "fired up".
    echo "== decode eager-fallback counter: $dlog"
    strings "$dlog" | awk '
      {a[NR]=$0} /Capture .*CUDA graph end/{cap=NR} /The server is fired up/{up=NR}
      END{
        if(!cap){print "  no CUDA-graph capture end found"; exit}
        for(i=cap+1;i<=NR;i++) if(a[i]~/NCCL INFO/){n++; if(up && i>up){m++; if(match(a[i],/\[[0-9]+\] NCCL INFO/)){r=substr(a[i],RSTART+1,RLENGTH-12); pr[r]++}}}
        printf "  NCCL INFO after last capture end : %d\n", n
        printf "  NCCL INFO after fired-up         : %d  (want 0)", m; for(r in pr) printf "  [%s]=%d", r, pr[r]; print ""
      }'
  fi
  ;;
audit)
  python3 "$HERE/tools/audit_live.v3x.yihou.py" --config "$CONFIG" --node "$NODE" --decode-node "$DNODE" \
    --prefill "http://$IP:$P_PORT" --decode "http://$DIP:$D_PORT" \
    --prefix "$CONTAINER_PREFIX" --out "$RD/audit" ;;
smoke)
  "${SSH[@]}" -n "$NODE" "curl -s -m 300 http://$IP:$ROUTER_PORT/v1/completions -H 'Content-Type: application/json' \
    -d '{\"model\":\"$SERVED_MODEL\",\"prompt\":\"List the first five prime numbers:\",\"max_tokens\":40,\"temperature\":0}'" \
    | tee "$RD/smoke-response.json"; echo
  grep -q '"text"' "$RD/smoke-response.json" && mark smoke ok || { mark smoke FAIL; exit 1; }
  echo "NOTE: simulated acceptance is ON -> the text is expected to be garbled; this only proves KV moved."
  ;;
bench)
  out="$RD/agentx-c$CONC${BENCH_TAG:+-$BENCH_TAG}"   # BENCH_TAG: re-run on a live stack
  mark bench "start CONC=$CONC DURATION=$AGENTX_DURATION warmup=$AGENTX_WARMUP_REQUESTS_PER_LANE out=$out"
  (cd "$BENCH_DIR" && ./agentx_bench.sh "${COMMON[@]}" "CONC=$CONC" "OUT_DIR=$out") > "$out.log" 2>&1
  rc=$?
  "${SSH[@]}" -n "$NODE" "curl -s -m 30 http://$DIP:$D_PORT/metrics" 2>/dev/null |
    grep '^sglang:spec_accept' > "$RD/spec_accept.after-bench.txt" || true
  if [[ $rc == 0 ]]; then mark bench "ok rc=0 $(tail -1 "$out.log")"; else mark bench "FAIL rc=$rc (see $out.log)"; exit $rc; fi
  ;;
stop)
  (cd "$BENCH_DIR" && ./stop.sh "${COMMON[@]}" "CONTAINER_STOP_TIMEOUT=${CONTAINER_STOP_TIMEOUT:-300}")
  src=$?
  sleep 5   # let the node-side `docker logs -f` writers flush and exit
  sync_back && [[ -n "$NODE_RW_ROOT" ]] && echo "node run dir $NODE_RD copied to $RD"
  exit $src
  ;;
*) sed -n '2,20p' "$0"; exit 2 ;;
esac
