#!/usr/bin/env bash
# Launch ONE DSv4.1-Flash vLLM server (plus its UMBP standalone server and vllm-router when the config says so) in ONE
# docker container named yihou-dsv41-<tag>. Run ON the GPU node (openai smci355), as user yihou (docker group).
#
# Usage: serve.sh <config-id|config.env> <gpu-list> <tag> [options]
#   config-id   e.g. r1-dp2-c128 -> <workspace>/configs/r1-dp2-c128.env (bash-sourceable, see configs/*.env)
#   gpu-list    comma list of HOST GPU indices, e.g. 0,1 (count must equal GPUS_NEEDED). Index space: see GPU_SELECT in the
#               config; HIP index may differ from rocm-smi index on some nodes (chaox BASELINE_REPRO.md) -> preflight records both.
#               'none' for a router-only config (ENGINE_ENABLE=0, GPUS_NEEDED=0; used by serve_pd.sh).
#   tag         [a-z0-9][a-z0-9._-]{0,55}; container yihou-dsv41-<tag>; one fresh tag per scored point.
# Options:
#   --dry-run          print the fully rendered docker argv, entry script (vllm/umbp/router argv + env) and exit; no side effects
#   --port-offset N    shift every port of the config by N (several servers on one node). N>0 on DP configs also appends
#                      --data-parallel-rpc-port 29550+N (NOT a CI flag; sufficiency for port isolation unverified)
#   --aiter-jit MODE   container (default): AITER JIT in the container's own package dir (one per server, dies with the --rm
#                      container; Muse s0b2 practice). dir: AITER_JIT_DIR=<local>/<tag>/aiter_jit (node-local, per tag;
#                      whether an empty AITER_JIT_DIR forces a full rebuild of prebuilt modules is unverified).
#   --reuse            allow an existing non-empty NFS run dir for this tag (default: refuse; fresh dir per point)
#   --wait             after launch, poll the client endpoint /health until 200 (or the container dies / 5400 s timeout)
# Paths (env overrides; every one must contain /yihou/ except the read-only model):
#   DSV41_NFS_ROOT   (/perf_apps/yihou/dsv41_vllm_pd_20261119)  logs -> $DSV41_NFS_ROOT/runs/<tag>/
#   DSV41_LOCAL_ROOT (/data/yihou/dsv41_vllm_pd_20261119)       container /root,/tmp,caches -> $DSV41_LOCAL_ROOT/<tag>/
#   DSV41_MODEL      (/perf_apps/chaox/models/DeepSeek-V4.1-Flash, mounted read-only)
#   DSV41_NODE_IP    (default: first address of `hostname -I`)
# PD (E7, 11-19): role configs rendered by serve_pd.sh use the same format plus ENGINE_ENABLE (0 = router-only container,
# no GPU devices), ROUTER_PREWAIT_URLS / ROUTER_PREWAIT_BOOTSTRAP (router starts only after P/D /health and the Mooncake
# bootstrap /query list every prefill DP rank) and DOCKER_IONIC_PROVIDER (mount the host ionic verbs provider, IX #5090).
set -euo pipefail

KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WS=$(cd "$KIT_DIR/../.." && pwd)
PROJ=dsv41_vllm_pd_20261119
NFS_ROOT=${DSV41_NFS_ROOT:-/perf_apps/yihou/$PROJ}
LOCAL_ROOT=${DSV41_LOCAL_ROOT:-/data/yihou/$PROJ}
MODEL_HOST=${DSV41_MODEL:-/perf_apps/chaox/models/DeepSeek-V4.1-Flash}
MODEL_CTR=/models/DeepSeek-V4.1-Flash

die() { echo "serve.sh: ERROR: $*" >&2; exit 2; }
[ $# -ge 3 ] || die "usage: serve.sh <config-id|config.env> <gpu-list> <tag> [--dry-run] [--port-offset N] [--aiter-jit container|dir] [--reuse] [--wait]"
CFG_ARG=$1; GPU_LIST=$2; TAG=$3; shift 3
DRY=0; OFF=0; AITER_JIT=container; REUSE=0; WAIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --port-offset) OFF=${2:?}; shift ;;
    --aiter-jit) AITER_JIT=${2:?}; shift ;;
    --reuse) REUSE=1 ;;
    --wait) WAIT=1 ;;
    *) die "unknown option $1" ;;
  esac
  shift
done

# ---- guards -------------------------------------------------------------------------------------------------------
[[ "$TAG" =~ ^[a-z0-9][a-z0-9._-]{0,55}$ ]] || die "unsafe tag '$TAG'"
[[ "$GPU_LIST" =~ ^(none|[0-9]+(,[0-9]+)*)$ ]] || die "bad gpu-list '$GPU_LIST'"
[[ "$OFF" =~ ^[0-9]{1,4}$ ]] || die "bad --port-offset '$OFF'"
case "$AITER_JIT" in container|dir) ;; *) die "bad --aiter-jit '$AITER_JIT'" ;; esac
for p in "$NFS_ROOT" "$LOCAL_ROOT"; do
  case "$p" in */yihou/*) ;; *) die "path '$p' does not contain /yihou/" ;; esac
done
if [ -f "$CFG_ARG" ]; then CFG_FILE=$CFG_ARG; else CFG_FILE=$WS/configs/$CFG_ARG.env; fi
[ -f "$CFG_FILE" ] || die "config not found: $CFG_FILE"

# ---- load config --------------------------------------------------------------------------------------------------
ROUTER_ENABLE=0; UMBP_ENABLE=0; DOCKER_RDMA=0; IPC=host; DP_SIZE=1
VLLM_PORT=20000; NIXL_PORT=5400; KV_EVENT_PORT=3000; ROUTER_PORT=0; ROUTER_PROM_PORT=0
VLLM_ENV=(); VLLM_CMD=(); ROUTER_CMD=(); UMBP_ENV=(); UMBP_CMD=(); UMBP_PREAMBLE=""
ENGINE_ENABLE=1; DOCKER_IONIC_PROVIDER=0; ROUTER_PREWAIT_URLS=(); ROUTER_PREWAIT_BOOTSTRAP=(); ROUTER_PREWAIT_TIMEOUT_S=5400
# shellcheck disable=SC1090
source "$CFG_FILE"
[ "${PD_CONFIG:-0}" = 1 ] && die "$CFG_FILE is a PD config (both roles): launch it with serve_pd.sh"
for v in CFG_ID CFG_SOURCE IMAGE GPUS_NEEDED GPU_SELECT PORT CONCURRENCY BENCH_TP; do
  [ -n "${!v:-}" ] || die "config $CFG_FILE lacks $v"
done
if [ "$ENGINE_ENABLE" = 1 ]; then
  [ ${#VLLM_CMD[@]} -gt 0 ] || die "config lacks VLLM_CMD"
  [ "$GPU_LIST" != none ] || die "gpu-list 'none' needs a router-only config (ENGINE_ENABLE=0)"
else
  [ "$ROUTER_ENABLE" = 1 ] || die "ENGINE_ENABLE=0 needs ROUTER_ENABLE=1 (router-only container)"
  [ "$GPU_LIST" = none ] && [ "$GPUS_NEEDED" = 0 ] || die "router-only config takes gpu-list 'none' and GPUS_NEEDED=0"
fi
GPUS=()
[ "$GPU_LIST" = none ] || IFS=, read -r -a GPUS <<< "$GPU_LIST"
[ "${#GPUS[@]}" -eq "$GPUS_NEEDED" ] || die "config $CFG_ID needs $GPUS_NEEDED GPUs, got ${#GPUS[@]} ($GPU_LIST)"
[ "${#GPUS[@]}" -eq 0 ] || [ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -eq "${#GPUS[@]}" ] || die "duplicate GPU in $GPU_LIST"
for g in "${GPUS[@]}"; do [ "$g" -le 7 ] || die "GPU index $g > 7"; done

NAME=yihou-dsv41-$TAG
RUN=$NFS_ROOT/runs/$TAG
LOC=$LOCAL_ROOT/$TAG
NODE_IP=${DSV41_NODE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
[ -n "$NODE_IP" ] || die "cannot determine node IP (set DSV41_NODE_IP)"

P_PORT=$((PORT + OFF)); P_VLLM_PORT=$((VLLM_PORT + OFF)); P_NIXL=$((NIXL_PORT + OFF)); P_KVEV=$((KV_EVENT_PORT + OFF))
P_RPORT=$((ROUTER_PORT + OFF)); P_RPROM=$((ROUTER_PROM_PORT + OFF))
if [ "$GPU_SELECT" = device-ids ]; then DEVICE_IDS=$GPU_LIST; else DEVICE_IDS=""; fi

subst() {  # substitute @TOKENS@ in one word
  local s=$1
  s=${s//@MODEL@/$MODEL_CTR}; s=${s//@PORT@/$P_PORT}; s=${s//@DEVICE_IDS@/$DEVICE_IDS}; s=${s//@NODE_IP@/$NODE_IP}
  s=${s//@VLLM_PORT@/$P_VLLM_PORT}; s=${s//@NIXL_PORT@/$P_NIXL}; s=${s//@KV_EVENT_PORT@/$P_KVEV}
  s=${s//@ROUTER_PORT@/$P_RPORT}; s=${s//@ROUTER_PROM_PORT@/$P_RPROM}
  printf '%s' "$s"
}
shq() {  # readable shell quoting: bare if safe, else single quotes
  local w
  for w in "$@"; do
    if [[ "$w" =~ ^[A-Za-z0-9_./:=,@%+-]+$ ]]; then printf '%s ' "$w"
    else printf "'%s' " "${w//\'/\'\\\'\'}"; fi
  done
}
render_words() {  # render_words <word>... -> one shell-quoted line with @TOKENS@ substituted
  local w out=()
  for w in "$@"; do out+=("$(subst "$w")"); done
  shq "${out[@]}"
}

V_CMD=("${VLLM_CMD[@]}")
DEVIATIONS=()
if [ "$ENGINE_ENABLE" = 1 ] && [ "$OFF" -gt 0 ] && [ "$DP_SIZE" -gt 1 ]; then
  V_CMD+=(--data-parallel-rpc-port "$((29550 + OFF))")
  DEVIATIONS+=("--port-offset $OFF: appended --data-parallel-rpc-port $((29550 + OFF)) (not in CI argv)")
fi
if [ "$ENGINE_ENABLE" = 1 ] && [ "$GPU_SELECT" = device-ids ]; then
  printf '%s\n' "${V_CMD[@]}" | grep -qx -- '--device-ids' || die "GPU_SELECT=device-ids but VLLM_CMD has no --device-ids"
fi
ENDPOINT_PORT=$P_PORT; [ "$ROUTER_ENABLE" = 1 ] && ENDPOINT_PORT=$P_RPORT

# ---- entry script (runs inside the container) -----------------------------------------------------------------------
render_entry() {
  cat <<EOS
#!/usr/bin/env bash
# Rendered by scripts/kit/serve.sh $(date -u +%FT%TZ) -- config $CFG_ID ($CFG_FILE), tag $TAG, gpus $GPU_LIST, offset $OFF.
# Do not edit; re-render. Runs inside container $NAME. Stops children by PID only (no pkill).
set -u
RUN=/yihou/run
LOGD=\$RUN
if ! ( : > "\$RUN/.write_test" ) 2>/dev/null; then
  LOGD=/yihou/local/logs; mkdir -p "\$LOGD"
  echo "WARN: \$RUN not writable from the container (NFS root squash?); logs go to node-local $LOC/logs" >&2
fi
say() { echo "\$(date -u +%FT%TZ) [entry] \$*" | tee -a "\$LOGD/entry.log"; }
PIDS=()
stop_children() {
  local p
  say "stopping children: \${PIDS[*]:-none}"
  for p in "\${PIDS[@]}"; do kill -TERM "\$p" 2>/dev/null; done
  for _ in \$(seq 1 30); do
    local alive=0
    for p in "\${PIDS[@]}"; do kill -0 "\$p" 2>/dev/null && alive=1; done
    [ \$alive = 0 ] && return 0
    sleep 2
  done
  for p in "\${PIDS[@]}"; do kill -KILL "\$p" 2>/dev/null; done
}
trap 'say "signal received"; stop_children; exit 143' TERM INT
say "start host=\$(hostname) image=$IMAGE"
{ python3 -c 'import vllm; print("vllm", vllm.__version__)'; pip list 2>/dev/null | grep -i -E 'vllm|aiter|nixl|mooncake|mori|umbp'; } > "\$LOGD/versions.txt" 2>&1
EOS
  if [ "${MC_STOPPED_FIX:-0}" = 1 ]; then
    cat <<'EOS'
# ---- MooncakeConnector STOPPED-prefill fix (llying-001/vllm#3; vLLM main 7a62ca0 still has the bug) ----
# A DSv4.1 prefill leg may end FINISHED_STOPPED on its single sampled token; upstream only sends KV on
# FINISHED_LENGTH_CAPPED, so the decode pull waits until VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT. Patch in-container only.
python3 -I - >> "$LOGD/entry.log" 2>&1 <<'PYEOF' || { say "MC_STOPPED_FIX failed; refusing to start"; exit 12; }
import importlib.util, sys
spec = importlib.util.find_spec("vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector")
path = spec.origin
src = open(path).read()
old = "if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:"
new = ("if request.status not in (RequestStatus.FINISHED_LENGTH_CAPPED, "
       "RequestStatus.FINISHED_STOPPED):  # yihou MC_STOPPED_FIX")
if "yihou MC_STOPPED_FIX" in src:
    print("MC_STOPPED_FIX already applied:", path); sys.exit(0)
n = src.count(old)
if n != 1:
    print(f"MC_STOPPED_FIX: expected 1 match, found {n} in {path}"); sys.exit(1)
open(path, "w").write(src.replace(old, new))
print("MC_STOPPED_FIX applied:", path)
PYEOF
say "MC_STOPPED_FIX: $(grep -c 'yihou MC_STOPPED_FIX' $(python3 -c 'import importlib.util as u;print(u.find_spec("vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector").origin)') ) patched line(s)"
EOS
  fi
  if [ "$UMBP_ENABLE" = 1 ]; then
    cat <<EOS
# ---- UMBP standalone server (CI service 'umbp-server', start: before_workers) ----
${UMBP_PREAMBLE}
env $(render_words "${UMBP_ENV[@]}") $(render_words "${UMBP_CMD[@]}") > "\$LOGD/umbp.log" 2>&1 &
UMBP_PID=\$!; PIDS+=(\$UMBP_PID)
say "umbp pid \$UMBP_PID; waiting for readiness pattern (timeout ${UMBP_READY_TIMEOUT_S}s)"
t0=\$(date +%s)
until grep -q -E $(shq "$UMBP_READY_PATTERN") "\$LOGD/umbp.log" 2>/dev/null; do
  kill -0 \$UMBP_PID 2>/dev/null || { say "umbp server exited before ready"; tail -n 30 "\$LOGD/umbp.log"; exit 10; }
  [ \$((\$(date +%s) - t0)) -gt ${UMBP_READY_TIMEOUT_S} ] && { say "umbp readiness timeout"; stop_children; exit 11; }
  sleep 5
done
say "umbp ready after \$((\$(date +%s) - t0)) s"
EOS
  fi
  if [ "$ROUTER_ENABLE" = 1 ]; then
    cat <<EOS
# ---- vllm-router install (CI setup_script vllm-router.sh: pip install vllm-router==\$ROUTER_VERSION) ----
if [ "\$(pip show vllm-router 2>/dev/null | awk '/^Version:/{print \$2}')" != "$ROUTER_VERSION" ]; then
  PIPI=(python3 -m pip install)
  python3 -m pip install --help 2>/dev/null | grep -q -- --break-system-packages && PIPI+=(--break-system-packages)
  { ls /yihou/wheels/vllm_router-$ROUTER_VERSION-*.whl >/dev/null 2>&1 \\
      && "\${PIPI[@]}" --no-index --find-links /yihou/wheels "vllm-router==$ROUTER_VERSION"; } > "\$LOGD/router_install.log" 2>&1 \\
    || "\${PIPI[@]}" "vllm-router==$ROUTER_VERSION" >> "\$LOGD/router_install.log" 2>&1 \\
    || { say "vllm-router install failed (offline wheel and PyPI)"; stop_children; exit 12; }
fi
say "vllm-router \$(pip show vllm-router 2>/dev/null | awk '/^Version:/{print \$2}')"
EOS
  fi
  if [ "$ENGINE_ENABLE" = 1 ]; then
    cat <<EOS
# ---- vLLM engine (CI role 'agg')${PD_ROLE:+; PD role $PD_ROLE} ----
env $(render_words "${VLLM_ENV[@]}") $(render_words "${V_CMD[@]}") > "\$LOGD/server.log" 2>&1 &
VLLM_PID=\$!; PIDS+=(\$VLLM_PID); say "vllm pid \$VLLM_PID"
EOS
  fi
  if [ "$ROUTER_ENABLE" = 1 ] && { [ ${#ROUTER_PREWAIT_URLS[@]} -gt 0 ] || [ ${#ROUTER_PREWAIT_BOOTSTRAP[@]} -gt 0 ]; }; then
    # PD: vllm-router queries each Mooncake bootstrap server ONCE at startup and caches the engine ids; an answer taken
    # before the prefill ranks registered has no engine ids and decode then recomputes every prompt (NVIDIA/srt-slurm#669
    # docs; router src/routers/http/vllm_pd_router.rs: query_mooncake_bootstrap). So wait here for /health of every
    # worker AND for /query to list every prefill DP rank. python3 stdlib only (curl is not guaranteed in the image).
    cat <<EOS
# ---- PD router pre-wait (P/D /health 200 + Mooncake bootstrap /query lists every prefill DP rank) ----
say "router pre-wait: health $(shq "${ROUTER_PREWAIT_URLS[@]}") bootstrap $(shq "${ROUTER_PREWAIT_BOOTSTRAP[@]}") (timeout ${ROUTER_PREWAIT_TIMEOUT_S}s)"
python3 -I - $(shq "${ROUTER_PREWAIT_TIMEOUT_S}") $(render_words "${ROUTER_PREWAIT_URLS[@]}") -- $(render_words "${ROUTER_PREWAIT_BOOTSTRAP[@]}") >> "\$LOGD/router_prewait.log" 2>&1 <<'PYEOF'
import json, sys, time, urllib.request
args = sys.argv[1:]; timeout = float(args[0]); sep = args.index("--")
health, boots = args[1:sep], args[sep + 1:]
t0 = time.time(); pending = {u: "health" for u in health}
pending.update({b.rsplit("=", 1)[0]: int(b.rsplit("=", 1)[1]) for b in boots})
while pending:
    for url, want in list(pending.items()):
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                body = r.read()
                if want == "health":
                    ok = r.status == 200
                else:
                    ids = json.loads(body or b"{}"); ok = len(ids) >= want
                    print(time.strftime("%FT%TZ", time.gmtime()), url, "ranks", sorted(ids), flush=True)
        except Exception as e:
            ok = False
        if ok:
            print(time.strftime("%FT%TZ", time.gmtime()), "ready", url, flush=True); pending.pop(url)
    if pending and time.time() - t0 > timeout:
        print("TIMEOUT waiting for", sorted(pending), flush=True); sys.exit(1)
    if pending:
        time.sleep(10)
print("pre-wait done after %.0f s" % (time.time() - t0), flush=True)
PYEOF
rc=\$?
[ \$rc = 0 ] || { say "router pre-wait failed rc=\$rc (see router_prewait.log)"; stop_children; exit 13; }
say "router pre-wait done"
EOS
  fi
  if [ "$ROUTER_ENABLE" = 1 ]; then
    cat <<EOS
# ---- vllm-router frontend (CI role 'vllm-router'; waits for workers itself, --worker-startup-timeout-secs) ----
$(render_words "${ROUTER_CMD[@]}") > "\$LOGD/router.log" 2>&1 &
ROUTER_PID=\$!; PIDS+=(\$ROUTER_PID); say "router pid \$ROUTER_PID"
EOS
  fi
  cat <<'EOS'
# ---- supervise: any child exit ends the container ----
while :; do
  for p in "${PIDS[@]}"; do
    if ! kill -0 "$p" 2>/dev/null; then
      wait "$p"; rc=$?
      say "child $p exited rc=$rc; stopping the rest"
      stop_children; exit "$rc"
    fi
  done
  sleep 10
done
EOS
}

# ---- docker argv --------------------------------------------------------------------------------------------------
DOCKER=(docker run -d --rm --name "$NAME" --network host --ipc "$IPC")
[ "$ENGINE_ENABLE" = 1 ] && DOCKER+=(--device /dev/kfd --device /dev/dri --group-add video)
DOCKER+=(--cap-add SYS_PTRACE --security-opt seccomp=unconfined --ulimit core=0 --ulimit nofile=1048576:1048576
  --label "yihou.dsv41.tag=$TAG" --label "yihou.dsv41.config=$CFG_ID" --label "yihou.dsv41.gpus=$GPU_LIST"
  -v "$MODEL_HOST:$MODEL_CTR:ro"
  -v "$LOC:/yihou/local" -v "$LOC/root:/root" -v "$LOC/tmp:/tmp"
  -v "$RUN:/yihou/run" -v "$RUN/kit:/yihou/kit:ro" -v "$NFS_ROOT/wheels:/yihou/wheels:ro")
[ "$UMBP_ENABLE" = 1 ] && DOCKER+=(-v "$LOC/umbp-host:/umbp-host")
if [ "$DOCKER_RDMA" = 1 ]; then
  DOCKER+=(--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1)
fi
# IX #5090 (MiniMax vLLM Mooncake PD on MI355X): "The image has no Ionic verbs provider; libibverbs loads the host's only
# through its driver entry." Whether OUR 60bd image lacks it is unverified (PD_README step 1 checks ibv_devices in-image).
IONIC_SO=/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so; IONIC_DRV=/etc/libibverbs.d/ionic.driver
if [ "$DOCKER_IONIC_PROVIDER" = 1 ]; then
  DOCKER+=(-v "$IONIC_SO:$IONIC_SO:ro" -v "$IONIC_DRV:$IONIC_DRV:ro")
fi
[ "$ENGINE_ENABLE" = 1 ] && [ "$GPU_SELECT" = hip-visible ] && DOCKER+=(-e "HIP_VISIBLE_DEVICES=$GPU_LIST")
DOCKER+=(-e HF_HOME=/yihou/local/cache/hf -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/yihou/local/cache/vllm
  -e XDG_CACHE_HOME=/yihou/local/cache/xdg -e TRITON_CACHE_DIR=/yihou/local/cache/triton
  -e TORCHINDUCTOR_CACHE_DIR=/yihou/local/cache/inductor -e PIP_CACHE_DIR=/yihou/local/cache/pip
  -e TMPDIR=/tmp -e PYTHONNOUSERSITE=1
  -e "VLLM_ENGINE_READY_TIMEOUT_S=${KIT_READY_TIMEOUT_S:-3600}")   # setup only: default 600 s is too short for 4 servers loading from NFS (e1 11-19)
[ "$AITER_JIT" = dir ] && DOCKER+=(-e AITER_JIT_DIR=/yihou/local/aiter_jit)
DOCKER+=(--entrypoint bash "$IMAGE" /yihou/kit/entry.sh)

print_plan() {
  echo "### serve.sh render: config=$CFG_ID tag=$TAG gpus=$GPU_LIST port-offset=$OFF aiter-jit=$AITER_JIT"
  echo "# config file : $CFG_FILE"
  echo "# source      : $CFG_SOURCE"
  echo "# container   : $NAME   image: $IMAGE"
  echo "# endpoint    : http://$NODE_IP:$ENDPOINT_PORT  (engine /metrics: http://$NODE_IP:$P_PORT/metrics)"
  echo "# harness     : concurrency=$CONCURRENCY TP(env, GPU denominator)=$BENCH_TP"
  echo "# logs (NFS)  : $RUN/{entry,server,umbp,router,router_install}.log versions.txt"
  echo "# node-local  : $LOC/{root,tmp,cache,aiter_jit,umbp-host,logs}"
  [ "$UMBP_ENABLE" = 1 ] && echo "# host DRAM   : UMBP pool ${UMBP_HOST_DRAM_GB:-?} GB (check preflight host_dram_free)"
  for d in "${DEVIATIONS[@]}"; do echo "# DEVIATION   : $d"; done
  echo
  echo "## docker argv"
  shq "${DOCKER[@]}"; echo
  echo
  echo "## entry.sh"
  render_entry
}

if [ "$DRY" = 1 ]; then print_plan; exit 0; fi

# ---- real launch --------------------------------------------------------------------------------------------------
command -v docker >/dev/null || die "docker not found"
if [ "$DOCKER_IONIC_PROVIDER" = 1 ]; then
  for f in "$IONIC_SO" "$IONIC_DRV"; do [ -f "$f" ] || die "DOCKER_IONIC_PROVIDER=1 but host file $f is missing"; done
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then die "container $NAME already exists (stop.sh $TAG)"; fi
# GPU overlap with our own running containers (labels); foreign containers are reported by preflight.sh only.
while read -r cname cgpus; do
  [ -n "$cname" ] || continue
  for g in "${GPUS[@]}"; do
    case ",$cgpus," in *",$g,"*) die "GPU $g already used by our container $cname (gpus $cgpus)";; esac
  done
done < <(docker ps --filter label=yihou.dsv41.gpus --format '{{.Names}} {{.Label "yihou.dsv41.gpus"}}')
for port in $([ "$ENGINE_ENABLE" = 1 ] && echo "$P_PORT") $([ "$ROUTER_ENABLE" = 1 ] && echo "$P_RPORT $P_RPROM"); do
  if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$port\$"; then die "port $port already in use"; fi
done
if [ -d "$RUN" ] && [ -n "$(ls -A "$RUN" 2>/dev/null)" ] && [ "$REUSE" = 0 ]; then
  die "run dir $RUN not empty (fresh tag per point, or --reuse)"
fi
mkdir -p "$RUN" "$NFS_ROOT/wheels" "$LOC/root" "$LOC/tmp" "$LOC/cache" "$LOC/aiter_jit" "$LOC/logs"
[ "$UMBP_ENABLE" = 1 ] && mkdir -p "$LOC/umbp-host"
# SECURITY (review 11-19): /perf_apps is a shared 0777 NFS. Container root may be root-squashed, so the run/log dirs must
# stay writable for others -> sticky 1777 (others cannot rename/replace our entries). Everything that is EXECUTED or
# SOURCED lives in $RUN/kit: owner-only writable dir, files 0644, mounted read-only into the container.
chmod 1777 "$RUN" "$LOC" "$LOC/logs"
mkdir -p "$RUN/kit"; chmod 0755 "$RUN/kit"
cp "$CFG_FILE" "$RUN/kit/config.env"
# REVIEW fix: record the EFFECTIVE (port-offset) values; run_point.sh/watchdog.sh source this file, and without these
# lines a --port-offset server's run_point would scrape the un-offset port (= another server's /metrics).
{ echo; echo "# ---- appended by serve.sh: effective values for this tag (port-offset $OFF) ----"
  echo "PORT=$P_PORT"; echo "ROUTER_PORT=$P_RPORT"; echo "ROUTER_PROM_PORT=$P_RPROM"
  echo "VLLM_PORT=$P_VLLM_PORT"; echo "NIXL_PORT=$P_NIXL"; echo "KV_EVENT_PORT=$P_KVEV"
  echo "KIT_PORT_OFFSET=$OFF"; echo "KIT_GPU_LIST=$GPU_LIST"; echo "KIT_NODE_IP=$NODE_IP"; } >> "$RUN/kit/config.env"
render_entry > "$RUN/kit/entry.sh"
chmod 0644 "$RUN/kit/config.env" "$RUN/kit/entry.sh"
print_plan > "$RUN/render.txt"
shq "${DOCKER[@]}" > "$RUN/docker_cmd.txt"; echo >> "$RUN/docker_cmd.txt"
"${DOCKER[@]}" > "$RUN/container_id.txt"
echo "serve.sh: started $NAME ($(cut -c1-12 "$RUN/container_id.txt")); logs $RUN"

if [ "$WAIT" = 1 ]; then
  t0=$(date +%s)
  until [ "$(curl -s -o /dev/null -m 5 -w '%{http_code}' "http://127.0.0.1:$ENDPOINT_PORT/health")" = 200 ]; do
    st=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo gone)
    [ "$st" = running ] || die "container $NAME is '$st' before ready; see $RUN/entry.log"
    [ $(( $(date +%s) - t0 )) -lt 5400 ] || die "not ready after 5400 s (container left running; stop.sh $TAG)"
    sleep 15
  done
  echo "serve.sh: $NAME ready after $(( $(date +%s) - t0 )) s at http://$NODE_IP:$ENDPOINT_PORT"
fi
