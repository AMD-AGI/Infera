#!/usr/bin/env bash
# Launch a 1P1D PD deployment (E7) from ONE PD config (configs/e7-pd-*.env, PD_CONFIG=1) as THREE containers, each via
# serve.sh (so every serve.sh fix applies: VLLM_ENGINE_READY_TIMEOUT_S, --ulimit nofile, kit/ ro mount, fresh run dirs,
# owner-only kit files, GPU/port guards):
#   yihou-dsv41-<tag>-p   prefill engine (kv_producer)         runs/<tag>-p/
#   yihou-dsv41-<tag>-d   decode engine  (kv_consumer)         runs/<tag>-d/
#   yihou-dsv41-<tag>     vllm-router PD mode, no GPU devices  runs/<tag>/   <- run_point.sh / pd_smoke.sh use THIS tag
# Run ON the node (same-node E7) or once per node with --only (cross-node). Role configs are rendered (all PD tokens
# resolved, port offset applied) to $DSV41_NFS_ROOT/pd_roles/<tag>/{p,d,r}.env (dry-run: <ws>/configs/renders/pd_roles/<tag>/)
# and handed to serve.sh with --port-offset 0.
#
# Usage: serve_pd.sh <pd-config-id|file> <p-gpus|none> <d-gpus|none> <tag> [options]
#   p-gpus/d-gpus  HOST GPU indices (HIP index space as serve.sh), e.g. 0,1 and 2,3; 'none' for a role not launched here
#   tag            [a-z0-9][a-z0-9._-]{0,52} (room for the -p/-d suffix)
# Options:
#   --dry-run            render role configs + serve.sh dry-runs of every selected role, no side effects outside <ws>
#   --port-offset N      add N to every port of the PD config (P/D engine, bootstrap, DP rpc, VLLM/NIXL/KV-event, router)
#   --only LIST          comma list of roles to launch here: p,d,router (default p,d,router = same node)
#   --p-host IP          host of the prefill engine as seen by the router and the decode engine (default: this node)
#   --d-host IP          host of the decode engine (default: this node)
#   --aiter-jit MODE     passed to serve.sh for the engine roles
#   --reuse              allow existing role files / run dirs (serve.sh --reuse)
#   --wait               serve.sh --wait on the LAST role launched here (router: its /health is 200 only after P and D
#                        are healthy and the router has read the bootstrap engine ids)
# Stop: stop.sh <tag> --prefix  (stops <tag>, <tag>-p, <tag>-d and <tag>-client; never other names).
set -euo pipefail

KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WS=$(cd "$KIT_DIR/../.." && pwd)
PROJ=dsv41_vllm_pd_20261119
NFS_ROOT=${DSV41_NFS_ROOT:-/perf_apps/yihou/$PROJ}

die() { echo "serve_pd.sh: ERROR: $*" >&2; exit 2; }
[ $# -ge 4 ] || die "usage: serve_pd.sh <pd-config> <p-gpus|none> <d-gpus|none> <tag> [--dry-run] [--port-offset N] [--only p,d,router] [--p-host IP] [--d-host IP] [--aiter-jit M] [--reuse] [--wait]"
CFG_ARG=$1; PG=$2; DG=$3; TAG=$4; shift 4
DRY=0; OFF=0; ONLY=p,d,router; PHOST=""; DHOST=""; AJ=container; REUSE=0; WAIT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --port-offset) OFF=${2:?}; shift ;;
    --only) ONLY=${2:?}; shift ;;
    --p-host) PHOST=${2:?}; shift ;;
    --d-host) DHOST=${2:?}; shift ;;
    --aiter-jit) AJ=${2:?}; shift ;;
    --reuse) REUSE=1 ;;
    --wait) WAIT=1 ;;
    *) die "unknown option $1" ;;
  esac
  shift
done
[[ "$TAG" =~ ^[a-z0-9][a-z0-9._-]{0,52}$ ]] || die "unsafe tag '$TAG' (max 53 chars)"
[[ "$OFF" =~ ^[0-9]{1,4}$ ]] || die "bad --port-offset '$OFF'"
[[ "$ONLY" =~ ^(p|d|router)(,(p|d|router))*$ ]] || die "bad --only '$ONLY'"
for g in "$PG" "$DG"; do [[ "$g" =~ ^(none|[0-9](,[0-9])*)$ ]] || die "bad gpu list '$g'"; done
for h in "$PHOST" "$DHOST"; do [ -z "$h" ] || [[ "$h" =~ ^[A-Za-z0-9._-]+$ ]] || die "bad host '$h'"; done
case "$NFS_ROOT" in */yihou/*) ;; *) die "path '$NFS_ROOT' does not contain /yihou/" ;; esac
has_role() { case ",$ONLY," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

if [ -f "$CFG_ARG" ]; then CFG_FILE=$CFG_ARG; else CFG_FILE=$WS/configs/$CFG_ARG.env; fi
[ -f "$CFG_FILE" ] || die "config not found: $CFG_FILE"
# shellcheck disable=SC1090
source "$CFG_FILE"
[ "${PD_CONFIG:-0}" = 1 ] || die "$CFG_FILE is not a PD config (PD_CONFIG=1); use serve.sh"
for v in CFG_ID IMAGE P_GPUS_NEEDED D_GPUS_NEEDED P_PORT D_PORT P_BOOTSTRAP_PORT ROUTER_PORT ROUTER_PROM_PORT \
         P_KV_TRANSFER_CONFIG D_KV_TRANSFER_CONFIG SPEC_CONFIG ROUTER_VERSION CONCURRENCY BENCH_TP DP_SIZE; do
  [ -n "${!v:-}" ] || die "PD config lacks $v"
done

# P and D must share every flag that shapes the KV-cache groups / bytes (g3 H1/H2/H5, g1 verdict 3: Mooncake does not
# verify MLA layout identity). They live in COMMON_CMD; refuse role extras that would make them differ.
for w in ${P_CMD_EXTRA[@]+"${P_CMD_EXTRA[@]}"} ${D_CMD_EXTRA[@]+"${D_CMD_EXTRA[@]}"}; do
  case "$w" in
    --speculative-config*|--attention-config*|--block-size*|--kv-cache-dtype*|--swa-bounded-replay*|--no-swa-bounded-replay*|\
    --disable-hybrid-kv-cache-manager*|--tensor-parallel-size*|--hf-overrides*|--quantization*|--max-model-len*)
      die "role extra '$w' must be identical on P and D: put it in COMMON_CMD" ;;
  esac
done

NODE_IP=${DSV41_NODE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}
[ -n "$NODE_IP" ] || die "cannot determine node IP (set DSV41_NODE_IP)"
PHOST=${PHOST:-$NODE_IP}; DHOST=${DHOST:-$NODE_IP}
CROSS=0; [ "$PHOST" = "$DHOST" ] || CROSS=1

# ---- GPU lists --------------------------------------------------------------------------------------------------------
count() { [ "$1" = none ] && echo 0 || awk -F, '{print NF}' <<< "$1"; }
has_role p && { [ "$(count "$PG")" = "$P_GPUS_NEEDED" ] || die "prefill needs $P_GPUS_NEEDED GPUs, got '$PG'"; }
has_role d && { [ "$(count "$DG")" = "$D_GPUS_NEEDED" ] || die "decode needs $D_GPUS_NEEDED GPUs, got '$DG'"; }
if has_role p && has_role d; then
  [ "$CROSS" = 0 ] || die "--only p,d launches both engines HERE, so --p-host must equal --d-host"
  for g in ${PG//,/ }; do case ",$DG," in *",$g,"*) die "GPU $g is in both the P and the D list";; esac; done
fi

# ---- ports (offset applied once, here) --------------------------------------------------------------------------------
o() { echo $(( $1 + OFF )); }
PP=$(o "$P_PORT"); DP=$(o "$D_PORT"); BP=$(o "$P_BOOTSTRAP_PORT"); RP=$(o "$ROUTER_PORT"); RPP=$(o "$ROUTER_PROM_PORT")

pdsubst() {  # resolve PD-level tokens in one word; $1 role (p|d|r), $2 word
  local r=$1 s=$2 kv=""
  case "$r" in p) kv=$P_KV_TRANSFER_CONFIG ;; d) kv=$D_KV_TRANSFER_CONFIG ;; esac
  kv=${kv//@MOONCAKE_PROTOCOL@/${MOONCAKE_PROTOCOL:-rdma}}
  kv=${kv//@MOONCAKE_DEVICE_NAME@/${MOONCAKE_DEVICE_NAME:-}}
  kv=${kv//@MOONCAKE_NUM_WORKERS@/${MOONCAKE_NUM_WORKERS:-10}}
  s=${s//@KV_TRANSFER_CONFIG@/$kv}; s=${s//@SPEC_CONFIG@/$SPEC_CONFIG}
  case "$r" in
    p) s=${s//@DP_RPC_PORT@/$(o "${P_DP_RPC_PORT:-29550}")} ;;
    d) s=${s//@DP_RPC_PORT@/$(o "${D_DP_RPC_PORT:-29650}")} ;;
  esac
  s=${s//@BOOTSTRAP_PORT@/$BP}; s=${s//@P_HOST@/$PHOST}; s=${s//@D_HOST@/$DHOST}
  s=${s//@P_PORT@/$PP}; s=${s//@D_PORT@/$DP}
  printf '%s' "$s"
}
q1() { printf "'%s'" "${1//\'/\'\\\'\'}"; }   # single-quote one word for a bash file
emit_array() {  # emit_array <role> <NAME> <word>...
  local r=$1 n=$2 w; shift 2
  echo "$n=("
  for w in "$@"; do printf '  %s\n' "$(q1 "$(pdsubst "$r" "$w")")"; done
  echo ")"
}

render_role() {  # render_role <p|d|r>
  local r=$1
  echo "# shellcheck shell=bash disable=SC2034"
  echo "# Rendered by scripts/kit/serve_pd.sh $(date -u +%FT%TZ) from $CFG_FILE (role $r, tag $TAG, port-offset $OFF,"
  echo "# p-host $PHOST, d-host $DHOST). Do not edit; re-render. Launch through serve_pd.sh (serve.sh --port-offset 0)."
  echo "CFG_SOURCE=$(q1 "$CFG_ID role $r <- $CFG_SOURCE")"
  echo "IMAGE=$(q1 "$IMAGE")"
  echo "IPC=${IPC:-host}"
  echo "CONCURRENCY=$CONCURRENCY"
  echo "BENCH_TP=$BENCH_TP"
  echo "DP_SIZE=$DP_SIZE"
  echo "UMBP_ENABLE=0"
  if [ "$r" = r ]; then
    echo "CFG_ID=$CFG_ID"
    echo "PD_ROLE=router"
    echo "ENGINE_ENABLE=0"
    echo "GPUS_NEEDED=0"
    echo "GPU_SELECT=none"
    echo "DOCKER_RDMA=0"
    echo "PORT=$RP                # router-only: no engine port (serve.sh checks ROUTER_PORT/ROUTER_PROM_PORT)"
    echo "DISAGG=${DISAGG:-true}"
    echo "ROUTER_ENABLE=1"
    echo "ROUTER_VERSION=$ROUTER_VERSION"
    echo "ROUTER_PORT=$RP"
    echo "ROUTER_PROM_PORT=$RPP"
    emit_array r ROUTER_CMD "${ROUTER_CMD[@]}"
    emit_array r ROUTER_PREWAIT_URLS "http://$PHOST:$PP/health" "http://$DHOST:$DP/health"
    emit_array r ROUTER_PREWAIT_BOOTSTRAP "http://$PHOST:$BP/query=$DP_SIZE"
    echo "ROUTER_PREWAIT_TIMEOUT_S=${ROUTER_PREWAIT_TIMEOUT_S:-5400}"
    echo "# ---- PD metadata for run_point.sh / pd_smoke.sh / summarize.py ----"
    echo "METRICS_URLS=http://$PHOST:$PP/metrics,http://$DHOST:$DP/metrics"
    echo "SERVER_LOG_PATH=$NFS_ROOT/runs/$TAG-d/server.log       # decode log: KV pool tokens for the result"
    echo "PREFILL_SERVER_LOG_PATH=$NFS_ROOT/runs/$TAG-p/server.log"
    echo "PD_P_TAG=$TAG-p"
    echo "PD_D_TAG=$TAG-d"
    echo "PD_P_URL=http://$PHOST:$PP"
    echo "PD_D_URL=http://$DHOST:$DP"
    echo "PD_BOOTSTRAP_URL=http://$PHOST:$BP"
    echo "PD_CROSS_NODE=$CROSS"
  else
    local up; up=$([ "$r" = p ] && echo P || echo D)
    local gpn=${up}_GPUS_NEEDED vp=${up}_VLLM_PORT np=${up}_NIXL_PORT kp=${up}_KV_EVENT_PORT
    local -n envx=${up}_ENV_EXTRA cmdx=${up}_CMD_EXTRA
    echo "CFG_ID=$CFG_ID-$r"
    echo "PD_ROLE=$([ "$r" = p ] && echo prefill || echo decode)"
    echo "ENGINE_ENABLE=1"
    echo "GPUS_NEEDED=${!gpn}"
    echo "GPU_SELECT=${GPU_SELECT:-device-ids}"
    echo "DOCKER_RDMA=${DOCKER_RDMA:-1}"
    echo "DOCKER_IONIC_PROVIDER=${DOCKER_IONIC_PROVIDER:-0}"
    echo "PORT=$([ "$r" = p ] && echo "$PP" || echo "$DP")"
    echo "VLLM_PORT=$(o "${!vp}")"
    echo "NIXL_PORT=$(o "${!np}")"
    echo "KV_EVENT_PORT=$(o "${!kp}")"
    echo "ROUTER_ENABLE=0"
    [ "$r" = p ] && echo "PD_BOOTSTRAP_PORT=$BP"
    emit_array "$r" VLLM_ENV "${COMMON_ENV[@]}" "${envx[@]}"
    emit_array "$r" VLLM_CMD "${COMMON_CMD[@]}" "${cmdx[@]}"
  fi
}

# ---- role files ---------------------------------------------------------------------------------------------------------
if [ "$DRY" = 1 ]; then RD=$WS/configs/renders/pd_roles/$TAG; else RD=$NFS_ROOT/pd_roles/$TAG; fi
case "$RD" in */yihou/*) ;; *) die "role dir '$RD' does not contain /yihou/" ;; esac
mkdir -p "$RD"; [ "$DRY" = 1 ] || chmod 0755 "$RD"
declare -A RFILE=([p]=$RD/p.env [d]=$RD/d.env [router]=$RD/r.env)
declare -A RKEY=([p]=p [d]=d [router]=r)
for role in p d router; do
  has_role "$role" || continue
  f=${RFILE[$role]}
  if [ "$DRY" = 0 ] && [ -e "$f" ] && [ "$REUSE" = 0 ]; then die "$f exists (fresh tag per point, or --reuse)"; fi
  render_role "${RKEY[$role]}" > "$f"
  [ "$DRY" = 1 ] || chmod 0644 "$f"
  bash -n "$f" || die "rendered $f does not parse"
done

# ---- launch (or dry-run) via serve.sh --------------------------------------------------------------------------------------
SARGS=(--port-offset 0)
[ "$DRY" = 1 ] && SARGS+=(--dry-run)
[ "$REUSE" = 1 ] && SARGS+=(--reuse)
LAUNCH=(); for role in p d router; do has_role "$role" && LAUNCH+=("$role"); done
LAST=${LAUNCH[$(( ${#LAUNCH[@]} - 1 ))]}

if [ "$DRY" = 0 ] && has_role p; then
  if ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$BP\$"; then die "Mooncake bootstrap port $BP already in use"; fi
fi
echo "### serve_pd.sh: config=$CFG_ID tag=$TAG offset=$OFF only=$ONLY p=$PHOST:$PP (gpus $PG, bootstrap $BP) d=$DHOST:$DP (gpus $DG) router=:$RP cross-node=$CROSS"
echo "# role configs: $RD/{p,d,r}.env"
for role in "${LAUNCH[@]}"; do
  case "$role" in
    p) gl=$PG; t=$TAG-p; extra=(--aiter-jit "$AJ") ;;
    d) gl=$DG; t=$TAG-d; extra=(--aiter-jit "$AJ") ;;
    router) gl=none; t=$TAG; extra=() ;;
  esac
  w=(); [ "$WAIT" = 1 ] && [ "$role" = "$LAST" ] && [ "$DRY" = 0 ] && w=(--wait)
  echo
  echo "######## role $role -> container yihou-dsv41-$t"
  DSV41_NODE_IP=$NODE_IP bash "$KIT_DIR/serve.sh" "${RFILE[$role]}" "$gl" "$t" "${SARGS[@]}" "${extra[@]}" "${w[@]}"
done
[ "$DRY" = 1 ] || echo "serve_pd.sh: launched ${LAUNCH[*]} for $TAG; client endpoint http://$NODE_IP:$RP (router); stop: stop.sh $TAG --prefix"
