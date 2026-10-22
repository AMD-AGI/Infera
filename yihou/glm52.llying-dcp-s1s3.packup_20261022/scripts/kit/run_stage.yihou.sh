#!/usr/bin/env bash
# Run ONE llying-kit stage on THIS node (call it on the compute node, detached):
#   spur exec <job> bash -c "cd /tmp && NK=<nk> NODE_IP=<ip> setsid nohup bash $LK/run_stage.yihou.sh <1|2|3> \
#       > $LK/../work/run_stage.S<s>-<nk>.log 2>&1 < /dev/null &"
# Steps: preflight (GPUs idle, ports free, image present, host RAM) -> up.sh -> smoke.sh (only if SIM_ACCEPT_LEN is
# empty) -> agentx.sh (C80, DURATION 1800, warm-up 10/lane) -> down.sh when DOWN=1 (docker stop only, logs saved).
# One run per stage (user directive 2026-10-22: single 30-min runs, no A1/A2).
set -uo pipefail
s="${1:?stage 1|2|3}"
LK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$LK/stages/S$s.env"
KIT="$LK/kit"
# Port / name defaults live in the kit's env.sh (the stage file only sets site values and knobs).
# shellcheck source=/dev/null
source "$KIT/env.sh"
ts() { date -u +%FT%TZ; }
echo "$(ts) stage S$s on $(hostname -s) NK=$NK NODE_IP=$NODE_IP CTR=$CTR IMAGE=$IMAGE WORK_DIR=$WORK_DIR"
mkdir -p "$WORK_DIR/cache"

# InferenceX: a private copy per work dir (the client container only mounts WORK_DIR and KIT; a symlink out of
# WORK_DIR would not resolve inside it). Source = our pinned copy at 918524ff.
IXSRC=/shared_nfs/yihou_glm52/dcp_pd_20261021/cache/InferenceX
[[ -d "$WORK_DIR/cache/InferenceX/.git" ]] || cp -a "$IXSRC" "$WORK_DIR/cache/InferenceX"

# Preflight (read-only).
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "$(ts) STOP: image $IMAGE not on this node"; exit 1; }
[[ -d "$SGLANG_OVERLAY" ]] || { echo "$(ts) STOP: overlay $SGLANG_OVERLAY missing"; exit 1; }
for i in $(seq 1 80); do
    mb=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);print(max(int(v["VRAM Total Used Memory (B)"]) for v in d.values())//1048576)' 2>/dev/null)
    [[ -n "$mb" && "$mb" -lt 400 ]] && break
    echo "$(ts) GPUs not idle (max ${mb:-?} MiB); waiting"; sleep 30
done
[[ -n "$mb" && "$mb" -lt 400 ]] || { echo "$(ts) STOP: GPUs never idle"; exit 1; }
busy=$(ss -Htln | awk '{print $4}' | sed 's/.*://' | sort -un |
    grep -xE "$ETCD_PORT|$ETCD_PEER_PORT|$ROUTER_PORT|$PREFILL_PORT|$DECODE_PORT|$BOOTSTRAP_PORT|$KV_SNAPSHOT_PORT|2743[4-9]|2744[0-4]|2753[4-9]|2754[0-4]|2775[7-9]|2776[0-4]" | tr '\n' ' ')
[[ -z "$busy" ]] || { echo "$(ts) STOP: ports already listening: $busy"; exit 1; }
free_gb=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "$(ts) preflight ok: GPUs idle (max ${mb} MiB), ports free, MemAvailable ${free_gb} GB, amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null)"

export RUN_ID="S$s-$(date -u +%Y%m%dT%H%M%SZ)"
echo "$(ts) up.sh RUN_ID=$RUN_ID"
bash "$KIT/engine/up.sh" || { echo "$(ts) STOP: up.sh failed (stack left for inspection)"; exit 1; }
if [[ -z "$SIM_ACCEPT_LEN" ]]; then
    echo "$(ts) smoke.sh (real acceptance)"; bash "$KIT/engine/smoke.sh"; echo "$(ts) smoke rc=$?"
fi
echo "$(ts) agentx.sh C$CONC ${DURATION}s"
bash "$KIT/engine/agentx/agentx.sh"; rc=$?
echo "$(ts) agentx rc=$rc; result: $WORK_DIR/runs/$RUN_ID/agentx/agentx_conc$CONC.json"
if [[ "${DOWN:-0}" == 1 ]]; then
    echo "$(ts) down.sh (docker stop, containers kept)"; bash "$KIT/engine/down.sh"
fi
echo "$(ts) STAGE S$s DONE"
