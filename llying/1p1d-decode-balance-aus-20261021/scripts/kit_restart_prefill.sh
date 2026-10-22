#!/usr/bin/env bash
# Queue step: replace the kit's prefill on a live deployment (the decode stays up). The old container is stopped and
# renamed <name>-<TAG> (kept for its logs), its followed log moves to prefill-0.before-<TAG>.log, and the new one starts
# through the kit's own engine/worker.sh once the old one's pinned host pools are back (4 ranks x 250 GB).
# GPUS picks the new prefill's devices: other GPUs than the old one's avoid waiting for its VRAM (10-65 min).
# MTP=1 gives the prefill the decode's EAGLE flags (PREFILL_EXTRA_ARGS; see RESULT §2.13 for why it needs them).
# Usage: kit_restart_prefill.sh RUN=<id> TAG=<suffix> [GPUS=0,1,2,3] [MTP=0|1]
set -uo pipefail
for a in "$@"; do export "$a"; done
: "${RUN:?}" "${TAG:?}"
export PREFILL0_GPUS="${GPUS:-4,5,6,7}"
source "$(dirname "$0")/kit_env.sh"
LOGS="$PR_RUNS/$RUN/logs"
extra=""
[[ "${MTP:-0}" == 1 ]] && extra="--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6"
name="$CTR-prefill0"
prefill_ids() {
    curl -sf -m 10 "$ROUTER_URL/v1/workers" |
        python3 -c 'import json,sys; print(*sorted(w["worker_id"] for w in json.load(sys.stdin)["workers"] if w["disagg_mode"] == "prefill"))'
}
old_id="$(prefill_ids)"
say "old prefill worker: ${old_id:-none}"

touch "$PR_RUNS/$RUN/.starting"   # kit_patrol.sh skips /health while this exists
trap 'rm -f "$PR_RUNS/$RUN/.starting"' EXIT
ssh "$P_NODE" "docker stop -t 60 $name >/dev/null && docker rename $name $name-$TAG" || die "stop $name"
sleep 5
[[ -e "$LOGS/prefill-0.log" ]] && mv "$LOGS/prefill-0.log" "$LOGS/prefill-0.before-$TAG.log"
say "stopped $name (kept as $name-$TAG)"
# worker_id is host:port, the same for the new process: wait for the old registration to lapse first, so the
# prefill the router lists afterwards is the new one.
for i in $(seq 1 60); do
    [[ -z "$(prefill_ids 2>/dev/null)" ]] && break
    sleep 10
done
[[ -z "$(prefill_ids 2>/dev/null)" ]] || die "router still lists the stopped prefill after 10 min"
say "old registration gone"

need_gb=1100
for i in $(seq 1 180); do   # 90 min: the driver frees a HiCache prefill's VRAM late
    avail="$(ssh "$P_NODE" "free -g | awk '/^Mem:/ {print \$7}'")"
    busy="$(ssh "$P_NODE" "rocm-smi --showmemuse --csv" | awk -F, -v g="$PREFILL0_GPUS" 'BEGIN{n=split(g,a,","); for(i=1;i<=n;i++) w["card" a[i]]=1} /^card/ && ($1 in w) && $2+0 >= 2 {print $1}' | tr '\n' ' ')"
    (( ${avail:-0} >= need_gb )) && [[ -z "$busy" ]] && break
    (( i % 6 == 1 )) && say "waiting: host available ${avail:-?} GB (need $need_gb), busy GPUs: ${busy:-none}"
    sleep 30
done
(( ${avail:-0} >= need_gb )) && [[ -z "$busy" ]] || die "prefill node not ready: host ${avail:-?} GB, busy GPUs $busy"

say "starting prefill on GPUs $PREFILL0_GPUS ${extra:+with: $extra}"
kit_on "$P_NODE" "PREFILL_EXTRA_ARGS=$(q "$extra") bash engine/worker.sh prefill 0" || die "worker.sh prefill 0"
follow "$name" "$P_NODE" "$LOGS/prefill-0.log"
for i in $(seq 1 120); do
    curl -sf -m 5 -o /dev/null "$PREFILL_URL/health" && break
    [[ "$(ssh "$P_NODE" "docker inspect -f '{{.State.Running}}' $name")" == true ]] || die "$name stopped while starting"
    sleep 15
done
curl -sf -m 5 -o /dev/null "$PREFILL_URL/health" || die "prefill not healthy"
# A prefill registers only after infera's start-up probe has moved KV to every decode rank (engine/up.sh).
for i in $(seq 1 90); do
    ids="$(prefill_ids 2>/dev/null)"
    [[ -n "$ids" ]] && break
    sleep 10
done
[[ -n "$ids" ]] || die "router does not list the new prefill"
say "new prefill registered: $ids"
grep -m1 -oE "'speculative_algorithm': [^,]+" "$LOGS/prefill-0.log"
say "prefill replaced (GPUs $PREFILL0_GPUS, MTP=${MTP:-0})"
