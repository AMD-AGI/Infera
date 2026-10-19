#!/usr/bin/env bash
# Remove what THIS run tag created, and only that. Prints a JSON report on
# stdout; exits 0 only if nothing with the label is left, none of the band's
# ports is still listening, and (when something was removed) the cards drained.
#
# A container is removed only if it carries the label
# ${E2E_KIT_LABEL_KEY}=${E2E_KIT_RUN_TAG} that deploy.sh put on it -- never by
# name alone and never by pattern: on a shared host a name match is how you kill
# somebody else's job, possibly your own other deployment.
#
# Before removing the engine container it hands the work root back to the
# calling user (the engine runs as root and writes its logs there), because a
# container that is gone can no longer chown what it wrote.
#
# After removal it waits for this tag's cards to fall back below 4096 MiB:
# `docker rm -f` returns before the driver frees VRAM, and an immediate relaunch
# can then fail. An absent VRAM reading is NOT treated as drained.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$HERE/env.sh"
: "${E2E_KIT_DRAIN_TIMEOUT:=300}"

ids=()
for id in $(docker ps -aq --filter "label=${KIT_LABEL}"); do
  lbl=$(docker inspect "$id" --format "{{ index .Config.Labels \"${E2E_KIT_LABEL_KEY}\" }}" 2>/dev/null)
  [ "$lbl" = "$E2E_KIT_RUN_TAG" ] && ids+=("$id")
done

reclaimed=false
for id in "${ids[@]:-}"; do
  [ -n "$id" ] || continue
  if docker exec "$id" test -d "$E2E_KIT_WORK_ROOT" >/dev/null 2>&1; then
    docker exec "$id" chown -R "$(id -u):$(id -g)" "$E2E_KIT_WORK_ROOT" >/dev/null 2>&1 && reclaimed=true
  fi
done

removed=()
for id in "${ids[@]:-}"; do
  [ -n "$id" ] || continue
  name=$(docker inspect "$id" --format '{{.Name}}' 2>/dev/null | sed 's#^/##')
  docker rm -f "$id" >/dev/null 2>&1 && removed+=("$name")
done

waited=false; drained=null; reading=""; t0=$(date +%s)
if [ "${#removed[@]}" -gt 0 ]; then
  waited=true; drained=false
  while :; do
    if reading=$(bash "$HERE/gpu_used.sh" "$E2E_KIT_GPU_DEVICES"); then
      busy=0; for pair in $reading; do [ "${pair#*:}" -gt 4096 ] && busy=1; done
      [ "$busy" = 0 ] && { drained=true; break; }
    fi
    [ $(( $(date +%s) - t0 )) -ge "$E2E_KIT_DRAIN_TIMEOUT" ] && break
    sleep 3
  done
else
  reading=$(bash "$HERE/gpu_used.sh" "$E2E_KIT_GPU_DEVICES" 2>/dev/null || true)
fi

left=$(docker ps -aq --filter "label=${KIT_LABEL}" | wc -l)
listening=$(ss -ltn 2>/dev/null | awk 'NR>1{print $4}' | sed 's/.*://' | sort -u)
still=(); for p in $KIT_ALL_PORTS; do grep -qx "$p" <<< "$listening" && still+=("$p"); done

R_TAG="$E2E_KIT_RUN_TAG" R_LABEL="$KIT_LABEL" R_REMOVED="${removed[*]:-}" R_LEFT="$left" \
R_STILL="${still[*]:-}" R_PORTS="$KIT_ALL_PORTS" R_GPUS="$E2E_KIT_GPU_DEVICES" R_WAITED="$waited" \
R_DRAINED="$drained" R_READING="$reading" R_RECLAIMED="$reclaimed" R_SECS="$(( $(date +%s) - t0 ))" \
R_AT="$(date -u +%FT%TZ)" python3 -c '
import json, os
e = os.environ
print(json.dumps({
  "at": e["R_AT"],
  "run_tag": e["R_TAG"],
  "label": e["R_LABEL"],
  "removed": e["R_REMOVED"].split(),
  "containers_left_with_label": int(e["R_LEFT"]),
  "ports_checked": [int(x) for x in e["R_PORTS"].split()],
  "ports_still_listening": [int(x) for x in e["R_STILL"].split()],
  "gpu_devices": e["R_GPUS"],
  "work_root_reclaimed": e["R_RECLAIMED"] == "true",
  "waited_for_drain": e["R_WAITED"] == "true",
  "drained": {"true": True, "false": False}.get(e["R_DRAINED"]),
  "vram_used_mib_after": e["R_READING"],
  "seconds": int(e["R_SECS"]),
}, indent=2))'
[ "$left" -eq 0 ] && [ "${#still[@]}" -eq 0 ] && [ "$drained" != false ]
