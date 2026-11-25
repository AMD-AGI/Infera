#!/bin/bash
# Run one tool-2 point (per-op micro-bench) in the kit container, with provenance. Deletes nothing.
# Usage: run_t2.sh NAME ENTRY [args...]
#   ENTRY: decode | prefill | e2e   (dsv41/profile_decode.py, dsv41/profile_prefill.py, main.py dsv41)
# Env: CONTAINER (default yihou-inproc-t2), WORK_DIR (host side of /out, for the printout),
#      ARGS_FILES ("presets/x.args ..." prepended), MODEL (default /models/DeepSeek-V4.1-Flash, passed as
#      --model-path unless given), EXTRA_ENV ("K=V ..." passed to docker exec -e), DRY_RUN=1.
# Result dir: /out/iterations/NAME (refuses to reuse one). Exit status = the tool's.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../../common/lib.sh"
[ $# -ge 2 ] || kit_die "usage: run_t2.sh NAME ENTRY [args...]"
NAME=$1 ENTRY=$2; shift 2
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || kit_die "NAME must match [A-Za-z0-9._-]+"
CTR=${CONTAINER:-yihou-inproc-t2}
T2=/kit/sglang_inproc/tool2_perop
case "$ENTRY" in
  decode) WD=$T2/dsv41; CMD=(python3 profile_decode.py);;
  prefill) WD=$T2/dsv41; CMD=(python3 profile_prefill.py);;
  e2e) WD=$T2; CMD=(python3 main.py dsv41);;
  *) kit_die "ENTRY must be decode|prefill|e2e";;
esac
PRE=()
if [ -n "${ARGS_FILES:-}" ]; then
  read -r -a AF <<< "$ARGS_FILES"
  mapfile -t PRE < <(kit_expand_args "${AF[@]}")
fi
MODEL_ARGS=()
case " $* ${PRE[*]:-} " in *" --model-path "*) ;; *) MODEL_ARGS=(--model-path "${MODEL:-/models/DeepSeek-V4.1-Flash}");; esac
IT=/out/iterations/$NAME
ENVS=()
for e in ${EXTRA_ENV:-}; do ENVS+=(-e "$e"); done
CMD+=("${MODEL_ARGS[@]}" "${PRE[@]}" "$@")
case "$ENTRY" in e2e) case " $* " in *" --result-json "*) ;; *) CMD+=(--result-json "$IT/result.json");; esac;;
  *) case " $* " in *" --result-dir "*) ;; *) CMD+=(--result-dir "$IT");; esac;; esac
if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' docker exec "${ENVS[@]}" -w "$WD" "$CTR" "${CMD[@]}"; echo; exit 0; fi
docker inspect "$CTR" >/dev/null 2>&1 || kit_die "container $CTR not running"
[ "$(docker inspect -f '{{index .Config.Labels "owner"}}' "$CTR")" = yihou ] || kit_die "$CTR is not ours"
if docker exec "$CTR" test -e "$IT"; then kit_die "$IT exists; pick a new NAME"; fi
docker exec "$CTR" mkdir -p "$IT"
docker exec "${ENVS[@]}" "$CTR" python3 /kit/common/provenance.py "$IT" --engine sglang "${CMD[@]}"
docker inspect "$CTR" | docker exec -i "$CTR" sh -c "python3 /kit/common/redact.py --docker-inspect > $IT/container-inspect.json"
t0=$(date +%s)
docker exec "${ENVS[@]}" -w "$WD" "$CTR" "${CMD[@]}" 2>&1 | docker exec -i "$CTR" sh -c "cat > $IT/runtime.log"
rc=${PIPESTATUS[0]}
echo "{\"exit_code\": $rc, \"wall_s\": $(( $(date +%s) - t0 )), \"name\": \"$NAME\", \"entry\": \"$ENTRY\"}" \
  | docker exec -i "$CTR" sh -c "cat > $IT/launch_status.json"
echo "point $NAME exit=$rc dir=${WORK_DIR:-/out}/iterations/$NAME"
exit "$rc"
