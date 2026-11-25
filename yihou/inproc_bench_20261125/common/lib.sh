#!/bin/bash
# Shared shell helpers of the in-process bench kits. Sourced by <kit>/scripts/*.sh, never executed.
# Contract: every container is named yihou-inproc-*, carries label owner=yihou, runs `sleep infinity`,
# and sees: the model read-only at /models/<name>, this repo dir read-only at /kit, a work dir at /out,
# and a cache dir at /cache (HOME/TMP/JIT caches; keep it under a yihou path on fast local disk).
# Nothing here deletes files. Stop/remove only containers whose name starts with yihou-inproc-.

kit_die() { echo "ERROR: $*" >&2; exit 2; }
kit_need() { for v in "$@"; do [ -n "${!v:-}" ] || kit_die "set $v"; done; }
kit_yihou_path() { case "$1" in *yihou*) ;; *) kit_die "$2=$1 must be under a yihou directory";; esac; }

# kit_create_container NAME IMAGE [ENVFILE...]
kit_create_container() {
  local name=$1 image=$2; shift 2
  kit_need MODEL_DIR WORK_DIR CACHE_DIR
  case "$name" in yihou-inproc-*) ;; *) kit_die "container name must start with yihou-inproc-";; esac
  kit_yihou_path "$WORK_DIR" WORK_DIR; kit_yihou_path "$CACHE_DIR" CACHE_DIR
  [ -d "$MODEL_DIR" ] || kit_die "MODEL_DIR=$MODEL_DIR not found"
  if docker inspect "$name" >/dev/null 2>&1; then echo "container $name exists; reuse it"; return 0; fi
  mkdir -p "$WORK_DIR" "$CACHE_DIR/home" "$CACHE_DIR/tmp"
  # KIT_DIR: a copy of this kit the docker daemon can read (it cannot read root-squashed NFS homes).
  local kit_root; kit_root=${KIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
  local mname; mname=$(basename "$MODEL_DIR")
  local args=(docker run -d --name "$name" --label owner=yihou --label kit=inproc_bench
    --network host --ipc host --device /dev/kfd --device /dev/dri --group-add video
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64g
    --ulimit nofile=1048576:1048576 --ulimit memlock=-1
    -v "$MODEL_DIR:/models/$mname:ro" -v "$kit_root:/kit:ro" -v "$WORK_DIR:/out" -v "$CACHE_DIR:/cache"
    -e HOME=/cache/home -e TMPDIR=/cache/tmp -e XDG_CACHE_HOME=/cache/xdg -e HF_HOME=/cache/hf
    -e TRITON_CACHE_DIR=/cache/triton -e AITER_JIT_DIR=/cache/aiter_jit -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor
    -e VLLM_CACHE_ROOT=/cache/vllm -e PYTHONUNBUFFERED=1 -e PYTHONNOUSERSITE=1)
  [ "${PIDHOST:-1}" = 1 ] && args+=(--pid host)   # 2nd RCCL container on a node needs it (ROCm/rccl#2188)
  [ -n "${GPUS:-}" ] && args+=(-e "ROCR_VISIBLE_DEVICES=$GPUS")
  local envfile line
  for envfile in "$@"; do
    [ -f "$envfile" ] || kit_die "env file $envfile not found"
    while IFS= read -r line; do
      case "$line" in ''|'#'*) continue;; esac
      args+=(-e "$line")
    done < "$envfile"
  done
  for e in ${EXTRA_ENV:-}; do args+=(-e "$e"); done
  "${args[@]}" --entrypoint sleep "$image" infinity
  docker inspect -f '{{.Id}} {{.Image}}' "$name"
}

# kit_run_point ENGINE CONTAINER NAME TOOL [tool args...]
# Runs /kit/<engine>_inproc/bench/<TOOL>.py inside CONTAINER with result dir /out/iterations/NAME.
kit_run_point() {
  local engine=$1 ctr=$2 name=$3 tool=$4; shift 4
  local it=/out/iterations/$name
  docker inspect "$ctr" >/dev/null 2>&1 || kit_die "container $ctr not running (create_container.sh first)"
  [ "$(docker inspect -f '{{index .Config.Labels "owner"}}' "$ctr")" = yihou ] || kit_die "$ctr is not ours"
  if docker exec "$ctr" test -e "$it"; then kit_die "$it exists; pick a new NAME"; fi
  local runner=(python3); [ -n "${TORCHRUN_NPROC:-}" ] && runner=(torchrun --nproc-per-node "$TORCHRUN_NPROC")
  local cmd=("${runner[@]}" "/kit/${engine}_inproc/bench/${tool}.py" "$@")
  case " $* " in *" --result-dir "*) ;; *) cmd+=(--result-dir "$it");; esac
  if [ "${DRY_RUN:-0}" = 1 ]; then printf '%q ' docker exec "$ctr" "${cmd[@]}"; echo; return 0; fi
  docker exec "$ctr" mkdir -p "$it"
  docker exec "$ctr" python3 /kit/common/provenance.py "$it" --engine "$engine" "${cmd[@]}"
  docker inspect "$ctr" | docker exec -i "$ctr" sh -c "cat > $it/container-inspect.json"
  local t0 rc; t0=$(date +%s)
  docker exec -w "$it" "$ctr" "${cmd[@]}" 2>&1 | docker exec -i "$ctr" sh -c "tee $it/runtime.log" | grep --line-buffered -E '^\[(step_bench|forward|infer_|engine_step)|Error|error:|Traceback'
  rc=${PIPESTATUS[0]}  # docker exec's status (no `|| true` here: it would reset PIPESTATUS)
  echo "{\"exit_code\": $rc, \"wall_s\": $(( $(date +%s) - t0 )), \"name\": \"$name\", \"tool\": \"$tool\"}" \
    | docker exec -i "$ctr" sh -c "cat > $it/launch_status.json"
  echo "point $name exit=$rc dir=$WORK_DIR/iterations/$name"
  return "$rc"
}

# kit_expand_args FILE... -> prints the args of preset files, one per line (comments/blank lines skipped)
kit_expand_args() {
  local f
  for f in "$@"; do
    [ -f "$f" ] || kit_die "args file $f not found"
    grep -v '^[[:space:]]*#' "$f" | tr -s ' \t' '\n\n' | grep -v '^$'
  done
}
