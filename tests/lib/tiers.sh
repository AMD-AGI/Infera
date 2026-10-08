# shellcheck shell=bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# The test tiers: unit, engine and PD-mixed e2e run in containers on one GPU node
# (here, or dispatched there via SLURM); the disagg tier lives in disagg.sh.

_amd_gpu_count() {
  local n=0 d
  for d in /sys/class/drm/renderD*/device/vendor; do
    [ -r "$d" ] && [ "$(cat "$d" 2>/dev/null)" = "0x1002" ] && n=$((n + 1))
  done
  echo "$n"
}

# Run in place or dispatch? INFERA_E2E_LOCAL=1 (the srun'd leg) / 0 decides outright;
# otherwise run here when this host has docker and >=8 AMD GPUs.
_run_here() {
  case "${INFERA_E2E_LOCAL:-}" in
    1) return 0 ;;
    0) return 1 ;;
  esac
  [ "$(_amd_gpu_count)" -ge 8 ] && command -v docker >/dev/null 2>&1
}

# Tells the dispatcher to exclude this node, then exits.
_node_dirty() {
  echo "${GPU_DIRTY_NODE_PREFIX}${INFERA_E2E_SLURM_NODE} $*" >&2
  exit 75
}

# Removes stale test containers before a tier. Only an exclusive node owner may also
# remove other tags' (orphaned) ones: elsewhere they can be a live sibling leg's.
prepare_node() {
  _run_here || return 0
  local exclusive=0 stale
  [ "${INFERA_E2E_EXCLUSIVE:-}" = 1 ] && exclusive=1
  if [ "$exclusive" -eq 1 ]; then
    echo "INFERA_E2E_SLURM_NODE=$INFERA_E2E_SLURM_NODE"
    command -v docker >/dev/null 2>&1 || _node_dirty "docker is unavailable"
    stale=$(
      {
        docker ps -a --filter label=infera.e2e.job_tag --format '{{.Names}}' || exit $?
        docker ps -a --filter name=infera-e2e- --filter name=infera-utest- \
          --format '{{.Names}} {{.Labels}}' |
          awk '$0 !~ /infera\.e2e\.job_tag=/{print $1}' || exit $?
      } 2>/dev/null | sort -u
    ) || _node_dirty "could not enumerate stale containers"
  else
    command -v docker >/dev/null 2>&1 || return 0
    stale=$(docker ps -a --filter "label=$CTR_LABEL" --format '{{.Names}}' 2>/dev/null)
  fi
  if [ -n "$stale" ]; then
    echo "[cleanup] $(hostname -s): removing stale containers: $(echo $stale)"
    if ! docker rm -f $stale >/dev/null 2>&1 && [ "$exclusive" -eq 1 ]; then
      _node_dirty "could not remove stale containers"
    fi
  fi
  if [ "$exclusive" -eq 1 ]; then
    python3 "$REPO/tests/e2e/harness/gpu_cleanup.py" || exit $?
  fi
  return 0
}

# --network=host: the nodes' first nameserver is 127.0.0.1, unreachable from a build netns.
# INFERA_E2E_BUILD_ARGS (K=V,...) is shared with the disagg launcher's builds.
build_image() {
  local df="$1" img="$2" args=() flags=() kv
  IFS=, read -ra args <<< "${INFERA_E2E_BUILD_ARGS:-}"
  for kv in "${args[@]}"; do [ -n "$kv" ] && flags+=(--build-arg "$kv"); done
  echo "[build] $img <- $df ${flags[*]}"
  docker build --network=host "${flags[@]}" -f "$REPO/$df" -t "$img" "$REPO"
}

# Prints "<image> <dockerfile>" for an engine.
_engine_image() {
  case "$1" in
    sglang) echo "$IMG_SGLANG $DF_SGLANG" ;;
    vllm) echo "$IMG_VLLM $DF_VLLM" ;;
    atom) echo "$IMG_ATOM $DF_ATOM" ;;
  esac
}

# Runs tests/lib/container_pytest.sh in a GPU container with the repo mounted read-only.
#   $1=container name suffix $2=image [docker flags...] -- [container_pytest.sh args...]
_container_pytest() {
  local name="$1" img="$2" flags=()
  shift 2
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do flags+=("$1"); shift; done
  shift
  docker run --rm --name "${CTR_PREFIX}${name}" "${CTR_LABELS[@]}" \
    "${GPU_FLAGS[@]}" "${SCRATCH_FLAGS[@]}" -e PYTHONDONTWRITEBYTECODE=1 "${flags[@]}" \
    -v "$REPO":/workspace:ro -w /workspace --entrypoint bash "$img" \
    -lc 'cd /workspace && exec bash tests/lib/container_pytest.sh "$@"' _ "$@"
}

run_unit() {
  echo "===== unit (pure-Python logic) ====="
  build_image "$DF_VLLM" "$IMG_VLLM" || return 1
  _container_pytest unit "$IMG_VLLM" -- "[unit]" -q tests/unit
}

# $1=Dockerfile $2=image $3=test dir
run_engine() {
  local df="$1" img="$2" scope="${3:-tests/engine}"
  echo "===== engine in $img — $scope (per-file, crash-isolated) ====="
  build_image "$df" "$img" || return 1
  _container_pytest engine "$img" -- --per-file "[engine $scope]" "$scope"
}

# One engine's PD-mixed suite against the shared etcd; -s keeps worker output live.
run_e2e_engine() {
  local img="$1" testpath="$2"
  echo "----- e2e in $img — $testpath -----"
  _container_pytest e2e "$img" --network host "${E2E_FLAGS[@]}" -e PYTHONUNBUFFERED=1 -- \
    "[e2e $img]" -v -s "$testpath"
}

run_e2e_mixed() {
  local engines=("$@") e img df rc=0 engarg=all
  echo "===== e2e PD-mixed (etcd + real workers, GPU): ${engines[*]} ====="
  if ! _run_here; then
    [ "${#engines[@]}" -eq 1 ] && engarg="${engines[0]}"
    echo "[mixed] not running in place — dispatching via srun (engines: ${engines[*]}, serial on 1 node)"
    _dispatch_slurm mixed e2e "$engarg" mixed
    return $?
  fi
  for e in "${engines[@]}"; do
    read -r img df <<< "$(_engine_image "$e")"
    build_image "$df" "$img" || return 1
  done

  docker rm -f "$ETCD_CTR" >/dev/null 2>&1
  echo "[e2e] starting temporary etcd ($ETCD_CTR)"
  docker run -d --rm --name "$ETCD_CTR" "${CTR_LABELS[@]}" --net host "$ETCD_IMG" \
    etcd --advertise-client-urls http://127.0.0.1:2379 \
         --listen-client-urls http://0.0.0.0:2379 >/dev/null
  sleep 5
  for e in "${engines[@]}"; do
    read -r img df <<< "$(_engine_image "$e")"
    if [ "$e" = vllm ]; then
      run_e2e_engine "$img" "tests/e2e/pd_mixed/vllm/" || rc=1
    else
      run_e2e_engine "$img" "tests/e2e/pd_mixed/$e/test_mixed.py" || rc=1
    fi
  done
  docker rm -f "$ETCD_CTR" >/dev/null 2>&1
  return "$rc"
}

# Report-only: both e2e tiers still run (degraded) without a reservation or disk.
_e2e_preflight() {
  local avail nodes
  if [ -n "${INFERA_E2E_RESERVATION:-}" ]; then
    if ! _have_sc; then
      echo "[e2e] ERROR: no scheduler CLI to check reservation '$INFERA_E2E_RESERVATION' — the tiers will use the open partition" >&2
    elif ! nodes=$(_reservation_nodes "$INFERA_E2E_RESERVATION"); then
      echo "[e2e] ERROR: cannot reach the scheduler to check reservation '$INFERA_E2E_RESERVATION' (query failed)" >&2
    elif [ -z "$nodes" ]; then
      echo "[e2e] ERROR: reservation '$INFERA_E2E_RESERVATION' does not exist (gone or expired)" >&2
    fi
  fi
  avail=$(df -Pk /home 2>/dev/null | awk 'NR==2{print $4}')
  case "$avail" in
    "" | *[!0-9]*) ;;
    *) [ "$avail" -lt 1048576 ] && echo "[e2e] WARNING: /home has $((avail / 1024)) MB free (under 1 GB)" >&2 ;;
  esac
  return 0
}

# run_e2e [engine] [scenario], in either order
#   engine: sglang | vllm | atom | all (default all); scenario: mixed | disag (default both)
run_e2e() {
  local engines=(sglang vllm atom) scenario="" tok rc=0
  for tok in "${1:-}" "${2:-}"; do
    case "$tok" in
      "") ;;
      all) engines=(sglang vllm atom) ;;
      sglang | vllm | atom) engines=("$tok") ;;
      mixed) scenario="mixed" ;;
      disag | disagg) scenario="disag" ;;
      *) echo "[e2e] unknown arg '$tok' (engine: sglang|vllm|atom|all; scenario: mixed|disag)"; return 2 ;;
    esac
  done
  _e2e_preflight
  [ "$scenario" != "disag" ] && { run_e2e_mixed "${engines[@]}" || rc=1; }
  [ "$scenario" != "mixed" ] && { run_e2e_disagg "${engines[@]}" || rc=1; }
  return "$rc"
}

unit_tier() {
  if _run_here; then
    run_unit
  else
    echo "[unit] not running in place — dispatching via srun"
    _dispatch_slurm unit unit
  fi
}

engine_tier() {
  local rc=0
  if ! _run_here; then
    echo "[engine] not running in place — dispatching via srun"
    _dispatch_slurm engine engine
    return $?
  fi
  run_engine "$DF_VLLM" "$IMG_VLLM" tests/engine/vllm || rc=1
  run_engine "$DF_SGLANG" "$IMG_SGLANG" tests/engine/sglang || rc=1
  return "$rc"
}
