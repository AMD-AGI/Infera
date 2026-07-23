#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
###############################################################################
# One-shot runner for the infera test suite.
#   tests/run_tests.sh unit | engine | all
#   tests/run_tests.sh e2e [sglang|vllm|atom|all] [mixed|disag]
# GPU tiers run in place when this host has docker + >=8 AMD GPUs, else srun onto
# one 8-GPU node. Env: INFERA_E2E_MODEL_DIR, INFERA_E2E_SLURM_PARTITION.

set -uo pipefail

SUITE="${1:-}"
SCRIPT="$(readlink -f "${BASH_SOURCE[0]}")"
REPO="$(dirname "$(dirname "$SCRIPT")")"

IMG_VLLM="infera/engine-vllm:test-local"
IMG_SGLANG="infera/engine-sglang:test-local"
IMG_ATOM="infera/engine-atom:test-local"
DF_VLLM="deploy/docker/Dockerfile.vllm"
DF_SGLANG="deploy/docker/Dockerfile.sglang"
DF_ATOM="deploy/docker/Dockerfile.atom"
ETCD_IMG="quay.io/coreos/etcd:v3.5.14"
# Container name prefix so a new run can wipe stragglers a prior run left behind.
CTR_PREFIX="infera-utest-"
ETCD_CTR="${CTR_PREFIX}etcd"
PIPDEPS='pip install -q pytest pytest-asyncio nats-py 2>/dev/null || true'

# --init reaps orphaned subprocesses; rest is ROCm device passthrough + host /boot
# (so ais-check can read the kernel's P2PDMA support).
GPU_FLAGS=(
  --init --privileged --ipc host --shm-size 16gb --ulimit memlock=-1
  --device /dev/kfd --device /dev/dri --group-add video --group-add render
  -v /boot:/boot:ro
)

# Per-run host scratch (HF cache + logs), shared into every container at /scratch.
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/infera-test.XXXXXX")"
mkdir -p "$SCRATCH/hf"
: > "$SCRATCH/failures.txt"
chmod 666 "$SCRATCH/failures.txt" 2>/dev/null || true
SCRATCH_FLAGS=(-v "$SCRATCH":/scratch -e HF_HOME=/scratch/hf)
# Worker logs persist on the host, mounted at /e2e-logs in the container.
E2E_LOG_DIR="/tmp/infera-e2e-logs"
mkdir -p "$E2E_LOG_DIR"
SCRATCH_FLAGS+=(-v "$E2E_LOG_DIR":/e2e-logs)

# PD-disag node hold (set by run_e2e_disagg); released on EXIT + INT/TERM so a
# killed run never leaks the reservation-pool nodes it held.
_HELD_JOBNAME=""
_release_held() {
  [ -n "$_HELD_JOBNAME" ] || return 0
  echo "[e2e disagg] releasing held nodes (scancel -n $_HELD_JOBNAME)" >&2
  scancel -n "$_HELD_JOBNAME" >/dev/null 2>&1 || true
  _HELD_JOBNAME=""
}
_cleanup_scratch() {
  _release_held
  local img="$IMG_SGLANG"
  docker image inspect "$IMG_VLLM" >/dev/null 2>&1 && img="$IMG_VLLM"
  docker image inspect "$img" >/dev/null 2>&1 && timeout -k 10 120 docker run --rm \
    -v "$SCRATCH":/scratch --entrypoint sh "$img" \
    -c 'rm -rf /scratch/* /scratch/.[!.]* 2>/dev/null' >/dev/null 2>&1 || true
  rm -rf "$SCRATCH" 2>/dev/null || true
}

# On interrupt, scancel the dispatched job (killing the srun client doesn't stop
# the Spur job). Job id from $_CUR_DISPATCH_OUT, else matched by job tag.
_CUR_DISPATCH_OUT=""
_cancel_dispatched() {
  local jids="" i suf csv
  if [ -n "$_CUR_DISPATCH_OUT" ] && [ -f "$_CUR_DISPATCH_OUT" ]; then
    jids=$(grep -oE 'srun: job [0-9]+' "$_CUR_DISPATCH_OUT" 2>/dev/null \
      | grep -oE '[0-9]+' | sort -u | tr '\n' ' ')
  fi
  if [ -z "$jids" ] && [ -n "${INFERA_E2E_JOB_TAG:-}" ]; then
    suf="-${INFERA_E2E_JOB_TAG}"
    jids=$(squeue -h -u "$(id -un)" -o '%i %j' 2>/dev/null \
      | awk -v suf="$suf" '$2 ~ /^infera-ci-/ && substr($2, length($2)-length(suf)+1)==suf {print $1}' \
      | tr '\n' ' ')
  fi
  [ -n "$jids" ] || return 0
  echo "[cleanup] cancelling dispatched SLURM job(s): $jids" >&2
  csv=$(echo $jids | tr ' ' ',')
  for i in 1 2 3 4 5; do  # retry: one scancel can hit a transient controller error
    scancel $jids >/dev/null 2>&1 || true
    sleep 2
    [ -z "$(squeue -h -j "$csv" -o '%i' 2>/dev/null)" ] && return 0
  done
}
trap _cleanup_scratch EXIT
trap '_release_held; _cancel_dispatched; exit 130' INT TERM
echo "[scratch] $SCRATCH  (worker logs: $E2E_LOG_DIR, kept)"

# INFERA_E2E_MODEL_DIR: bind read-only at the same path + forward the var (if the
# dir lives on the compute node only, just forward it; the remote re-run mounts it).
E2E_FLAGS=()
if [ -n "${INFERA_E2E_MODEL_DIR:-}" ]; then
  if [ -d "$INFERA_E2E_MODEL_DIR" ]; then
    E2E_FLAGS+=(-v "$INFERA_E2E_MODEL_DIR":"$INFERA_E2E_MODEL_DIR":ro
                -e INFERA_E2E_MODEL_DIR="$INFERA_E2E_MODEL_DIR")
    echo "[e2e] model dir: $INFERA_E2E_MODEL_DIR (read-only)"
  else
    echo "[e2e] model dir absent here — forwarding '$INFERA_E2E_MODEL_DIR' for the remote run" >&2
  fi
fi

# --- SLURM dispatch (Spur + stock SLURM) --------------------------------------
# Partition: INFERA_E2E_SLURM_PARTITION, else the cluster default (sinfo '*').
_default_partition() {
  command -v sinfo >/dev/null 2>&1 || return 0
  sinfo -h -o '%P' 2>/dev/null | sed -n 's/\*$//p' | head -1
}
SLURM_PART="${INFERA_E2E_SLURM_PARTITION:-$(_default_partition)}"
SLURM_PART="${SLURM_PART:-amd-spur}"
SLURM_TIME="${INFERA_E2E_SLURM_TIME:-02:30:00}"

_have_slurm() { command -v srun >/dev/null 2>&1; }
# Print up to $1 idle nodes in the partition, skipping the comma list in $2.
_pick_idle_nodes() {
  local count="$1" excl=",${2:-}," n out=()
  while read -r n; do
    [ -n "$n" ] || continue
    case "$excl" in *,"$n",*) continue ;; esac
    out+=("$n"); [ "${#out[@]}" -ge "$count" ] && break
  done < <(sinfo -h -N -p "$SLURM_PART" -t idle -o '%n' 2>/dev/null | awk 'NF && !seen[$0]++')
  printf '%s\n' "${out[@]-}"
}
# AMD GPU count on THIS host (renderD* with PCI vendor 0x1002 == AMD).
_amd_gpu_count() {
  local n=0 d
  for d in /sys/class/drm/renderD*/device/vendor; do
    [ -r "$d" ] && [ "$(cat "$d" 2>/dev/null)" = "0x1002" ] && n=$((n + 1))
  done
  echo "$n"
}
# In-place GPU tier iff docker + >=8 AMD GPUs. INFERA_E2E_LOCAL=1 forces it (remote).
_local_eligible() { [ "$(_amd_gpu_count)" -ge 8 ] && command -v docker >/dev/null 2>&1; }

# Reservation free-node count. Prints free count, -1 if gone, -2 if no scontrol.
_reservation_free() {
  local rname="$1" nodes run n free=0
  command -v scontrol >/dev/null 2>&1 || { echo -2; return; }
  nodes=$(scontrol show reservation "$rname" 2>/dev/null | awk -v r="ReservationName=$rname" '
    BEGIN{RS="";FS="\n"}
    $1==r { for(i=1;i<=NF;i++) if($i ~ /Nodes=/){ n=$i; sub(/.*Nodes=/,"",n); sub(/[[:space:]].*/,"",n); print n; exit } }')
  [ -n "$nodes" ] || { echo -1; return; }
  nodes=$(printf '%s' "$nodes" | tr ',' ' ')
  run=$(squeue -h -t running -o '%N' 2>/dev/null)
  for n in $nodes; do
    printf '%s\n' "$run" | grep -qw -- "$n" || free=$((free + 1))
  done
  echo "$free"
}
# Count in-flight 'spill' jobs to cap borrowed nodes at INFERA_E2E_SPILL_MAX.
_spill_inflight() { squeue -h -u "$(id -un)" -o '%j' 2>/dev/null | grep -c -- 'spill' || true; }

# Watchdog for a dispatched srun: a gres job can land in JobHoldMaxRequeue (held,
# never starts) and hang srun; if so, mark + scancel it so srun returns and the
# loop resubmits. Stops once RUNNING. $1=srun-output file, $2=marker file.
_watch_dispatch_hold() {
  local jid="" state reason i=0
  while [ "$i" -lt 1440 ]; do  # ~2h safety cap; parent kills us when srun ends
    i=$((i + 1)); sleep 5
    [ -n "$jid" ] || jid=$(grep -oE 'job [0-9]+' "$1" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    [ -n "$jid" ] || continue
    reason=$(squeue -h -j "$jid" -o '%r' 2>/dev/null)
    if printf '%s' "$reason" | grep -q 'JobHoldMaxRequeue'; then
      echo held > "$2"; scancel "$jid" >/dev/null 2>&1; return 0
    fi
    state=$(squeue -h -j "$jid" -o '%t' 2>/dev/null)
    [ "$state" = "R" ] && return 0
  done
}

# Re-run this script's args on one 8-GPU SLURM node (INFERA_E2E_LOCAL=1). Bad
# nodes, transient controller errors and JobHoldMaxRequeue are retried. $1=label.
_dispatch_slurm() {
  local label="$1"; shift
  if ! _have_slurm; then
    echo "[$label] WARNING: no SLURM (srun) — skipping" >&2
    return 0
  fi
  local out="$SCRATCH/.dispatch-$label.out"
  _CUR_DISPATCH_OUT="$out"   # let the INT/TERM trap scancel this job if aborted

  # CI (buffered srun) -> remote writes to a shared-NFS file we tail -F; local ->
  # srun forwards to $out. INFERA_DISPATCH_LOGDIR forces the shared path.
  local shared=0 logdir="" logf="" tailf="$out"
  if [ -n "${GITHUB_ACTIONS:-}" ] || [ "${CI:-}" = "true" ] || [ -n "${INFERA_DISPATCH_LOGDIR:-}" ]; then
    shared=1
    logdir="${INFERA_DISPATCH_LOGDIR:-$HOME/infera-cicd-shared-logs}"
    mkdir -p "$logdir" 2>/dev/null || true
    logf="$logdir/${label}-${INFERA_E2E_JOB_TAG:-local}-$$.log"
    tailf="$logf"
  fi

  local prc=1 attempt=0 max_attempts=5 exclude="" ran
  local holdretries=0 max_holdretries="${INFERA_E2E_HOLD_RETRIES:-40}"
  while [ "$attempt" -lt "$max_attempts" ]; do
    attempt=$((attempt + 1))
    local xflag=()
    [ -n "$exclude" ] && xflag=(-x "$exclude")
    # Reservation policy: use it while it has free nodes; when full spill to the
    # open partition (up to INFERA_E2E_SPILL_MAX) or queue; if gone, drop it.
    local resv=() jobname="infera-ci-${label}${INFERA_E2E_JOB_TAG:+-$INFERA_E2E_JOB_TAG}" mode="open"
    if [ -n "${INFERA_E2E_RESERVATION:-}" ]; then
      local rfree smax inflight
      rfree=$(_reservation_free "$INFERA_E2E_RESERVATION")
      smax="${INFERA_E2E_SPILL_MAX:-2}"
      if [ "$rfree" = "-1" ]; then
        echo "[$label] WARNING: reservation '$INFERA_E2E_RESERVATION' not found — falling back to '$SLURM_PART'" >&2
        mode="resv-gone->open"
      elif [ "$rfree" != "0" ]; then
        resv=(--reservation="$INFERA_E2E_RESERVATION"); mode="resv"   # free>0, or -2 (no scontrol)
      else
        inflight=$(_spill_inflight)
        if [ "$smax" -gt 0 ] && [ "$inflight" -lt "$smax" ]; then
          jobname="infera-ci-${label}-spill${INFERA_E2E_JOB_TAG:+-$INFERA_E2E_JOB_TAG}"
          mode="spill($((inflight + 1))/$smax)"
        else
          resv=(--reservation="$INFERA_E2E_RESERVATION"); mode="resv-wait"
        fi
      fi
    fi
    echo "[$label] dispatch $attempt/$max_attempts to '$SLURM_PART' mode=$mode${exclude:+ exclude=$exclude} (remote: $*)"
    echo "[$label] streaming remote output below (live via $tailf):"
    : > "$out"; [ -n "$logf" ] && : > "$logf"
    stdbuf -oL tail -n +1 -F "$tailf" 2>/dev/null &   # line-buffered live tail (NFS-tolerant)
    local tailpid=$!
    # Shared mode: remote redirects its own fd1/fd2 to $logf (NFS); local: srun forwards to $out.
    local remote=(bash "$SCRIPT" "$@")
    [ "$shared" -eq 1 ] && \
      remote=(bash -c 'lf="$1"; shift; exec >"$lf" 2>&1; exec bash "$@"' _ "$logf" "$SCRIPT" "$@")
    # Background srun + wait so the INT/TERM trap fires promptly on a CI cancel.
    local holdmark="$SCRATCH/.dispatch-hold-$label"; : > "$holdmark"
    _watch_dispatch_hold "$out" "$holdmark" &
    local watchpid=$!
    INFERA_E2E_LOCAL=1 \
      srun -N1 -p "$SLURM_PART" --gres=gpu:8 -t "$SLURM_TIME" \
        -J "$jobname" "${xflag[@]}" "${resv[@]}" \
        "${remote[@]}" > "$out" 2>&1 &
    local srunpid=$!
    wait "$srunpid"; prc=$?
    kill "$watchpid" 2>/dev/null; wait "$watchpid" 2>/dev/null
    sleep 3; kill "$tailpid" 2>/dev/null; wait "$tailpid" 2>/dev/null  # let tail flush final lines
    [ "$prc" -eq 0 ] && break
    # Transient JobHoldMaxRequeue (watchdog cancelled it): resubmit, don't burn an attempt.
    if [ -s "$holdmark" ] || grep -qi 'JobHoldMaxRequeue' "$out" 2>/dev/null; then
      holdretries=$((holdretries + 1))
      if [ "$holdretries" -gt "$max_holdretries" ]; then
        echo "[$label] still JobHoldMaxRequeue after $max_holdretries retries — giving up" >&2; break
      fi
      echo "[$label] dispatched job held (JobHoldMaxRequeue) — cancelled + retrying ($holdretries/$max_holdretries)" >&2
      attempt=$((attempt - 1)); sleep 5; continue
    fi
    # Bad node -> exclude + retry elsewhere; transient controller error -> wait + retry.
    ran="$(sed -n 's/.*running on \([A-Za-z0-9._-]*\).*/\1/p' "$out" | tail -1)"
    if grep -qiE 'node failure|Cannot connect to the Docker daemon' "$out" ${logf:+"$logf"} 2>/dev/null; then
      [ -n "$ran" ] && exclude="${exclude:+$exclude,}$ran"
      echo "[$label] node ${ran:-?} unusable — excluding, retrying elsewhere" >&2
      continue
    fi
    if grep -qiE 'not the Raft leader|service is currently unavailable|job submission failed' "$out" 2>/dev/null; then
      echo "[$label] transient controller error — retry in 15s" >&2; sleep 15; continue
    fi
    break  # genuine test/build failure
  done
  [ "$shared" -eq 1 ] && find "$logdir" -maxdepth 1 -type f -name '*.log' \
    -mmin "+${INFERA_DISPATCH_LOG_TTL_MIN:-14400}" -delete 2>/dev/null
  return "$prc"
}

# docker build an engine image. --network=host so RUN steps resolve DNS via the
# host resolver; the layer cache makes a no-op build fast.
build_image() {
  local df="$1" img="$2"
  echo "[build] $img <- $df"
  docker build --network=host -f "$REPO/$df" -t "$img" "$REPO"
}

run_unit() {
  echo "===== unit (pure-Python logic) ====="
  build_image "$DF_VLLM" "$IMG_VLLM" || return 1
  docker run --rm --name "${CTR_PREFIX}unit" "${GPU_FLAGS[@]}" "${SCRATCH_FLAGS[@]}" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -v "$REPO":/workspace:ro -w /workspace --entrypoint bash "$IMG_VLLM" -lc \
    "$PIPDEPS; python3 -m pytest -p no:cacheprovider -o addopts= -q -rfE tests/unit 2>&1 | stdbuf -oL tee /scratch/.unit.out; rc=\${PIPESTATUS[0]}; grep -aE '^(FAILED|ERROR) ' /scratch/.unit.out 2>/dev/null | sed 's/^/[unit] /' >> /scratch/failures.txt; exit \$rc"
}

# tests/engine one file at a time so a single native crash can't abort the run.
run_engine() {
  local df="$1" img="$2" scope="${3:-tests/engine}"
  echo "===== engine in $img — $scope (per-file, crash-isolated) ====="
  build_image "$df" "$img" || return 1
  docker run --rm --name "${CTR_PREFIX}engine" "${GPU_FLAGS[@]}" "${SCRATCH_FLAGS[@]}" \
    -e PYTHONDONTWRITEBYTECODE=1 -e INFERA_TEST_SCOPE="$scope" \
    -v "$REPO":/workspace:ro -w /workspace --entrypoint bash "$img" -lc '
      pip install -q pytest pytest-asyncio nats-py 2>/dev/null || true
      cd /workspace
      PYT="python3 -m pytest -p no:cacheprovider -o addopts= -q -rfE"
      rc=0
      for f in $(find "$INFERA_TEST_SCOPE" -name "test_*.py" | sort); do
        echo "----- pytest $f -----"
        $PYT "$f" 2>&1 | stdbuf -oL tee /scratch/.engine_f.out; code=${PIPESTATUS[0]}
        out=$(cat /scratch/.engine_f.out)
        case $code in
          139|134|137) line="CRASH(exit=$code)"; rc=1
              echo "[engine $INFERA_TEST_SCOPE] CRASH(exit=$code) $f" >> /scratch/failures.txt ;;
          0)  line=$(printf "%s" "$out" | grep -E "passed|failed|skipped|no tests ran" | tail -1) ;;
          5)  line="no tests ran (whole file skipped — not a failure)" ;;
          *)  line=$(printf "%s" "$out" | grep -E "passed|failed|error|skipped" | tail -1)
              [ -z "$line" ] && line="(exit=$code)"; rc=1
              fails=$(printf "%s\n" "$out" | grep -aE "^(FAILED|ERROR) ")
              if [ -n "$fails" ]; then
                printf "%s\n" "$fails" | sed "s|^|[engine $INFERA_TEST_SCOPE] |" >> /scratch/failures.txt
              else
                echo "[engine $INFERA_TEST_SCOPE] $f (exit=$code)" >> /scratch/failures.txt
              fi ;;
        esac
        printf "  %-56s %s\n" "$f" "$line"
      done
      exit $rc'
}

# Run one engine's PD-mixed suite in its own image against the shared etcd.
run_e2e_engine() {
  local img="$1" testpath="$2"
  echo "----- e2e in $img — $testpath -----"
  docker run --rm --name "${CTR_PREFIX}e2e" --network host "${GPU_FLAGS[@]}" "${SCRATCH_FLAGS[@]}" "${E2E_FLAGS[@]}" \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -v "$REPO":/workspace:ro -w /workspace --entrypoint bash "$img" -lc \
    "$PIPDEPS; python3 -m pytest -p no:cacheprovider -o addopts= -rfE -v -s $testpath 2>&1 | stdbuf -oL tee /scratch/.e2e.out; rc=\${PIPESTATUS[0]}; grep -aE '^(FAILED|ERROR) ' /scratch/.e2e.out 2>/dev/null | sed 's|^|[e2e $img] |' >> /scratch/failures.txt; exit \$rc"
}

# PD-mixed: run in place when eligible, else dispatch to one SLURM node. $@=engines.
run_e2e_mixed() {
  local engines=("$@")
  echo "===== e2e PD-mixed (etcd + real workers, GPU): ${engines[*]} ====="

  if [ -z "${INFERA_E2E_LOCAL:-}" ] && ! _local_eligible; then
    local engarg=all
    [ "${#engines[@]}" -eq 1 ] && engarg="${engines[0]}"
    echo "[mixed] no docker/GPU here — dispatching via srun (engines: ${engines[*]}, serial on 1 node)"
    _dispatch_slurm mixed e2e "$engarg" mixed
    return $?
  fi

  local e
  for e in "${engines[@]}"; do
    case "$e" in
      sglang) build_image "$DF_SGLANG" "$IMG_SGLANG" || return 1 ;;
      vllm)   build_image "$DF_VLLM"   "$IMG_VLLM"   || return 1 ;;
      atom)   build_image "$DF_ATOM"   "$IMG_ATOM"   || return 1 ;;
    esac
  done

  docker rm -f "$ETCD_CTR" >/dev/null 2>&1 || true
  echo "[e2e] starting temporary etcd ($ETCD_CTR)"
  docker run -d --rm --name "$ETCD_CTR" --net host "$ETCD_IMG" \
    etcd --advertise-client-urls http://127.0.0.1:2379 \
         --listen-client-urls http://0.0.0.0:2379 >/dev/null
  sleep 5

  local rc=0 img
  for e in "${engines[@]}"; do
    case "$e" in
      sglang) img="$IMG_SGLANG" ;;
      vllm)   img="$IMG_VLLM" ;;
      atom)   img="$IMG_ATOM" ;;
    esac
    # vLLM's mixed dir carries an extra kvd-offload test (test_mixed_kvd.py) —
    # run the whole dir so it's collected; other engines have only test_mixed.py.
    local testpath="tests/e2e/pd_mixed/$e/test_mixed.py"
    [ "$e" = vllm ] && testpath="tests/e2e/pd_mixed/$e/"
    run_e2e_engine "$img" "$testpath" || rc=1
  done

  docker rm -f "$ETCD_CTR" >/dev/null 2>&1 || true
  return "$rc"
}

# --- PD-disag node holding (opt-in; shared reservation pool) -------------------
# disag's many short `srun -w` steps free the pinned nodes between steps; on a
# shared pool a concurrent job can grab one mid-test. So hold both nodes' GPUs for
# the whole run with `spur alloc --gres=gpu:8` (test's no-gres steps still
# co-schedule; docker uses the physical GPUs), released at the end.
# On when INFERA_E2E_HOLD_NODES=1 or INFERA_E2E_RESERVATION is set; =0 forces off.
_hold_enabled() {
  command -v spur >/dev/null 2>&1 || return 1
  case "${INFERA_E2E_HOLD_NODES:-auto}" in
    1) return 0 ;; 0) return 1 ;;
    *) [ -n "${INFERA_E2E_RESERVATION:-}" ] ;;
  esac
}
# Two free nodes in reservation $1 (no running job on them; reserved-idle shows
# 'resv' not 'idle', so subtract running nodes from the reservation set).
_reserved_free_pair() {
  local resv="$1" run out=() n
  run="$(squeue -h -t running -o '%N' 2>/dev/null | tr ',' '\n' | sed '/^$/d' | sort -u)"
  while read -r n; do
    [ -n "$n" ] || continue
    printf '%s\n' "$run" | grep -qx -- "$n" && continue
    out+=("$n"); [ "${#out[@]}" -ge 2 ] && break
  done < <(scontrol show reservation 2>/dev/null | awk -v r="ReservationName=$resv" '
      BEGIN{RS="";FS="\n"} { for(i=1;i<=NF;i++) if($i==r){
        for(j=1;j<=NF;j++) if($j ~ /Nodes=/){n=$j; sub(/.*Nodes=/,"",n); sub(/[[:space:]].*/,"",n); print n; exit} } }' \
      | tr ',' '\n' | sed '/^$/d')
  [ "${#out[@]}" -ge 2 ] || return 1
  printf '%s\n%s\n' "${out[0]}" "${out[1]}"
}
# Hold one node's GPUs, retrying past the probabilistic JobHoldMaxRequeue until
# Granted or INFERA_E2E_HOLD_MAXWAIT (default 300s). Readiness = "Granted" banner
# in the log (Spur shows the granted salloc as pending in squeue). $1=node $2=log.
_hold_one() {
  local jid attempt=0 d
  local sleeps="${INFERA_E2E_HOLD_SLEEP:-10800}" tlim="${INFERA_E2E_HOLD_TIME:-02:30:00}"
  local deadline=$((SECONDS + ${INFERA_E2E_HOLD_MAXWAIT:-300}))
  while [ "$SECONDS" -lt "$deadline" ]; do
    attempt=$((attempt + 1))
    : > "$2"
    nohup bash -c "printf 'sleep %s\n' '$sleeps' | spur alloc -N1 -n1 --gres=gpu:8 \
      -p '$SLURM_PART' --reservation='$INFERA_E2E_RESERVATION' -w '$1' \
      -J '$_HELD_JOBNAME' -t '$tlim'" > "$2" 2>&1 &
    d=$((SECONDS + 20))
    while [ "$SECONDS" -lt "$d" ]; do
      grep -q 'Granted job allocation' "$2" 2>/dev/null && return 0
      grep -q 'JobHoldMaxRequeue' "$2" 2>/dev/null && break
      sleep 2
    done
    grep -q 'Granted job allocation' "$2" 2>/dev/null && return 0
    jid=$(grep -oE 'job allocation [0-9]+' "$2" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    [ -n "$jid" ] && scancel "$jid" >/dev/null 2>&1
    echo "[e2e disagg] $1: holder attempt $attempt not granted (transient JobHoldMaxRequeue) — retrying" >&2
    sleep 3
  done
  return 1
}

# PD-disaggregated e2e (cross-node): a host pytest orchestrator drives
# etcd/router/prefill/decode on TWO nodes (INFERA_E2E_NODES + srun). $@=engines.
run_e2e_disagg() {
  local engines=("$@")
  echo "===== e2e PD-disaggregated (cross-node, 2 nodes): ${engines[*]} ====="
  if ! _have_slurm; then
    echo "[e2e disagg] WARNING: no SLURM (srun) — skipping PD-disaggregated tests" >&2
    return 0
  fi
  python3 -c "import pytest, pytest_asyncio, httpx" >/dev/null 2>&1 \
    || { echo "[e2e disagg] WARNING: missing host deps (pytest/pytest-asyncio/httpx) — skipping" >&2; return 0; }

  # Opt-in: hold a node pair for the whole run + pin INFERA_E2E_NODES to it
  # (released via _release_held on exit / trap / function end).
  local held_pair=""
  if _hold_enabled; then
    : "${INFERA_E2E_RESERVATION:?holding needs INFERA_E2E_RESERVATION (the pool)}"
    local hpair hn1 hn2
    if [ -n "${INFERA_E2E_NODES:-}" ]; then
      hn1="${INFERA_E2E_NODES%%,*}"; hn2="${INFERA_E2E_NODES##*,}"
    else
      local waited=0 wtimeout="${INFERA_E2E_WAIT_NODES_TIMEOUT:-0}" wiv="${INFERA_E2E_WAIT_NODES_INTERVAL:-30}"
      until hpair="$(_reserved_free_pair "$INFERA_E2E_RESERVATION")"; do
        if [ "$wtimeout" -gt 0 ] && [ "$waited" -ge "$wtimeout" ]; then
          echo "[e2e disagg] <2 free nodes in '$INFERA_E2E_RESERVATION' after ${wtimeout}s — aborting" >&2
          return 1
        fi
        echo "[e2e disagg] waiting for 2 free nodes in reservation '$INFERA_E2E_RESERVATION' (waited ${waited}s)…" >&2
        sleep "$wiv"; waited=$((waited + wiv))
      done
      hn1="$(printf '%s\n' "$hpair" | sed -n 1p)"; hn2="$(printf '%s\n' "$hpair" | sed -n 2p)"
    fi
    _HELD_JOBNAME="infera-hold-${INFERA_E2E_JOB_TAG:-$$}"
    echo "[e2e disagg] holding $hn1,$hn2 for the whole run (reservation=$INFERA_E2E_RESERVATION)"
    if _hold_one "$hn1" "$SCRATCH/.hold-n1.out" && _hold_one "$hn2" "$SCRATCH/.hold-n2.out"; then
      INFERA_E2E_NODES="$hn1,$hn2"; held_pair="$hn1,$hn2"
      echo "[e2e disagg] both nodes held; pinned INFERA_E2E_NODES=$INFERA_E2E_NODES"
    else
      echo "[e2e disagg] could not hold a node pair; aborting PD-disag" >&2
      _release_held; return 1
    fi
  fi

  local rc=0 e prc out="$SCRATCH/.e2e-disag.out"
  local max_attempts=3 attempt exclude n1 n2 nodes ok
  for e in "${engines[@]}"; do
    echo "----- e2e disagg — tests/e2e/pd_disag/$e -----"
    attempt=0; ok=0; exclude=""
    while [ "$attempt" -lt "$max_attempts" ]; do
      attempt=$((attempt + 1))
      # Pinned/held pair wins (held: every attempt); else pick two idle nodes.
      if [ -n "${INFERA_E2E_NODES:-}" ] && { [ "$attempt" -eq 1 ] || [ -n "$held_pair" ]; }; then
        n1="${INFERA_E2E_NODES%%,*}"; n2="${INFERA_E2E_NODES##*,}"
      else
        nodes="$(_pick_idle_nodes 2 "$exclude")"
        n1="$(printf '%s\n' "$nodes" | sed -n 1p)"
        n2="$(printf '%s\n' "$nodes" | sed -n 2p)"
      fi
      if [ -z "$n1" ] || [ -z "$n2" ] || [ "$n1" = "$n2" ]; then
        echo "[e2e disagg] WARNING: could not find 2 usable idle nodes in '$SLURM_PART' — skipping $e" >&2
        break
      fi
      echo "[e2e disagg] $e attempt $attempt/$max_attempts on nodes: $n1 (prefill), $n2 (decode)"
      INFERA_E2E_NODES="$n1,$n2" INFERA_E2E_SLURM_PARTITION="$SLURM_PART" \
      PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
        python3 -m pytest -p no:cacheprovider -o addopts= -rfE -v -s \
          "$REPO/tests/e2e/pd_disag/$e" 2>&1 | tee "$out"
      prc=${PIPESTATUS[0]}
      [ "$prc" -eq 0 ] && { ok=1; break; }
      if grep -qiE 'node failure|Cannot connect to the Docker daemon|could not resolve a routable IP|docker build .* failed' "$out"; then
        exclude="${exclude:+$exclude,}$n1,$n2"
        echo "[e2e disagg] $e hit a bad node ($n1/$n2) — excluding, retrying with a fresh pair" >&2
        continue
      fi
      # Back-to-back reuse can leave Mooncake ports in TIME_WAIT: let them drain.
      if grep -qiE 'Address already in use|bind:|exited before becoming active' "$out"; then
        echo "[e2e disagg] $e transient bind/port collision — retry in 45s (ports draining)" >&2
        sleep 45; continue
      fi
      break  # genuine test failure
    done
    grep -aE '^(FAILED|ERROR) ' "$out" 2>/dev/null | sed "s|^|[e2e disagg $e] |" >> "$SCRATCH/failures.txt"
    [ "$ok" -eq 1 ] || rc=1
  done
  [ -n "$held_pair" ] && _release_held   # also covered by the EXIT / INT-TERM trap
  return "$rc"
}

# run_e2e [engine] [scenario] (order-independent). engine: sglang|vllm|atom|all
# (default all); scenario: mixed|disag (default both).
run_e2e() {
  local engines=(sglang vllm atom) scenario="" tok
  for tok in "${1:-}" "${2:-}"; do
    case "$tok" in
      "") ;;
      all) engines=(sglang vllm atom) ;;
      sglang|vllm|atom) engines=("$tok") ;;
      mixed) scenario="mixed" ;;
      disag|disagg) scenario="disag" ;;
      *) echo "[e2e] unknown arg '$tok' (engine: sglang|vllm|atom|all; scenario: mixed|disag)"; return 2 ;;
    esac
  done
  local rc=0
  [ "$scenario" != "disag" ] && { run_e2e_mixed "${engines[@]}" || rc=1; }
  [ "$scenario" != "mixed" ] && { run_e2e_disagg "${engines[@]}" || rc=1; }
  return "$rc"
}

# unit / engine tiers: run in place when eligible (or the dispatched remote), else
# hand the whole tier to a SLURM node.
unit_tier() {
  if [ -n "${INFERA_E2E_LOCAL:-}" ] || _local_eligible; then run_unit
  else echo "[unit] no docker/GPU here — dispatching via srun"; _dispatch_slurm unit unit; fi
}
engine_tier() {
  if [ -n "${INFERA_E2E_LOCAL:-}" ] || _local_eligible; then
    local rc=0
    run_engine "$DF_VLLM" "$IMG_VLLM" tests/engine/vllm || rc=1
    run_engine "$DF_SGLANG" "$IMG_SGLANG" tests/engine/sglang || rc=1
    return "$rc"
  else
    echo "[engine] no docker/GPU here — dispatching via srun"; _dispatch_slurm engine engine
  fi
}

# On a container host, wipe stale infera-utest-* containers a prior run leaked.
if command -v docker >/dev/null 2>&1 && { [ -n "${INFERA_E2E_LOCAL:-}" ] || _local_eligible; }; then
  docker ps -aq --filter "name=^${CTR_PREFIX}" 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1 || true
fi

rc=0
case "$SUITE" in
  unit)   unit_tier || rc=1 ;;
  engine) engine_tier || rc=1 ;;
  e2e)    run_e2e "${2:-}" "${3:-}" || rc=1 ;;
  all)    unit_tier || rc=1
          engine_tier || rc=1
          run_e2e || rc=1 ;;
  *) echo "usage: $0 unit | engine | all | e2e [sglang|vllm|atom|all] [mixed|disag]"; exit 2 ;;
esac

if [ "$rc" -ne 0 ]; then
  echo ""
  echo "===================== FAILED TEST SUMMARY ====================="
  if [ -s "$SCRATCH/failures.txt" ]; then
    sort -u "$SCRATCH/failures.txt" | sed 's/^/  /'
  else
    echo "  (a tier failed but no per-test detail was captured — likely an image"
    echo "   build error or a native crash before pytest ran; scan above.)"
  fi
  echo "==============================================================="
fi

[ "$rc" -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit "$rc"
