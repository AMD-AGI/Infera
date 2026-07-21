#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# One-shot runner for the infera test suite.
#
#   tests/run_tests.sh unit                      # pure-Python logic suite
#   tests/run_tests.sh engine                    # vllm/sglang engine suites (GPU)
#   tests/run_tests.sh e2e [sglang|vllm|atom|all] [mixed|disag]
#   tests/run_tests.sh all                       # unit + engine + e2e
#
# GPU tiers run in place when this host has docker + >=8 AMD GPUs, else `srun` the
# tier onto one 8-GPU node. PD-disag orchestrates prefill+decode on two idle nodes.
# Env: INFERA_E2E_MODEL_DIR (models, RO-mounted), INFERA_E2E_SLURM_PARTITION.

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
# Every container we launch carries this prefix so a new run can wipe stragglers
# a killed/cancelled prior run left on the same (reused) node.
CTR_PREFIX="infera-utest-"
ETCD_CTR="${CTR_PREFIX}etcd"
PIPDEPS='pip install -q pytest pytest-asyncio nats-py 2>/dev/null || true'

# --init reaps orphaned engine subprocesses; the rest is ROCm device passthrough
# (+ host /boot so ais-check can read the kernel's P2PDMA support).
GPU_FLAGS=(
  --init --privileged --ipc host --shm-size 16gb --ulimit memlock=-1
  --device /dev/kfd --device /dev/dri --group-add video --group-add render
  -v /boot:/boot:ro
)

# Per-run host scratch (HF cache + logs), shared into every container at
# /scratch and removed on exit (via a container, since containers write as root).
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/infera-test.XXXXXX")"
mkdir -p "$SCRATCH/hf"
: > "$SCRATCH/failures.txt"
chmod 666 "$SCRATCH/failures.txt" 2>/dev/null || true
SCRATCH_FLAGS=(-v "$SCRATCH":/scratch -e HF_HOME=/scratch/hf)
# Worker (engine) logs persist on the host at a fixed dir, mounted at /e2e-logs
# in the container (the harness writes there when that mount exists).
E2E_LOG_DIR="/tmp/infera-e2e-logs"
mkdir -p "$E2E_LOG_DIR"
SCRATCH_FLAGS+=(-v "$E2E_LOG_DIR":/e2e-logs)
_cleanup_scratch() {
  local img="$IMG_SGLANG"
  docker image inspect "$IMG_VLLM" >/dev/null 2>&1 && img="$IMG_VLLM"
  docker image inspect "$img" >/dev/null 2>&1 && docker run --rm -v "$SCRATCH":/scratch \
    --entrypoint sh "$img" -c 'rm -rf /scratch/* /scratch/.[!.]* 2>/dev/null' >/dev/null 2>&1 || true
  rm -rf "$SCRATCH" 2>/dev/null || true
}

# On interrupt (Ctrl-C / CI SIGTERM) scancel the dispatched job: killing the srun
# client does NOT stop the Spur job. Job id from $_CUR_DISPATCH_OUT, else job tag.
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
  # Retry: a single scancel can hit a transient Spur controller error.
  csv=$(echo $jids | tr ' ' ',')
  for i in 1 2 3 4 5; do
    scancel $jids >/dev/null 2>&1 || true
    sleep 2
    [ -z "$(squeue -h -j "$csv" -o '%i' 2>/dev/null)" ] && return 0
  done
}
trap _cleanup_scratch EXIT
trap '_cancel_dispatched; exit 130' INT TERM
echo "[scratch] $SCRATCH  (worker logs: $E2E_LOG_DIR, kept)"

# INFERA_E2E_MODEL_DIR: bind the model tree read-only at the same path + forward
# the var. If absent here (lives on the compute node) just forward it; the remote
# re-run mounts it.
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

# --- SLURM dispatch (Spur scheduler and stock SLURM) --------------------
# Partition: honor INFERA_E2E_SLURM_PARTITION; else the cluster's default (the
# one sinfo flags with '*'), so this works on amd-spur and stock SLURM alike.
_default_partition() {
  command -v sinfo >/dev/null 2>&1 || return 0
  sinfo -h -o '%P' 2>/dev/null | sed -n 's/\*$//p' | head -1
}
SLURM_PART="${INFERA_E2E_SLURM_PARTITION:-$(_default_partition)}"
SLURM_PART="${SLURM_PART:-amd-spur}"
SLURM_TIME="${INFERA_E2E_SLURM_TIME:-04:00:00}"

_have_slurm() { command -v srun >/dev/null 2>&1; }
# Print up to $1 idle nodes in the partition (one per line), skipping the
# comma-separated exclude list in $2. Used to place the PD-disagg node pair.
_pick_idle_nodes() {
  local count="$1" excl=",${2:-}," n out=()
  while read -r n; do
    [ -n "$n" ] || continue
    case "$excl" in *,"$n",*) continue ;; esac
    out+=("$n"); [ "${#out[@]}" -ge "$count" ] && break
  done < <(sinfo -h -N -p "$SLURM_PART" -t idle -o '%n' 2>/dev/null | awk 'NF && !seen[$0]++')
  printf '%s\n' "${out[@]-}"
}
# AMD GPU count on THIS host (one renderD* per GPU; PCI vendor 0x1002 == AMD).
_amd_gpu_count() {
  local n=0 d
  for d in /sys/class/drm/renderD*/device/vendor; do
    [ -r "$d" ] && [ "$(cat "$d" 2>/dev/null)" = "0x1002" ] && n=$((n + 1))
  done
  echo "$n"
}

# --- shared local-vs-SLURM decision (used by unit / engine / e2e-mixed) -------
# THIS host can run a containerized GPU tier in place iff it has docker + >=8 AMD
# GPUs. INFERA_E2E_LOCAL=1 (set by _dispatch_slurm on the remote) forces in-place.
_local_eligible() { [ "$(_amd_gpu_count)" -ge 8 ] && command -v docker >/dev/null 2>&1; }

# Reservation free-node count (spill helper; Spur has no srun --immediate).
# Prints free count, -1 if reservation gone/expired, -2 if scontrol unavailable.
_reservation_free() {
  local rname="$1" nodes run n free=0
  command -v scontrol >/dev/null 2>&1 || { echo -2; return; }
  # Spur ignores the NAME arg and dumps all reservations; match the exact block.
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
# Count in-flight 'spill'-marked jobs to cap borrowed nodes at INFERA_E2E_SPILL_MAX.
# Best-effort: concurrent dispatchers can race the cap.
_spill_inflight() {
  squeue -h -u "$(id -un)" -o '%j' 2>/dev/null | grep -c -- 'spill' || true
}

# Re-run this script's args on one 8-GPU SLURM node with INFERA_E2E_LOCAL=1 (the
# remote runs in place). Bad nodes are excluded and retried; transient controller
# errors are retried. $1=label, $2..=script args.
_dispatch_slurm() {
  local label="$1"; shift
  if ! _have_slurm; then
    echo "[$label] WARNING: no SLURM (srun) — skipping" >&2
    return 0
  fi
  # $out: srun's own client banners/errors (job id, "running on <node>", ...).
  local out="$SCRATCH/.dispatch-$label.out"
  _CUR_DISPATCH_OUT="$out"   # let the INT/TERM trap scancel this job if aborted

  # CI (buffered srun) -> remote writes to a SHARED-NFS file we `tail -F`; local ->
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
  while [ "$attempt" -lt "$max_attempts" ]; do
    attempt=$((attempt + 1))
    local xflag=()
    [ -n "$exclude" ] && xflag=(-x "$exclude")
    # Reservation policy: use it while it has free nodes; when full, spill to the
    # open partition up to INFERA_E2E_SPILL_MAX borrowed nodes (else queue on it);
    # if it's gone, drop --reservation (a stale one PENDs forever on Spur).
    local resv=() jobname="infera-ci-${label}${INFERA_E2E_JOB_TAG:+-$INFERA_E2E_JOB_TAG}" mode="open"
    if [ -n "${INFERA_E2E_RESERVATION:-}" ]; then
      local rfree smax inflight
      rfree=$(_reservation_free "$INFERA_E2E_RESERVATION")
      smax="${INFERA_E2E_SPILL_MAX:-0}"
      if [ "$rfree" = "-1" ]; then
        echo "[$label] WARNING: reservation '$INFERA_E2E_RESERVATION' not found — falling back to open partition '$SLURM_PART'" >&2
        mode="resv-gone->open"
      elif [ "$rfree" != "0" ]; then
        # free>0, or -2 (no scontrol): use the reservation.
        resv=(--reservation="$INFERA_E2E_RESERVATION"); mode="resv"
      else
        inflight=$(_spill_inflight)
        if [ "$smax" -gt 0 ] && [ "$inflight" -lt "$smax" ]; then
          # spill marker sits before the run_id-engine suffix so ci.yml reclaim matches.
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
    # stdbuf -oL line-buffers tail to our stdout; -F follows by name + retry
    # (tolerates the remote truncating on open, and polls over NFS).
    stdbuf -oL tail -n +1 -F "$tailf" 2>/dev/null &
    local tailpid=$!
    # Shared mode: the remote redirects its own fd1/fd2 to $logf (NFS) before
    # exec'ing so output lands there live; local mode: srun forwards it to $out.
    local remote=(bash "$SCRIPT" "$@")
    [ "$shared" -eq 1 ] && \
      remote=(bash -c 'lf="$1"; shift; exec >"$lf" 2>&1; exec bash "$@"' _ "$logf" "$SCRIPT" "$@")
    # Background srun + `wait`: a foreground srun would defer the INT/TERM trap
    # until it returns, so a CI cancel could kill us before the trap scancels the
    # job; `wait` is interrupted by the signal so the trap runs promptly.
    INFERA_E2E_LOCAL=1 \
      srun -N1 -p "$SLURM_PART" --gres=gpu:8 -t "$SLURM_TIME" \
        -J "$jobname" "${xflag[@]}" "${resv[@]}" \
        "${remote[@]}" > "$out" 2>&1 &
    local srunpid=$!
    wait "$srunpid"; prc=$?
    # Let tail catch the final (NFS-propagated) lines before stopping it.
    sleep 3; kill "$tailpid" 2>/dev/null; wait "$tailpid" 2>/dev/null
    [ "$prc" -eq 0 ] && break
    # Retry on transient faults. Docker errors are in $logf (shared) or $out
    # (local); "running on <node>" is always an srun banner in $out.
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
  # Shared mode only: prune logs older than 10 days (INFERA_DISPATCH_LOG_TTL_MIN).
  [ "$shared" -eq 1 ] && find "$logdir" -maxdepth 1 -type f -name '*.log' \
    -mmin "+${INFERA_DISPATCH_LOG_TTL_MIN:-14400}" -delete 2>/dev/null
  return "$prc"
}

# docker build the engine image. --network=host so RUN steps (pip) resolve DNS
# via the host resolver (these nodes list "nameserver 127.0.0.1" first, which a
# default bridge build netns can't reach). Layer cache makes a no-op build fast.
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

# tests/engine one file at a time so a single ROCm/HIP native crash can't abort
# the run. Each image runs only its own subtree. $1=Dockerfile $2=image $3=scope.
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
        # Stream pytest output live (so CI shows real test progress) AND keep a
        # copy for the crash/summary/failure classification below.
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
  # Verbose: pytest capture off (-s, live worker stdout) + per-test names (-v).
  docker run --rm --name "${CTR_PREFIX}e2e" --network host "${GPU_FLAGS[@]}" "${SCRATCH_FLAGS[@]}" "${E2E_FLAGS[@]}" \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -v "$REPO":/workspace:ro -w /workspace --entrypoint bash "$img" -lc \
    "$PIPDEPS; python3 -m pytest -p no:cacheprovider -o addopts= -rfE -v -s $testpath 2>&1 | stdbuf -oL tee /scratch/.e2e.out; rc=\${PIPESTATUS[0]}; grep -aE '^(FAILED|ERROR) ' /scratch/.e2e.out 2>/dev/null | sed 's|^|[e2e $img] |' >> /scratch/failures.txt; exit \$rc"
}

# PD-mixed: run in place when eligible, else dispatch to one SLURM node. Locally:
# build each engine image, start a temp etcd, run each engine's suite against it.
# $@ = engines.
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
    run_e2e_engine "$img" "tests/e2e/pd_mixed/$e/test_mixed.py" || rc=1
  done

  docker rm -f "$ETCD_CTR" >/dev/null 2>&1 || true
  return "$rc"
}

# PD-disaggregated e2e (cross-node): a pytest orchestrator here drives
# etcd/router/prefill/decode on TWO idle nodes (via INFERA_E2E_NODES + srun; see
# harness/launcher.py). A bad node is excluded and a fresh pair tried. $@=engines.
run_e2e_disagg() {
  local engines=("$@")
  echo "===== e2e PD-disaggregated (cross-node, 2 nodes): ${engines[*]} ====="
  if ! _have_slurm; then
    echo "[e2e disagg] WARNING: no SLURM (srun) — skipping PD-disaggregated tests" >&2
    return 0
  fi
  python3 -c "import pytest, pytest_asyncio, httpx" >/dev/null 2>&1 \
    || { echo "[e2e disagg] WARNING: missing host deps (pytest/pytest-asyncio/httpx) — skipping" >&2; return 0; }

  local rc=0 e prc out="$SCRATCH/.e2e-disag.out"
  local max_attempts=3 attempt exclude n1 n2 nodes ok
  for e in "${engines[@]}"; do
    echo "----- e2e disagg — tests/e2e/pd_disag/$e -----"
    attempt=0; ok=0; exclude=""
    while [ "$attempt" -lt "$max_attempts" ]; do
      attempt=$((attempt + 1))
      # A user-pinned pair (INFERA_E2E_NODES) wins on the first try; otherwise
      # pick two idle nodes, skipping any excluded after a bad attempt.
      if [ -n "${INFERA_E2E_NODES:-}" ] && [ "$attempt" -eq 1 ]; then
        n1="${INFERA_E2E_NODES%%,*}"; n2="${INFERA_E2E_NODES##*,}"
      else
        nodes="$(_pick_idle_nodes 2 "$exclude")"
        n1="$(printf '%s\n' "$nodes" | sed -n 1p)"
        n2="$(printf '%s\n' "$nodes" | sed -n 2p)"
      fi
      if [ -z "$n1" ] || [ -z "$n2" ] || [ "$n1" = "$n2" ]; then
        echo "[e2e disagg] WARNING: could not find 2 usable idle nodes in '$SLURM_PART' — skipping $e (set INFERA_E2E_SLURM_PARTITION to your cluster's partition)" >&2
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
      break  # genuine test failure — stop and report it
    done
    grep -aE '^(FAILED|ERROR) ' "$out" 2>/dev/null | sed "s|^|[e2e disagg $e] |" >> "$SCRATCH/failures.txt"
    [ "$ok" -eq 1 ] || rc=1
  done
  return "$rc"
}

# run_e2e [engine] [scenario]   (order-independent)
#   engine    sglang | vllm | atom | all   (default all)
#   scenario  mixed | disag                (default BOTH mixed + disag)
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

# unit / engine tiers, gated by the SAME local-vs-SLURM decision as e2e-mixed:
# run in place when eligible (or already the dispatched remote), else hand the
# whole tier to a SLURM node via the shared dispatcher.
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

# On a container host (dispatched node or local-eligible box) wipe stale
# infera-utest-* containers a killed/cancelled prior run left behind, so leaked
# GPU/etcd containers can't OOM or clash with this run (reserved nodes are reused).
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
