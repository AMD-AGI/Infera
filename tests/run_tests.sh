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
# On every container we launch, so a new run can wipe what a killed one left on
# this (reused) node.
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
#
# This script has no `set -e`, so an unchecked mktemp is not merely untidy: on a
# node where TMPDIR is not writable, $SCRATCH goes empty and every path built
# from it silently retargets the filesystem root — `mkdir /hf`, `: > /failures.txt`
# — which also fail, and the run limps on to report
#
#     (a tier failed but no per-test detail was captured — likely an image
#      build error or a native crash before pytest ran; scan above.)
#
# i.e. it blames the image for an unwritable /tmp. Seen on a Spur node in CI.
# Fail here instead, naming the directory, so the next person reads one line.
SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/infera-test.XXXXXX" 2>/dev/null)" || SCRATCH=""
if [ -z "$SCRATCH" ] || [ ! -d "$SCRATCH" ] || [ ! -w "$SCRATCH" ]; then
  echo "FATAL: cannot create a writable scratch dir under '${TMPDIR:-/tmp}'." >&2
  echo "       This node's temp space is unusable, so the run cannot start." >&2
  echo "       Set TMPDIR to somewhere writable, or take the node out of the pool." >&2
  ls -ld "${TMPDIR:-/tmp}" >&2 2>/dev/null || true
  exit 1
fi
mkdir -p "$SCRATCH/hf"
: > "$SCRATCH/failures.txt"
chmod 666 "$SCRATCH/failures.txt" 2>/dev/null || true
SCRATCH_FLAGS=(-v "$SCRATCH":/scratch -e HF_HOME=/scratch/hf)
# Worker (engine) logs, mounted at /e2e-logs in the container, one file per case.
# In CI a per-run NFS folder keyed by the job tag and shared with the dispatch
# log, written live so it survives scancel/preempt/SIGKILL; else node-local /tmp.
if [ -n "${GITHUB_ACTIONS:-}" ] || [ "${CI:-}" = "true" ] || [ -n "${INFERA_DISPATCH_LOGDIR:-}" ]; then
  SHARED_LOG_DIR="${INFERA_DISPATCH_LOGDIR:-$HOME/infera-cicd-shared-logs}/${INFERA_E2E_JOB_TAG:-local}"
  E2E_LOG_DIR="$SHARED_LOG_DIR"
else
  SHARED_LOG_DIR=""
  E2E_LOG_DIR="/tmp/infera-e2e-logs"
fi
mkdir -p "$E2E_LOG_DIR"
# World-writable: the in-container writer (root, or nobody under an NFS squash)
# does not share our uid/gid, so group perms will not do. The sticky bit stops a
# co-tenant renaming or clobbering another run's logs. Shared dir only.
if [ -n "$SHARED_LOG_DIR" ]; then chmod 1777 "$E2E_LOG_DIR" 2>/dev/null || true; fi
SCRATCH_FLAGS+=(-v "$E2E_LOG_DIR":/e2e-logs)
# The disagg tier drives its containers over srun and writes their logs here on
# the orchestrator, so it needs the path itself, not the container mount above.
export INFERA_E2E_LOG_DIR="$E2E_LOG_DIR"

_cleanup_scratch() {
  local img="$IMG_SGLANG"
  docker image inspect "$IMG_VLLM" >/dev/null 2>&1 && img="$IMG_VLLM"
  docker image inspect "$img" >/dev/null 2>&1 && timeout -k 10 120 docker run --rm \
    -v "$SCRATCH":/scratch --entrypoint sh "$img" \
    -c 'rm -rf /scratch/* /scratch/.[!.]* 2>/dev/null' >/dev/null 2>&1 || true
  rm -rf "$SCRATCH" 2>/dev/null || true
}

# Killing the srun client does NOT stop the Spur job, so a Ctrl-C / CI SIGTERM
# has to scancel it: job id from $_CUR_DISPATCH_OUT, else from the job tag.
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
# Nodes the running PD-disagg attempt placed containers on. A killed run skips
# pytest's teardown, so without this a cancel leaves prefill+decode on the GPUs.
_DISAG_NODES=""
_wipe_disag_nodes() {
  local n resv="${INFERA_E2E_RESERVATION:+--reservation=$INFERA_E2E_RESERVATION}"
  for n in ${_DISAG_NODES//,/ }; do
    echo "[cleanup] removing PD containers on $n" >&2
    # Unnamed, SLURM would call this "bash" and leave it UNLIMITED — invisible to
    # ci.yml's `infera-ci-`+run-id reclaim, on the very cancel that reclaim cleans up.
    srun -N1 -n1 -p "$SLURM_PART" -w "$n" $resv ${INFERA_E2E_SRUN_EXTRA:-} \
      -J "infera-ci-wipe-${INFERA_E2E_JOB_TAG:-local}" -t 00:05:00 \
      bash -lc 'docker rm -f $(docker ps -aq --filter name=infera-e2e-) 2>/dev/null || true' \
      >/dev/null 2>&1 || true
  done
  _DISAG_NODES=""
}
# SLURM job holding the PD pair's GPUs for the whole run (see _hold_pair).
_HOLDER_JID=""
_release_hold() {
  [ -n "$_HOLDER_JID" ] || return 0
  echo "[e2e disagg] releasing held nodes (scancel $_HOLDER_JID)" >&2
  scancel "$_HOLDER_JID" >/dev/null 2>&1 || true
  _HOLDER_JID=""
}
trap '_release_hold; _cleanup_scratch' EXIT
trap '_wipe_disag_nodes; _release_hold; _cancel_dispatched; exit 130' INT TERM
echo "[scratch] $SCRATCH  (worker logs: $E2E_LOG_DIR${SHARED_LOG_DIR:+ [shared NFS, live]})"

# Banner the worker-log dir at both ends of the run: the GH Actions log is long
# and read from the bottom after a failure, so repeating it saves a hunt.
_log_dir_banner() {
  echo ""
  echo "=================== E2E WORKER LOG LOCATION ==================="
  echo "  $E2E_LOG_DIR"
  if [ -n "$SHARED_LOG_DIR" ]; then
    echo "  (shared NFS, written live — survives scancel/preempt/SIGKILL)"
  else
    echo "  (node-local /tmp — NOT shared; lost when this machine is reclaimed)"
  fi
  echo "==============================================================="
}
_log_dir_banner

# Bind the model tree read-only at the same path. If it is absent here it lives
# on the compute node, so just forward the var and let the remote re-run mount it.
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

# --- SLURM dispatch (Spur scheduler and stock SLURM) -------------------------
# Fall back to the cluster default partition (the one sinfo stars) so this works
# on amd-spur and stock SLURM alike.
_default_partition() {
  command -v sinfo >/dev/null 2>&1 || return 0
  sinfo -h -o '%P' 2>/dev/null | sed -n 's/\*$//p' | head -1
}
SLURM_PART="${INFERA_E2E_SLURM_PARTITION:-$(_default_partition)}"
SLURM_PART="${SLURM_PART:-amd-spur}"
SLURM_TIME="${INFERA_E2E_SLURM_TIME:-02:00:00}"
# Burst QoS is a fallback, never used up front: a dispatch queued this long on
# the group node limit is resubmitted with it (see _watch_job / _dispatch_slurm).
QOS_FALLBACK="${INFERA_E2E_SLURM_QOS_FALLBACK:-amd-burst-qos}"
QOS_WAIT="${INFERA_E2E_QOS_WAIT:-30}"
# _hold_pair's own window: its -N2 --gres=gpu:8 batch job needs longer to start
# than a single-node srun, and giving up early only churns the pair-hold race.
HOLD_WAIT="${INFERA_E2E_HOLD_WAIT:-60}"

_have_slurm() { command -v srun >/dev/null 2>&1; }
# The nodes reservation $1 covers, one per line ('' if it is gone/expired).
# Spur ignores the NAME arg and dumps all reservations; match the exact block.
_reservation_nodes() {
  scontrol show reservation "$1" 2>/dev/null | awk -v r="ReservationName=$1" '
    BEGIN{RS="";FS="\n"}
    $1==r { for(i=1;i<=NF;i++) if($i ~ /Nodes=/){ n=$i; sub(/.*Nodes=/,"",n); sub(/[[:space:]].*/,"",n); print n; exit } }' \
    | tr ',' '\n' | sed '/^$/d'
}
# Ask the NODE, not squeue: a multi-node job's %N is a compacted hostlist
# (crsuse2-m2m-[090,183]) holding neither full name, and Spur has no `scontrol
# show hostnames`. Unreadable => busy, never hand out what we cannot verify.
_node_free() {
  local alloc
  alloc=$(scontrol show node "$1" 2>/dev/null | grep -oE 'CPUAlloc=[0-9]+' | head -1 | cut -d= -f2)
  [ -n "$alloc" ] && [ "$alloc" -eq 0 ] 2>/dev/null
}
# Free nodes for the PD-disagg pair, one per line: the reservation's own (a
# reserved node reads 'resv', never 'idle', so sinfo would miss it), else the
# partition's idle ones.
_candidate_nodes() {
  local n nodes=""
  [ -n "${INFERA_E2E_RESERVATION:-}" ] && nodes="$(_reservation_nodes "$INFERA_E2E_RESERVATION")"
  if [ -z "$nodes" ]; then
    sinfo -h -N -p "$SLURM_PART" -t idle -o '%n' 2>/dev/null | awk 'NF && !seen[$0]++'
    return
  fi
  for n in $nodes; do
    _node_free "$n" && echo "$n"
  done
}
# Up to $1 free nodes, skipping the comma-separated exclude list in $2. Collect
# before filtering: breaking out of a `< <(...)` mid-stream prints EPIPE noise.
_pick_idle_nodes() {
  local count="$1" excl=",${2:-}," n out=() all
  all="$(_candidate_nodes)"
  while read -r n; do
    [ -n "$n" ] || continue
    case "$excl" in *,"$n",*) continue ;; esac
    out+=("$n"); [ "${#out[@]}" -ge "$count" ] && break
  done <<< "$all"
  printf '%s\n' "${out[@]-}"
}
# Wait for two free nodes rather than give up: engines run in parallel and the
# mixed tier shares the pool, so a pair is often only free later. The CI job
# timeout is the real backstop. $1=exclude list.
_wait_for_pair() {
  local excl="$1" waited=0 every=30 limit="${INFERA_E2E_WAIT_NODES_TIMEOUT:-6400}" nodes
  while :; do
    nodes="$(_pick_idle_nodes 2 "$excl")"
    [ "$(printf '%s\n' "$nodes" | sed '/^$/d' | wc -l)" -ge 2 ] && { printf '%s\n' "$nodes"; return 0; }
    [ "$waited" -ge "$limit" ] && return 1
    [ $((waited % 120)) -eq 0 ] &&
      echo "[e2e disagg] fewer than 2 free nodes — waiting (${waited}s/${limit}s)" >&2
    sleep "$every"; waited=$((waited + every))
  done
}
# A running pair holder other than $1, on the same nodes and submitted earlier.
_rival_holder() {
  local self="$1" mine
  mine=$(squeue -h -j "$self" -o '%N' 2>/dev/null)
  [ -n "$mine" ] || return 0
  squeue -h -t running -o '%i %j %N' 2>/dev/null | awk -v self="$self" -v mine="$mine" '
    $2 ~ /^infera-ci-hold-/ && $3 == mine && $1 + 0 < self + 0 { print $1; exit }'
}
# Hold both PD nodes' GPUs for the whole run: disagg's per-step sruns leave them
# idle in between, so SLURM would hand one out and the fixed ports (etcd 2379,
# router 8000, ...) collide. Our own no-gres steps co-schedule. Sets _HOLDER_JID.
_hold_pair() {
  local pair="$1" script="$SCRATCH/hold.sh" jid st rs waited qos=() i other
  # A real script file, not --wrap: on Spur --wrap always NODE_FAILs at -N2.
  printf '#!/bin/bash\nsleep %s\n' "${INFERA_E2E_HOLD_SLEEP:-10800}" > "$script"
  for i in 1 2 3; do
    jid=$(sbatch --parsable -N2 -n2 -w "$pair" --gres=gpu:8 -p "$SLURM_PART" \
      -t "$SLURM_TIME" -J "infera-ci-hold-${INFERA_E2E_JOB_TAG:-local}" \
      ${INFERA_E2E_RESERVATION:+--reservation="$INFERA_E2E_RESERVATION"} \
      "${qos[@]}" "$script" 2>/dev/null) || continue
    waited=0
    while [ "$waited" -lt "$HOLD_WAIT" ]; do
      st=$(scontrol show job "$jid" 2>/dev/null | grep -oE 'JobState=[A-Z_]+' | cut -d= -f2)
      rs=$(scontrol show job "$jid" 2>/dev/null | grep -oE 'Reason=[A-Za-z]+' | cut -d= -f2)
      if [ "$st" = RUNNING ]; then
        # --gres does fence the pair, but two engines can submit in the same
        # instant, before either holder exists to be seen. Lower job id keeps it
        # and the other yields, so they cannot both back off and re-collide.
        other=$(_rival_holder "$jid")
        [ -z "$other" ] && { _HOLDER_JID="$jid"; return 0; }
        scancel "$jid" >/dev/null 2>&1
        echo "[e2e disagg] holder $jid started on $pair but $other holds it too — yielding" >&2
        return 1
      fi
      case "$st" in NODE_FAIL | FAILED | CANCELLED) break ;; esac
      sleep 5; waited=$((waited + 5))
    done
    # Read the reason BEFORE cancelling; the burst QoS is the way past a group
    # node limit, and a launch failure is Spur being flaky — both just retry.
    [ "${#qos[@]}" -eq 0 ] && [ "${rs#QOSGrp}" != "$rs" ] && qos=(-q "$QOS_FALLBACK")
    scancel "$jid" >/dev/null 2>&1
    echo "[e2e disagg] hold attempt $i on $pair not started (${st:-?}/${rs:-?}) — retrying" >&2
  done
  return 1
}
# One renderD* per GPU; PCI vendor 0x1002 == AMD.
_amd_gpu_count() {
  local n=0 d
  for d in /sys/class/drm/renderD*/device/vendor; do
    [ -r "$d" ] && [ "$(cat "$d" 2>/dev/null)" = "0x1002" ] && n=$((n + 1))
  done
  echo "$n"
}
# Shared by unit / engine / e2e-mixed. INFERA_E2E_LOCAL=1 (set by _dispatch_slurm
# on the remote) forces in-place.
_local_eligible() { [ "$(_amd_gpu_count)" -ge 8 ] && command -v docker >/dev/null 2>&1; }

# Spill helper (Spur has no srun --immediate): free count, -1 if the reservation
# is gone/expired, -2 if scontrol is unavailable.
_reservation_free() {
  local rname="$1" nodes n free=0
  command -v scontrol >/dev/null 2>&1 || { echo -2; return; }
  nodes=$(_reservation_nodes "$rname")
  [ -n "$nodes" ] || { echo -1; return; }
  for n in $nodes; do
    _node_free "$n" && free=$((free + 1))
  done
  echo "$free"
}
# Caps borrowed nodes at INFERA_E2E_SPILL_MAX; concurrent dispatchers can race it.
_spill_inflight() {
  squeue -h -u "$(id -un)" -o '%j' 2>/dev/null | grep -c -- 'spill' || true
}

# Report why the dispatch is still queued (a waiting job prints NOTHING, so a CI
# run looks hung and gets cancelled), and cancel + flag the two waits the caller
# can act on. $1=srun-out $2=hold-flag $3=qos-flag $4=label
_watch_job() {
  local out="$1" hold="$2" qos="$3" label="$4" jid="" state reason waited=0
  local every="${INFERA_E2E_QUEUE_LOG_INTERVAL:-60}" next="${INFERA_E2E_QUEUE_LOG_INTERVAL:-60}"
  while sleep 5; do
    waited=$((waited + 5))
    # Both srun banners: "Pending job allocation N" (the only one a job that
    # never starts prints) and "job N running on ...".
    [ -n "$jid" ] || jid=$(grep -oE 'job (allocation )?[0-9]+' "$out" 2>/dev/null \
      | grep -oE '[0-9]+' | head -1)
    [ -n "$jid" ] || continue
    state=$(squeue -h -j "$jid" -o '%T' 2>/dev/null)
    reason=$(squeue -h -j "$jid" -o '%r' 2>/dev/null)
    [ "$state" = "PENDING" ] || continue
    if [ "$reason" = "JobHoldMaxRequeue" ]; then
      : > "$hold"; scancel "$jid" >/dev/null 2>&1; return
    fi
    if [ "$reason" = "QOSGrpNodeLimit" ] && [ "$waited" -ge "$QOS_WAIT" ]; then
      : > "$qos"; scancel "$jid" >/dev/null 2>&1; return
    fi
    if [ "$waited" -ge "$next" ]; then
      next=$((waited + every))
      echo "[$label] still QUEUED on SLURM after ${waited}s — job $jid, reason=${reason:-unknown}" >&2
    fi
  done
}

_dispatch_slurm() {
  local label="$1"; shift
  if ! _have_slurm; then
    echo "[$label] WARNING: no SLURM (srun) — skipping" >&2
    return 0
  fi
  # srun's own client banners/errors (job id, "running on <node>", ...).
  local out="$SCRATCH/.dispatch-$label.out"
  _CUR_DISPATCH_OUT="$out"

  # CI (buffered srun) -> remote writes to a SHARED-NFS file we `tail -F`; local ->
  # srun forwards to $out. The dispatch stream log lands in the SAME per-run folder
  # ($SHARED_LOG_DIR, keyed by job tag) as the live worker logs, so one run's entire
  # trace — dispatch banners + every engine worker's log — sits together on NFS.
  local shared=0 logdir="" logf="" tailf="$out"
  if [ -n "$SHARED_LOG_DIR" ]; then
    shared=1
    logdir="$SHARED_LOG_DIR"   # already created + chmod'd at startup
    logf="$logdir/dispatch-${label}-$$.log"
    tailf="$logf"
  fi

  local prc=1 attempt=0 max_attempts=5 exclude="" ran
  local holdflag="$SCRATCH/.hold-$label" held=0 max_held="${INFERA_E2E_HOLD_MAX_RETRY:-30}"
  local qosflag="$SCRATCH/.qos-$label" qos=()
  while [ "$attempt" -lt "$max_attempts" ]; do
    attempt=$((attempt + 1))
    local xflag=()
    [ -n "$exclude" ] && xflag=(-x "$exclude")
    # Use the reservation while it has free nodes; when full, spill to the open
    # partition up to INFERA_E2E_SPILL_MAX borrowed nodes (else queue on it); if
    # it is gone, drop --reservation (a stale one PENDs forever on Spur).
    local resv=() jobname="infera-ci-${label}${INFERA_E2E_JOB_TAG:+-$INFERA_E2E_JOB_TAG}" mode="open"
    if [ -n "${INFERA_E2E_RESERVATION:-}" ]; then
      local rfree smax inflight
      rfree=$(_reservation_free "$INFERA_E2E_RESERVATION")
      smax="${INFERA_E2E_SPILL_MAX:-2}"
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
    echo "[$label] submitted to SLURM — the job now QUEUES until the scheduler frees a node," \
         "which can take a while on a busy cluster. Nothing prints until it starts;" \
         "queue status follows every ${INFERA_E2E_QUEUE_LOG_INTERVAL:-60}s."
    echo "[$label] streaming remote output below (live via $tailf):"
    : > "$out"; [ -n "$logf" ] && : > "$logf"; rm -f "$holdflag" "$qosflag"
    # -F follows by name + retries, tolerating the remote truncating on open.
    stdbuf -oL tail -n +1 -F "$tailf" 2>/dev/null &
    local tailpid=$!
    # Shared mode: the remote points its own fd1/fd2 at $logf (NFS) before exec'ing.
    local remote=(bash "$SCRIPT" "$@")
    [ "$shared" -eq 1 ] && \
      remote=(bash -c 'lf="$1"; shift; exec >"$lf" 2>&1; exec bash "$@"' _ "$logf" "$SCRIPT" "$@")
    # Background srun + `wait`: a foreground srun would defer the INT/TERM trap
    # until it returns, so a CI cancel could kill us before the trap scancels the
    # job; `wait` is interrupted by the signal so the trap runs promptly.
    INFERA_E2E_LOCAL=1 \
      srun -N1 -p "$SLURM_PART" --gres=gpu:8 -t "$SLURM_TIME" \
        -J "$jobname" "${xflag[@]}" "${resv[@]}" "${qos[@]}" \
        "${remote[@]}" > "$out" 2>&1 &
    local srunpid=$!
    _watch_job "$out" "$holdflag" "$qosflag" "$label" &
    local holdpid=$!
    wait "$srunpid"; prc=$?
    kill "$holdpid" 2>/dev/null; wait "$holdpid" 2>/dev/null
    # Let tail catch the final (NFS-propagated) lines before stopping it.
    sleep 3; kill "$tailpid" 2>/dev/null; wait "$tailpid" 2>/dev/null
    [ "$prc" -eq 0 ] && break
    # Held job: the watchdog already cancelled it — resubmit without burning a
    # real attempt, and fail the tier once the hold retries are exhausted.
    if [ -f "$holdflag" ]; then
      held=$((held + 1)); prc=1
      if [ "$held" -ge "$max_held" ]; then
        echo "[$label] job stuck in JobHoldMaxRequeue after $held retries — giving up" >&2
        break
      fi
      echo "[$label] job held (JobHoldMaxRequeue) — cancelled, retry $held/$max_held in 5s" >&2
      attempt=$((attempt - 1)); sleep 5; continue
    fi
    # Queued on the group node limit past $QOS_WAIT: the watchdog cancelled it —
    # resubmit once on the burst QoS (not a real attempt); refused again = fail.
    if [ -f "$qosflag" ]; then
      prc=1
      if [ "${#qos[@]}" -gt 0 ]; then
        echo "[$label] still QOSGrpNodeLimit on --qos=$QOS_FALLBACK — giving up" >&2
        break
      fi
      qos=(--qos="$QOS_FALLBACK")
      echo "[$label] QOSGrpNodeLimit for ${QOS_WAIT}s — resubmitting with --qos=$QOS_FALLBACK" >&2
      attempt=$((attempt - 1)); continue
    fi
    # Docker errors land in $logf (shared) or $out (local); "running on <node>"
    # is always an srun banner in $out.
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
  # Shared mode only: prune old logs (10 days, INFERA_DISPATCH_LOG_TTL_MIN). Drop
  # each aged-out per-run FOLDER whole — deleting only its *.log would strand the
  # folder for ever on any stray non-log file. Second sweep: pre-folder flat logs.
  if [ "$shared" -eq 1 ]; then
    local ttl="${INFERA_DISPATCH_LOG_TTL_MIN:-14400}" root
    root="$(dirname "$logdir")"
    find "$root" -mindepth 1 -maxdepth 1 -type d -mmin "+$ttl" -exec rm -rf {} + 2>/dev/null || true
    find "$root" -mindepth 1 -maxdepth 1 -type f -name '*.log' -mmin "+$ttl" -delete 2>/dev/null || true
  fi
  return "$prc"
}

# --network=host so RUN steps (pip) resolve DNS via the host resolver: these
# nodes list "nameserver 127.0.0.1" first, unreachable from a bridge build netns.
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

# One file at a time so a single ROCm/HIP native crash cannot abort the run.
# $1=Dockerfile $2=image $3=scope.
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
        # tee: stream live for CI, keep a copy for the classification below.
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

# One engine's PD-mixed suite in its own image against the shared etcd. Verbose:
# -s keeps worker stdout live, -v names each test.
run_e2e_engine() {
  local img="$1" testpath="$2"
  echo "----- e2e in $img — $testpath -----"
  docker run --rm --name "${CTR_PREFIX}e2e" --network host "${GPU_FLAGS[@]}" "${SCRATCH_FLAGS[@]}" "${E2E_FLAGS[@]}" \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 \
    -v "$REPO":/workspace:ro -w /workspace --entrypoint bash "$img" -lc \
    "$PIPDEPS; python3 -m pytest -p no:cacheprovider -o addopts= -rfE -v -s $testpath 2>&1 | stdbuf -oL tee /scratch/.e2e.out; rc=\${PIPESTATUS[0]}; grep -aE '^(FAILED|ERROR) ' /scratch/.e2e.out 2>/dev/null | sed 's|^|[e2e $img] |' >> /scratch/failures.txt; exit \$rc"
}

# Run in place when eligible, else dispatch the whole tier to one SLURM node.
# Locally: build each image, start a temp etcd, run each engine against it.
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
    local testpath="tests/e2e/pd_mixed/$e/test_mixed.py"
    [ "$e" = vllm ] && testpath="tests/e2e/pd_mixed/$e/"
    run_e2e_engine "$img" "$testpath" || rc=1
  done

  docker rm -f "$ETCD_CTR" >/dev/null 2>&1 || true
  return "$rc"
}

# Cross-node: a pytest orchestrator here drives etcd/router/prefill/decode on TWO
# idle nodes (INFERA_E2E_NODES + srun; see harness/launcher.py). A bad node is
# excluded and a fresh pair tried.
run_e2e_disagg() {
  local engines=("$@")
  echo "===== e2e PD-disaggregated (cross-node, 2 nodes): ${engines[*]} ====="
  if ! _have_slurm; then
    echo "[e2e disagg] WARNING: no SLURM (srun) — skipping PD-disaggregated tests" >&2
    return 0
  fi
  python3 -c "import pytest, pytest_asyncio, httpx" >/dev/null 2>&1 \
    || { echo "[e2e disagg] WARNING: missing host deps (pytest/pytest-asyncio/httpx) — skipping" >&2; return 0; }

  if [ -n "$SHARED_LOG_DIR" ]; then
    exec > >(stdbuf -oL tee -a "$SHARED_LOG_DIR/dispatch-disag-$$.log") 2>&1
  fi

  # An expired reservation is worse than none — every step's `srun --reservation`
  # would fail. Drop it, as _dispatch_slurm does for the mixed tier.
  if [ -n "${INFERA_E2E_RESERVATION:-}" ] && [ -z "$(_reservation_nodes "$INFERA_E2E_RESERVATION")" ]; then
    echo "[e2e disagg] WARNING: reservation '$INFERA_E2E_RESERVATION' not found — falling back to open partition '$SLURM_PART'" >&2
    unset INFERA_E2E_RESERVATION
  fi

  local rc=0 e prc out="$SCRATCH/.e2e-disag.out"
  local max_attempts=3 attempt exclude n1 n2 nodes ok
  local races max_races="${INFERA_E2E_HOLD_RACE_MAX:-10}"
  for e in "${engines[@]}"; do
    echo "----- e2e disagg — tests/e2e/pd_disag/$e -----"
    attempt=0; ok=0; exclude=""; races=0
    while [ "$attempt" -lt "$max_attempts" ]; do
      attempt=$((attempt + 1))
      # A user-pinned pair (INFERA_E2E_NODES) wins on the first try.
      if [ -n "${INFERA_E2E_NODES:-}" ] && [ "$attempt" -eq 1 ]; then
        n1="${INFERA_E2E_NODES%%,*}"; n2="${INFERA_E2E_NODES##*,}"
      else
        nodes="$(_wait_for_pair "$exclude")"
        n1="$(printf '%s\n' "$nodes" | sed -n 1p)"
        n2="$(printf '%s\n' "$nodes" | sed -n 2p)"
      fi
      if [ -z "$n1" ] || [ -z "$n2" ] || [ "$n1" = "$n2" ]; then
        echo "[e2e disagg] WARNING: no 2 free nodes in '$SLURM_PART' within ${INFERA_E2E_WAIT_NODES_TIMEOUT:-6400}s — skipping $e" >&2
        break
      fi
      # Losing the race is not a node fault, so the pair must NOT join $exclude:
      # with a small pool the engine would exclude every node and then starve on
      # an idle cluster. Bounded so a pathological loser fails loudly instead.
      if ! _hold_pair "$n1,$n2"; then
        races=$((races + 1))
        if [ "$races" -ge "$max_races" ]; then
          echo "[e2e disagg] lost the node-hold race $races times — giving up on $e" >&2
          break
        fi
        echo "[e2e disagg] could not hold $n1,$n2 (race $races/$max_races) — re-picking in 30s" >&2
        attempt=$((attempt - 1)); sleep 30; continue
      fi
      races=0
      echo "[e2e disagg] $e attempt $attempt/$max_attempts on nodes: $n1 (prefill), $n2 (decode)"
      _DISAG_NODES="$n1,$n2"
      INFERA_E2E_NODES="$n1,$n2" INFERA_E2E_SLURM_PARTITION="$SLURM_PART" \
      PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
        python3 -m pytest -p no:cacheprovider -o addopts= -rfE -v -s \
          "$REPO/tests/e2e/pd_disag/$e" 2>&1 | tee "$out"
      prc=${PIPESTATUS[0]}
      _DISAG_NODES=""   # pytest returned, so its fixtures already tore the stack down
      _release_hold
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

# Report-only: both tiers can still run (degraded) without a reservation or with
# a nearly full /home, and a hard exit here would cost a whole CI run to find out.
_e2e_preflight() {
  local avail
  if [ -n "${INFERA_E2E_RESERVATION:-}" ] && command -v scontrol >/dev/null 2>&1 \
     && [ -z "$(_reservation_nodes "$INFERA_E2E_RESERVATION")" ]; then
    echo "[e2e] ERROR: reservation '$INFERA_E2E_RESERVATION' does not exist (gone or expired)" >&2
  fi
  avail=$(df -Pk /home 2>/dev/null | awk 'NR==2{print $4}')
  case "$avail" in
    "" | *[!0-9]*) ;;
    *) [ "$avail" -lt 1048576 ] &&
      echo "[e2e] WARNING: /home has $((avail / 1024)) MB free (under 1 GB)" >&2 ;;
  esac
  return 0
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
  _e2e_preflight
  local rc=0
  [ "$scenario" != "disag" ] && { run_e2e_mixed "${engines[@]}" || rc=1; }
  [ "$scenario" != "mixed" ] && { run_e2e_disagg "${engines[@]}" || rc=1; }
  return "$rc"
}

# Gated by the SAME local-vs-SLURM decision as e2e-mixed.
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

# Reserved nodes get reused, so wipe what a killed run left behind before its
# leaked GPU/etcd containers can OOM or clash with this one.
if command -v docker >/dev/null 2>&1 && { [ -n "${INFERA_E2E_LOCAL:-}" ] || _local_eligible; }; then
  stale=$(docker ps -a --filter "name=^${CTR_PREFIX}" --filter name=infera-e2e- \
    --format '{{.Names}}' 2>/dev/null)
  if [ -n "$stale" ]; then
    echo "[cleanup] $(hostname -s): removing stale containers: $(echo $stale | tr '\n' ' ')"
    docker rm -f $stale >/dev/null 2>&1 || true
  fi
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

# Repeat the log location last (pass or fail): on a failure this is the first
# thing visible at the bottom of the GH Actions log, right where debugging starts.
_log_dir_banner
if [ -d "$E2E_LOG_DIR" ]; then
  ls -1 "$E2E_LOG_DIR"/*.log 2>/dev/null | sed 's|^|  |' || true
fi

[ "$rc" -eq 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit "$rc"
