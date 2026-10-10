# shellcheck shell=bash
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# Host-side setup shared by every tier: site profile, target arch and images,
# per-job container names, scratch and worker-log directories.

# Must match tests/e2e/harness/images.py (pinned by test_arch_overlay.py).
IMG_VLLM="infera/engine-vllm:test-local"
IMG_SGLANG="infera/engine-sglang:test-local"
IMG_ATOM="infera/engine-atom:test-local"
DF_VLLM="deploy/docker/Dockerfile.vllm"
DF_SGLANG="deploy/docker/Dockerfile.sglang"
DF_ATOM="deploy/docker/Dockerfile.atom"
ETCD_IMG="quay.io/coreos/etcd:v3.5.14"
GPU_DIRTY_NODE_PREFIX="INFERA_E2E_GPU_NODE_DIRTY_NODE="

# --init reaps orphaned engine subprocesses; /boot lets ais-check read P2PDMA support.
GPU_FLAGS=(
  --init --privileged --ipc host --shm-size 16gb --ulimit memlock=-1
  --device /dev/kfd --device /dev/dri --group-add video --group-add render
  -v /boot:/boot:ro
)

# Sets $1 to the remaining args unless the caller already set it, then exports it.
# Site profiles under tests/sites/ may only call this.
site_default() {
  local var="$1"
  shift
  [ -n "${!var:-}" ] || printf -v "$var" '%s' "$*"
  export "${var?}"
}

load_site_profile() {
  [ -n "${INFERA_E2E_SITE:-}" ] || return 0
  local f="$REPO/tests/sites/${INFERA_E2E_SITE}.env"
  if [ ! -f "$f" ]; then
    echo "FATAL: no site profile named '${INFERA_E2E_SITE}' (looked for $f)." >&2
    echo "       available profiles:" >&2
    ls -1 "$REPO/tests/sites"/*.env 2>/dev/null | sed 's|.*/||; s|\.env$||; s|^|         |' >&2
    exit 2
  fi
  # shellcheck source=/dev/null
  . "$f"
  echo "[site] profile '$INFERA_E2E_SITE' loaded from $f"
}

# First GPU agent's gfx name, else empty. Read as-is: pipefail can fail this on a good node.
_probe_gfx_arch() {
  command -v rocminfo >/dev/null 2>&1 || return 0
  local names
  names="$(rocminfo 2>/dev/null | sed -n 's/^ *Name: *\(gfx[0-9a-f]*\) *$/\1/p')"
  printf '%s' "${names%%$'\n'*}"
}

# A declared arch is forwarded to the remote leg; a probed one is not, so the
# compute node probes itself instead of inheriting a GPU-less login host's answer.
resolve_arch() {
  local src="declared"
  GFX_ARCH="${INFERA_E2E_GFX_ARCH:-}"
  if [ -n "$GFX_ARCH" ]; then
    export INFERA_E2E_GFX_ARCH
  else
    GFX_ARCH="$(_probe_gfx_arch)"
    src="detected on $(hostname -s)"
  fi
  if [ -z "$GFX_ARCH" ]; then
    GFX_ARCH="gfx950"
    src="default — no GPU here to ask"
  fi
  case "$GFX_ARCH" in
    gfx950) ;;
    gfx942)
      # Only sglang's image is arch-specific; vLLM and ATOM serve both arches.
      IMG_SGLANG="infera/engine-sglang-gfx942:test-local"
      DF_SGLANG="deploy/docker/Dockerfile.sglang.gfx942"
      ;;
    *)
      echo "FATAL: GPU architecture '$GFX_ARCH' ($src) is neither gfx950 nor gfx942." >&2
      exit 2
      ;;
  esac
  echo "[arch] target GPU architecture: $GFX_ARCH ($src)"
}

# Containers are named and labelled per job, so matrix legs sharing a node never
# remove each other's etcd or test container.
init_job_names() {
  export INFERA_E2E_JOB_TAG="${INFERA_E2E_JOB_TAG:-local-$$}"
  CTR_TAG="$(printf '%s' "$INFERA_E2E_JOB_TAG" | tr -c 'A-Za-z0-9_.-' '-')"
  CTR_TAG="${CTR_TAG:-local}"
  CTR_PREFIX="infera-utest-${CTR_TAG}-"
  ETCD_CTR="${CTR_PREFIX}etcd"
  CTR_LABEL="infera.e2e.job_tag=${CTR_TAG}"
  CTR_LABELS=(--label "$CTR_LABEL")
  # SLURM's node name, not a container hostname, so dirty-node reports can be excluded.
  export INFERA_E2E_SLURM_NODE="${SLURMD_NODENAME:-$(hostname -s)}"
}

# Without `set -e` an empty $SCRATCH would silently retarget every path at /, so
# an unusable TMPDIR must stop the run here.
init_scratch() {
  local tmp="${TMPDIR:-/tmp}"
  SCRATCH="$(mktemp -d "$tmp/infera-test.XXXXXX" 2>/dev/null)" || SCRATCH=""
  if [ -z "$SCRATCH" ] || [ ! -d "$SCRATCH" ] || [ ! -w "$SCRATCH" ]; then
    echo "FATAL: cannot create a writable scratch dir under '$tmp'." >&2
    echo "       Set TMPDIR to somewhere writable, or take the node out of the pool." >&2
    ls -ld "$tmp" >&2 2>/dev/null
    exit 1
  fi
  mkdir -p "$SCRATCH/hf"
  : > "$SCRATCH/failures.txt"
  chmod 666 "$SCRATCH/failures.txt" 2>/dev/null
  SCRATCH_FLAGS=(-v "$SCRATCH":/scratch -e HF_HOME=/scratch/hf)
}

# In CI worker logs go live to a per-run NFS folder, so they survive scancel and
# SIGKILL; locally to node-local /tmp.
init_log_dir() {
  SHARED_LOG_DIR=""
  E2E_LOG_DIR="/tmp/infera-e2e-logs"
  if [ -n "${GITHUB_ACTIONS:-}" ] || [ "${CI:-}" = "true" ] || [ -n "${INFERA_DISPATCH_LOGDIR:-}" ]; then
    SHARED_LOG_DIR="${INFERA_DISPATCH_LOGDIR:-$HOME/infera-cicd-shared-logs}/$INFERA_E2E_JOB_TAG"
    E2E_LOG_DIR="$SHARED_LOG_DIR"
  fi
  mkdir -p "$E2E_LOG_DIR"
  # In-container writers have another uid; the sticky bit stops runs clobbering each other.
  if [ -n "$SHARED_LOG_DIR" ]; then chmod 1777 "$E2E_LOG_DIR" 2>/dev/null; fi
  SCRATCH_FLAGS+=(-v "$E2E_LOG_DIR":/e2e-logs)
  export INFERA_E2E_LOG_DIR="$E2E_LOG_DIR"
  echo "[scratch] $SCRATCH  (worker logs: $E2E_LOG_DIR${SHARED_LOG_DIR:+ [shared NFS, live]})"
}

init_e2e_flags() {
  E2E_FLAGS=(-e INFERA_E2E_SLURM_NODE="$INFERA_E2E_SLURM_NODE")
  if [ -n "${INFERA_E2E_GFX_ARCH:-}" ]; then
    E2E_FLAGS+=(-e INFERA_E2E_GFX_ARCH="$INFERA_E2E_GFX_ARCH")
  fi
  # pytest -k over the matrices' case ids, e.g. INFERA_E2E_K='GLM-5.2'.
  if [ -n "${INFERA_E2E_K:-}" ]; then
    E2E_FLAGS+=(-e INFERA_E2E_K="$INFERA_E2E_K")
    echo "[e2e] case filter: -k '$INFERA_E2E_K'"
  fi
  [ -n "${INFERA_E2E_MODEL_DIR:-}" ] || return 0
  if [ -d "$INFERA_E2E_MODEL_DIR" ]; then
    E2E_FLAGS+=(-v "$INFERA_E2E_MODEL_DIR":"$INFERA_E2E_MODEL_DIR":ro
                -e INFERA_E2E_MODEL_DIR="$INFERA_E2E_MODEL_DIR")
    echo "[e2e] model dir: $INFERA_E2E_MODEL_DIR (read-only)"
  else
    echo "[e2e] model dir absent here — forwarding '$INFERA_E2E_MODEL_DIR' for the remote run" >&2
  fi
}

# Containers write the scratch as root, so empty it from inside one.
_cleanup_scratch() {
  [ -n "${SCRATCH:-}" ] || return 0
  local img="$IMG_SGLANG"
  docker image inspect "$IMG_VLLM" >/dev/null 2>&1 && img="$IMG_VLLM"
  docker image inspect "$img" >/dev/null 2>&1 && timeout -k 10 120 docker run --rm \
    -v "$SCRATCH":/scratch --entrypoint sh "$img" \
    -c 'rm -rf /scratch/* /scratch/.[!.]* 2>/dev/null' >/dev/null 2>&1
  rm -rf "$SCRATCH" 2>/dev/null || true
}

# Printed at both ends of the run: a failed CI log is read from the bottom.
log_dir_banner() {
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

# A tier that could not run must fail, not pass; INFERA_E2E_ALLOW_SKIP=1 opts out.
#   $1=label  $2=what is wrong  $3=how to fix it
_SKIPPED_TIERS=""
_skip_or_fail() {
  local label="$1" why="$2" fix="$3"
  if [ "${INFERA_E2E_ALLOW_SKIP:-}" = 1 ]; then
    _SKIPPED_TIERS="${_SKIPPED_TIERS:+$_SKIPPED_TIERS, }$label"
    echo "[$label] SKIPPED (INFERA_E2E_ALLOW_SKIP=1): $why" >&2
    return 0
  fi
  echo "[$label] FATAL: $why" >&2
  echo "[$label] fix: $fix" >&2
  echo "[$label] (or INFERA_E2E_ALLOW_SKIP=1 to skip this tier instead of failing)" >&2
  return 1
}
