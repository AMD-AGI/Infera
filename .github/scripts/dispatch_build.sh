#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Build+push an engine image on a SLURM node (the runner has no docker).
# Token reaches the remote via srun --export (Spur does not forward stdin).
#   .github/scripts/dispatch_build.sh <sglang|vllm|atom|kvd|server>
set -uo pipefail

engine="${1:?usage: dispatch_build.sh <engine>}"
BTP="$(dirname "$(readlink -f "$0")")/build_test_push.sh"

# Capture the token, then drop it from the runner env until srun exports it to the
# remote step only (avoid leaving it in --export=ALL for unrelated local tools).
token="${INFERAIMAGE_DOCKERHUB_TOKEN:-}"
unset INFERAIMAGE_DOCKERHUB_TOKEN
[ -n "$token" ] || { echo "INFERAIMAGE_DOCKERHUB_TOKEN is empty" >&2; exit 1; }

part="${INFERA_E2E_SLURM_PARTITION:-amd-spur}"
resv=(); [ -n "${INFERA_E2E_RESERVATION:-}" ] && resv=(--reservation="$INFERA_E2E_RESERVATION")
jobname="infera-build-${INFERA_E2E_JOB_TAG:-$engine}"

# $out: srun client banners (job id, "running on <node>", ...). Remote build
# output goes to $logf on shared NFS in CI so tail -F can stream it live to GHA
# (buffered srun stdout does not show up in the Actions log).
out="${TMPDIR:-/tmp}/.dispatch-build-${INFERA_E2E_JOB_TAG:-$engine}.out"
shared=0 logdir="" logf="" tailf="$out"
if [ -n "${GITHUB_ACTIONS:-}" ] || [ "${CI:-}" = "true" ] || [ -n "${INFERA_DISPATCH_LOGDIR:-}" ]; then
  shared=1
  logdir="${INFERA_DISPATCH_LOGDIR:-$HOME/infera-cicd-shared-logs}"
  mkdir -p "$logdir" 2>/dev/null || true
  logf="$logdir/build-${INFERA_E2E_JOB_TAG:-$engine}-$$.log"
  tailf="$logf"
fi

remote=(bash "$BTP" ship "$engine")
[ "$shared" -eq 1 ] && \
  remote=(bash -c 'lf="$1"; shift; exec >"$lf" 2>&1; exec bash "$@"' _ "$logf" "$BTP" ship "$engine")

# Vars the remote build needs; token is injected in a subshell (see _run_srun).
_srun_exports="INFERAIMAGE_DOCKERHUB_TOKEN,IMAGE,ID,PR,PATH,HOME,USER,LOGNAME,TMPDIR"

_run_srun() {
  # Spur srun does not forward stdin — export the token explicitly for this step.
  (
    export INFERAIMAGE_DOCKERHUB_TOKEN="$token"
    srun --export="$_srun_exports" -N1 -p "$part" -t 02:00:00 \
      -J "$jobname" "${resv[@]}" \
      "${remote[@]}"
  )
}

if [ "$shared" -eq 0 ]; then
  _run_srun
  exit $?
fi

echo "[build] streaming remote output below (live via $tailf):"
: > "$out"; : > "$logf"
# stdbuf -oL line-buffers tail to our stdout; -F follows by name + retry
# (tolerates the remote truncating on open, and polls over NFS).
stdbuf -oL tail -n +1 -F "$tailf" 2>/dev/null &
tailpid=$!
# Background srun + `wait`: a foreground srun would defer SIGTERM handling until
# it returns, so a CI cancel could kill us before release.yml reclaims the job.
_run_srun > "$out" 2>&1 &
srunpid=$!
wait "$srunpid"; prc=$?
# Let tail catch the final (NFS-propagated) lines before stopping it.
sleep 3; kill "$tailpid" 2>/dev/null; wait "$tailpid" 2>/dev/null || true
find "$logdir" -maxdepth 1 -type f -name '*.log' \
  -mmin "+${INFERA_DISPATCH_LOG_TTL_MIN:-14400}" -delete 2>/dev/null
exit "$prc"
