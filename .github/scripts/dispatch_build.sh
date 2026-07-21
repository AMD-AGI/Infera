#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Build+push an engine image on a SLURM node (the runner has no docker). Spur's
# srun propagates the caller's env, so the token is set inline on the srun call.
#   .github/scripts/dispatch_build.sh <sglang|vllm|atom|kvd|server>
set -uo pipefail

engine="${1:?usage: dispatch_build.sh <engine>}"
BTP="$(dirname "$(readlink -f "$0")")/build_test_push.sh"

# Keep the token out of this script's env; _run_srun re-exports it for srun only.
token="${INFERAIMAGE_DOCKERHUB_TOKEN:-}"
unset INFERAIMAGE_DOCKERHUB_TOKEN
[ -n "$token" ] || { echo "INFERAIMAGE_DOCKERHUB_TOKEN is empty" >&2; exit 1; }

part="${INFERA_E2E_SLURM_PARTITION:-amd-spur}"
resv=(); [ -n "${INFERA_E2E_RESERVATION:-}" ] && resv=(--reservation="$INFERA_E2E_RESERVATION")
jobname="infera-build-${INFERA_E2E_JOB_TAG:-$engine}"

# $out holds srun client banners; remote build output goes to $logf on shared
# NFS so tail -F can stream it live to GHA (buffered srun stdout would not show).
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

_run_srun() {
  (
    export INFERAIMAGE_DOCKERHUB_TOKEN="$token"
    srun -N1 -p "$part" -t 02:00:00 \
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
# -F follows by name + retry, tolerating the remote truncating on open over NFS.
stdbuf -oL tail -n +1 -F "$tailf" 2>/dev/null &
tailpid=$!
# Background srun + `wait` so a CI cancel's SIGTERM is handled promptly.
_run_srun > "$out" 2>&1 &
srunpid=$!
wait "$srunpid"; prc=$?
sleep 3; kill "$tailpid" 2>/dev/null; wait "$tailpid" 2>/dev/null || true
# srun banners/errors land in $out (never in the NFS tail); surface them so a
# failure before the remote runs is still diagnosable in GHA.
if [ -s "$out" ]; then
  echo "[build] srun client output:"; sed 's/^/  /' "$out"
fi
find "$logdir" -maxdepth 1 -type f -name '*.log' \
  -mmin "+${INFERA_DISPATCH_LOG_TTL_MIN:-14400}" -delete 2>/dev/null
exit "$prc"
