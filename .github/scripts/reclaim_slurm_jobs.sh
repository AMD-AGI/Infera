#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Cancel the SLURM jobs one CI job dispatched, and keep at it until the queue
# confirms they are gone.
#   reclaim_slurm_jobs.sh <job-name-prefix> <job-name-suffix>
#
# The five inline copies this replaces discarded squeue's stderr, which made a
# FAILED query indistinguishable from an EMPTY one: a transient controller error
# read as "nothing to reclaim" and broke the retry loop on its first pass -- that
# error being the only reason the loop existed. On 2026-08-06 four jobs sitting in
# the queue when reclaim ran survived it and held 3 of the reservation's 4 nodes
# until their time limit expired.
set -uo pipefail

prefix="${1:?usage: $0 <job-name-prefix> <job-name-suffix>}"
suffix="${2:?usage: $0 <job-name-prefix> <job-name-suffix>}"
budget="${RECLAIM_TIMEOUT:-120}"
interval="${RECLAIM_INTERVAL:-5}"
me="$(id -un)"
deadline=$(( SECONDS + budget ))

echo "reclaiming SLURM jobs named ${prefix}*${suffix}"

while :; do
  # Exit code, not emptiness, is what separates an unreachable controller from a
  # clean queue; stderr is folded in so the CI log names the failure.
  if ! queue=$(squeue -h -u "$me" -o '%i %j' 2>&1); then
    echo "squeue failed, retrying (this is NOT an empty queue): $queue"
  else
    ids=$(printf '%s\n' "$queue" | awk -v p="$prefix" -v s="$suffix" '
      index($2, p) == 1 && length($2) >= length(s) &&
      substr($2, length($2) - length(s) + 1) == s { print $1 }')
    [ -z "$ids" ] && { echo "confirmed: no ${prefix}*${suffix} jobs left"; exit 0; }
    echo "cancelling: $ids"
    scancel $ids 2>&1 || echo "scancel returned non-zero, retrying"
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    # A leaked job holds a reserved GPU node until its time limit, so this has to
    # be findable in the log rather than inferred later from a reservation that
    # looks idle and is not.
    echo "::error::could not confirm reclaim of ${prefix}*${suffix} within ${budget}s; check for leaked SLURM jobs"
    squeue -u "$me" -o '%.10i %.44j %.2t %.10M %R' 2>&1 || true
    exit 1
  fi
  sleep "$interval"
done
