#!/usr/bin/env bash
# Drive the remaining fixlen rounds back-to-back. Runs ON the prefill node.
#
# Sequential by necessity: all 8 rounds share ONE server, so two rounds at once
# would measure each other. Between rounds nothing is reset -- the server is
# deliberately frozen, and the radix cache carrying over is part of the
# deployment being measured, not a contaminant (each round's own kvd
# before/after snapshot bounds what it inherited).
#
#   ROUNDS="p50:32 p50:64 ..." sweep.sh
set -u
W=/mnt/vast/c_huggingface/bench_20260801
ROUNDS="${ROUNDS:-p50:32 p50:64 p50:128 p90:1 p90:32 p90:64 p90:128}"

for r in $ROUNDS; do
  pair="${r%%:*}"; c="${r##*:}"
  echo "############################################################"
  echo "### $(date -u +%H:%M:%S)  PAIR=$pair C=$c"
  echo "############################################################"
  PAIR="$pair" C="$c" bash "$W/scripts/fixlen_round.sh" 2>&1 \
    | grep -vE '^\s*$' | tail -45
  echo
done
echo "=== SWEEP DONE $(date -u) ==="
