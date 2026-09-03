#!/bin/sh
# STEP 5 -- the workset's OWN performance test, on the candidate.
#
# Under the workset's own `protocol`. You do not choose `groups` or
# `iters_per_group`: a comparison between a 5-group sample and a 3-group one is
# not a comparison, and the baseline this stage divides by was measured under
# the workset's protocol.
#
# **Run only after STEP 4 exits 0.** This script refuses otherwise, because the
# ordering is the substance of the rule and not a convention.
#
# Read the `rsd` before the medians. The baseline side is tight (~2% on a
# steady node) and an optimised kernel has measured ~8% round to round on this
# hardware, unexplained since 2026-08-31. A single sample of the loose side is
# not a measurement.
set -eu

INPUTS=""; CANDIDATE=""; OUT=""; CORRECTNESS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --inputs) INPUTS="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --correctness) CORRECTNESS="$2"; shift 2 ;;
    *) echo "usage: 50_performance.sh --inputs <f> --candidate <f> --out <f> [--correctness <f>]" >&2; exit 1 ;;
  esac
done
[ -n "$INPUTS" ] && [ -n "$CANDIDATE" ] && [ -n "$OUT" ] || { echo "--inputs, --candidate and --out are all required" >&2; exit 1; }

# Default to where STEP 4 was told to write, so the guard holds even when the
# caller forgets to name it. An absent report is a refusal, never a pass.
[ -n "$CORRECTNESS" ] || CORRECTNESS="$(dirname "$OUT")/correctness.json"
if [ ! -f "$CORRECTNESS" ]; then
  echo "refusing to time a kernel whose correctness was never established: no $CORRECTNESS." >&2
  echo "Run STEP 4 first. A kernel that is faster and wrong is not a partial success." >&2
  exit 1
fi
PY="${KFO_PYTHON:-python3}"
if ! "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("passed") is True else 1)' "$CORRECTNESS"; then
  echo "refusing to time a kernel that failed correctness (see $CORRECTNESS)." >&2
  exit 1
fi

exec "$(dirname "$0")/run_entrypoint.py" \
  --inputs "$INPUTS" --role performance --candidate "$CANDIDATE" --out "$OUT"
