#!/bin/sh
# STEP 4 -- the workset's OWN correctness test, on the candidate.
#
# Not forge's internal SNR gate: that is forge grading its own homework, and it
# is recorded separately under `evidence.forge`. The workset is the oracle,
# m3's `check_workset_runs` has already executed it on this hardware, and this
# step runs the same entrypoint with the candidate selected.
#
# **This step is separate from STEP 5 and runs before it, and that ordering is
# the point rather than a convenience.** A kernel that is faster and wrong must
# never reach a timing loop. Correctness is not a percentage: a kernel that is
# right on two shapes of three is wrong.
#
# Exit 0 = every declared correctness case passed. Anything else ends the run;
# go to STEP 6 and write the handoff with the failure in it.
set -eu

INPUTS=""; CANDIDATE=""; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --inputs) INPUTS="$2"; shift 2 ;;
    --candidate) CANDIDATE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "usage: 40_correctness.sh --inputs <f> --candidate <f> --out <f>" >&2; exit 1 ;;
  esac
done
[ -n "$INPUTS" ] && [ -n "$CANDIDATE" ] && [ -n "$OUT" ] || { echo "--inputs, --candidate and --out are all required" >&2; exit 1; }
[ -f "$CANDIDATE" ] || { echo "no candidate kernel at $CANDIDATE" >&2; exit 1; }

exec "$(dirname "$0")/run_entrypoint.py" \
  --inputs "$INPUTS" --role correctness --candidate "$CANDIDATE" --out "$OUT"
