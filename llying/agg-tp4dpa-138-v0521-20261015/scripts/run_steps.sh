#!/usr/bin/env bash
# Run the commands of a steps file in order (one per line, relative to this kit; '#' starts a
# comment), stopping at the first failure. Progress goes to runtime/runs/steps.<name>.log and the
# number of finished steps to runtime/runs/steps.<name>.done, so a rerun resumes after them.
# Usage: run_steps.sh STEPS_FILE   (start it with start_steps.sh, which detaches it on the node)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
AGG="$(cd "$HERE/.." && pwd)"
steps="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
name="$(basename "$steps" .steps)"
log="$AGG/runtime/runs/steps.$name.log"
done_file="$AGG/runtime/runs/steps.$name.done"
mkdir -p "$AGG/runtime/runs"
say() { echo "$(date -u +%FT%TZ) $*" >> "$log"; }
finished="$(cat "$done_file" 2>/dev/null || echo 0)"
n=0
cd "$AGG"
while IFS= read -r line; do
    line="${line%%#*}"
    [[ -z "${line// }" ]] && continue
    n=$((n + 1))
    (( n <= finished )) && continue
    say "step $n start: $line"
    bash -c "$line" >> "$log" 2>&1
    rc=$?
    if (( rc != 0 )); then
        say "step $n FAILED rc=$rc: $line"
        exit "$rc"
    fi
    echo "$n" > "$done_file"
    say "step $n done"
done < "$steps"
say "all $n steps done"
