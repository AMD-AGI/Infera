#!/usr/bin/env bash
# Purpose: One py-spy window on every scheduler process of a decode container:
#   `record` as raw collapsed stacks with idle frames kept (so waits in
#   collectives and event syncs are visible), then one `dump` per process.
# Usage: pyspy_window.sh CONTAINER OUT_DIR TAG [SECONDS]   (run on the decode node)
# Artifacts: OUT_DIR/<TAG>-procs.txt, OUT_DIR/<TAG>-<proc>.raw,
#   OUT_DIR/<TAG>-<proc>.log, OUT_DIR/<TAG>-dump.txt
# --nonblocking: the target is never paused, so sampling does not stretch steps.
set -uo pipefail
c="$1"; out="$2"; tag="$3"; secs="${4:-90}"
spy=/opt/venv/bin/py-spy
mkdir -p "$out"
docker exec "$c" sh -c 'for p in /proc/[0-9]*; do
    n=$(tr "\0" " " <"$p/cmdline" 2>/dev/null)
    case "$n" in sglang::scheduler*) echo "${p#/proc/} ${n%% *}";; esac
done' >"$out/$tag-procs.txt"
[[ -s "$out/$tag-procs.txt" ]] || { echo "no scheduler processes in $c" >&2; exit 1; }

while read -r pid name; do
    docker exec "$c" "$spy" record --pid "$pid" --duration "$secs" --rate 100 \
        --idle --threads --nonblocking --format raw -o "/tmp/$tag-${name#sglang::}.raw" \
        >"$out/$tag-${name#sglang::}.log" 2>&1 &
done <"$out/$tag-procs.txt"
wait

while read -r pid name; do
    f="$tag-${name#sglang::}.raw"
    docker exec "$c" cat "/tmp/$f" >"$out/$f" && docker exec "$c" rm -f "/tmp/$f"
    echo "== $name ($pid)"
    docker exec "$c" "$spy" dump --pid "$pid" --nonblocking 2>&1
done <"$out/$tag-procs.txt" >"$out/$tag-dump.txt"
