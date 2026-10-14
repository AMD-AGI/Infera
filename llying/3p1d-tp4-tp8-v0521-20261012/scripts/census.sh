#!/usr/bin/env bash
# Scan server logs for faults and KV-transfer errors. Usage: census.sh SERVER_LOG_DIR [SINCE_ISO]
# SINCE_ISO (e.g. 2026-10-12T19:00:00) limits the scan to lines at or after that docker
# timestamp. Prints one block per log; exits 1 if any FATAL pattern is present.
set -uo pipefail
dir="$1"; since="${2:-}"
fatal_re='Memory access fault|out of memory|OutOfMemory|Traceback \(most recent call last\)|exited with code|scheduler died|transfer failed|Process failed for slice|retry counter exceeded|Session [^ ]+ failed\.|hipIpc[A-Za-z]* failed|NCCL WARN'
status=0
for log in "$dir"/*.log; do
    [[ -e "$log" ]] || continue
    body() { if [[ -n "$since" ]]; then awk -v s="$since" 'substr($0,1,19) >= s' "$log"; else cat "$log"; fi; }
    hcas="$(body | grep -a -o 'Found [0-9]* HCAs' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    ready="$(body | grep -a -c 'fired up and ready')"
    echo "== $(basename "$log") lines=$(body | wc -l) ready=$ready hcas=[$hcas]"
    hits="$(body | grep -a -E "$fatal_re" | grep -a -v -E 'Ignore import error|tokens_exceeded' )"
    if [[ -n "$hits" ]]; then
        status=1
        echo "$hits" | sed -E 's/^[^ ]+ //' | cut -c1-240 | sort | uniq -c | sort -rn | head -12 | sed 's/^/   FATAL /'
    fi
done
exit $status
