#!/usr/bin/env bash
# Copy the DeepSeek-V4.1-Flash weights from /shared_nfs to this node's NVMe, so an engine start does not page-fault
# ~476 GB in from NFS. Skips the copy when $MODEL/.complete records the same total size.
set -uo pipefail
source "$(dirname "$0")/env.sh"
h=$(hostname -s)
want=$(find "$MODEL_SRC" -type f ! -path '*/.cache/*' -printf '%s\n' | awk '{s += $1} END {print s}')
if [[ -f "$MODEL/.complete" && "$(cat "$MODEL/.complete")" == "$want" ]]; then echo "$h: $MODEL already staged ($want bytes)"; exit 0; fi
mkdir -p "$MODEL"
t0=$(date +%s)
(cd "$MODEL_SRC" && find . -type d ! -path './.cache*' -printf '%P\n' | while read -r d; do [ -n "$d" ] && mkdir -p "$MODEL/$d"; done)
(cd "$MODEL_SRC" && find . -type f ! -path './.cache/*' -printf '%P\n') | xargs -P 24 -I{} cp --preserve=timestamps "$MODEL_SRC/{}" "$MODEL/{}"
got=$(find "$MODEL" -type f ! -name .complete -printf '%s\n' | awk '{s += $1} END {print s}')
dt=$(( $(date +%s) - t0 ))
if [[ "$got" == "$want" ]]; then
    echo "$want" > "$MODEL/.complete"
    echo "$h: staged $((got / 1000000000)) GB in ${dt} s ($((got / 1000000 / (dt > 0 ? dt : 1))) MB/s)"
else
    echo "$h: staging incomplete ($got of $want bytes after ${dt} s)"; exit 1
fi
