#!/bin/bash
# Copy the GLM-5.2-MXFP4 weights from /shared_nfs to this node's NVMe (same path on every node), so an engine start does
# not page-fault 438 GB in from NFS (11-16 v3-219: main weights 330-540 s, NextN 325 s, vs ~3-5 min stack starts on AUS).
# Usage on a node: stage_model.sh [SRC] [DST]; skips the copy when DST/.complete records the same total size.
SRC=${1:-/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4}
DST=${2:-/mnt/m2m_nobackup/liyingli_pdr/models/GLM-5.2-MXFP4}
h=$(hostname -s)
want=$(find "$SRC" -maxdepth 1 -type f -printf '%s\n' | awk '{s += $1} END {print s}')
if [[ -f "$DST/.complete" && "$(cat "$DST/.complete")" == "$want" ]]; then echo "$h: $DST already staged ($want bytes)"; exit 0; fi
mkdir -p "$DST"
t0=$(date +%s)
find "$SRC" -maxdepth 1 -type f -printf '%f\n' | xargs -P 24 -I{} cp --preserve=timestamps "$SRC/{}" "$DST/{}"
got=$(find "$DST" -maxdepth 1 -type f ! -name .complete -printf '%s\n' | awk '{s += $1} END {print s}')
dt=$(( $(date +%s) - t0 ))
if [[ "$got" == "$want" ]]; then
    echo "$want" > "$DST/.complete"
    echo "$h: staged $((got / 1000000000)) GB in ${dt} s ($((got / 1000000 / (dt > 0 ? dt : 1))) MB/s)"
else
    echo "$h: staging incomplete ($got of $want bytes after ${dt} s)"; exit 1
fi
