#!/bin/bash
# Screen a held job: needs BOTH the merged-mtp image AND actually-free GPUs.
J="$1"
spur exec "$J" bash -c '
H=$(hostname)
IMG=$(docker images --format "{{.Repository}}:{{.Tag}}" 2>/dev/null | grep -cE "^infera/engine-sglang:merged-mtp$")
USED=$(rocm-smi --showmeminfo vram 2>/dev/null | grep -oE "Used Memory \(B\): [0-9]+" | grep -oE "[0-9]+$" | sort -rn | head -1)
USED_GB=$(( ${USED:-0} / 1000000000 ))
echo "$H img=$IMG maxUsedGB=$USED_GB"
' 2>/dev/null | tail -1
