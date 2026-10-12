#!/usr/bin/env bash
# R054: cross-stream wait latency vs GPU_MAX_HW_QUEUES on ONE idle GPU (default GPU 4). ~1-2 min total.
# usage: run_xstream_wait.yihou.sh <node> <job> [gpu=4] [image]
set -uo pipefail
node=${1:?node}; job=${2:?job}; G=${3:-4}; IMG=${4:-infera-sglang:v0521-yihou-1010-mc4340}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
out=$D/diag/bprime/out.xstream.$(date -u +%Y%m%d-%H%M%S).yihou; mkdir -p "$out"
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS="$node:$job"
for cfg in "hwq_default|" "hwq1|-e GPU_MAX_HW_QUEUES=1" "hwq2|-e GPU_MAX_HW_QUEUES=2" "hwq16|-e GPU_MAX_HW_QUEUES=16" \
           "hwq_default_spin20|-e XS_SPIN_MS=20" "hwq1_spin20|-e GPU_MAX_HW_QUEUES=1 -e XS_SPIN_MS=20"; do
  name=${cfg%%|*}; E=${cfg#*|}
  ssh "$node" "timeout 180 docker run --rm --name yihou-xstream --device /dev/kfd --device /dev/dri --group-add video --network none \
    -e HIP_VISIBLE_DEVICES=$G -e PYTHONUNBUFFERED=1 $E -v $D:$D:ro --entrypoint python3 $IMG $D/diag/bprime/xstream_wait.yihou.py" \
    > "$out/$name.log" 2>&1 < /dev/null
  echo "== $name: $(grep -a '^RESULT' "$out/$name.log" || tail -2 "$out/$name.log")"
done
echo "[xstream] $out"
