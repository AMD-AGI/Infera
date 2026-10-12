#!/usr/bin/env bash
# R055: cross-stream event-wait start latency vs HW-queue / signal settings. One idle GPU (default 4), one container
# at a time, ~20-30 s per arm, 9 arms (<= ~5 min). Modes idle/third/decode inside each arm (xstream_wait2.yihou.py).
# usage: run_xstream_wait2.yihou.sh <node> <job> [gpu=4] [image]
set -uo pipefail
node=${1:?node}; job=${2:?job}; G=${3:-4}; IMG=${4:-infera-sglang:v0521-yihou-1010-mc4340}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
out=$D/diag/bprime/out.xstream2.$(date -u +%Y%m%d-%H%M%S).yihou; mkdir -p "$out"
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH SPUR_SSH_JOBS="$node:$job"
ARMS=(
  "hwq_def|"
  "hwq1|-e GPU_MAX_HW_QUEUES=1"
  "hwq2|-e GPU_MAX_HW_QUEUES=2"
  "hwq16|-e GPU_MAX_HW_QUEUES=16"
  "hwq_def_intr0|-e HSA_ENABLE_INTERRUPT=0"
  "hwq_def_cpuwait0|-e ROC_CPU_WAIT_FOR_SIGNAL=0"
  "hwq_def_actwait50|-e ROC_ACTIVE_WAIT_TIMEOUT=50"
  "hwq16_cpuwait0|-e GPU_MAX_HW_QUEUES=16 -e ROC_CPU_WAIT_FOR_SIGNAL=0"
  "hwq_def_qlog|-e AMD_LOG_LEVEL=3 -e AMD_LOG_MASK=0x10 -e XS_REPS=3"
)
for arm in "${ARMS[@]}"; do
  name=${arm%%|*}; E=${arm#*|}
  ssh "$node" "timeout 150 docker run --rm --name yihou-xstream2 --device /dev/kfd --device /dev/dri --group-add video --network none \
    -e HIP_VISIBLE_DEVICES=$G -e PYTHONUNBUFFERED=1 $E -v $D:$D:ro --entrypoint python3 $IMG $D/diag/bprime/xstream_wait2.yihou.py" \
    > "$out/$name.log" 2>&1 < /dev/null
  echo "== $name"; grep -a '^RESULT\|^streams' "$out/$name.log" | cut -c1-230 || tail -3 "$out/$name.log"
done
# queue-mask check: what the LOG_QUEUE-only arm printed (Created SWq / Selected queue lines and nothing else)
echo "== qlog line kinds:"; grep -a ':3:' "$out/hwq_def_qlog.log" | sed -E 's/.*\] //' | cut -c1-60 | sort | uniq -c | sort -rn | head -10
echo "[xstream2] $out"
