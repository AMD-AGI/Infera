#!/usr/bin/env bash
# exit 0 when all 8 GPUs on 102 < 400 MiB and no containers; prints the job used
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH
for i in $(seq 1 120); do
  for j in 207495 202546; do
    st=$(squeue -j $j -h -o %T 2>/dev/null)
    [[ "$st" == RUNNING ]] || continue
    out=$(SPUR_SSH_JOBS=crsuse2-m2m-102:$j timeout 60 ssh crsuse2-m2m-102 'echo C=$(docker ps -q | wc -l); amd-smi metric -m 2>/dev/null | grep USED_VRAM | awk "{print \$2}" | paste -sd" "' 2>/dev/null)
    c=$(echo "$out" | sed -n 's/^C=//p'); v=$(echo "$out" | tail -1)
    echo "$(date -u +%T) job=$j containers=$c vram=[$v]"
    n=$(echo $v | wc -w); mx=$(echo $v | tr ' ' '\n' | sort -n | tail -1)
    if [[ "$c" == 0 && "$n" == 8 && "$mx" -lt 400 ]]; then echo "IDLE job=$j"; exit 0; fi
    break
  done
  sleep 60
done
exit 1
