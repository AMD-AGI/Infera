#!/bin/bash
MODE=${1:-dmabuf}
SIG=/mnt/vast/c_huggingface/sig
VMM=/sys/kernel/debug/dri/0000:75:00.0/amdgpu_vram_mm
OUT=/mnt/vast/c_huggingface/ro_$MODE.log
rm -f $SIG/PHASE $SIG/GO; mkdir -p $SIG; : > $OUT
# dmesg baseline marker (non-follow snapshot before)
DBEFORE=$(dmesg | tail -1)

( for k in $(seq 1 6); do
    for w in $(seq 1 400); do [ -f $SIG/PHASE ] && break; sleep 0.1; done
    [ -f $SIG/PHASE ] || break
    ph=$(cat $SIG/PHASE)
    ttm=$(grep -m1 'total:' $VMM | sed -E 's/.*free:[[:space:]]*([0-9]+)MiB.*/\1/')
    ttmG=$(awk "BEGIN{printf \"%.2f\", $ttm/1024.0}")
    echo "TTM_free=${ttmG}G  ||  $ph" >> $OUT
    touch $SIG/GO
    for w in $(seq 1 100); do [ -f $SIG/GO ] || break; sleep 0.1; done
  done ) &
SAMP=$!
docker exec -e HIP_VISIBLE_DEVICES=0 dmabuf_probe bash -lc "cd /root && ./mvp $MODE"
kill $SAMP 2>/dev/null; wait 2>/dev/null
echo "================ $MODE REORDERED (TTM aligned) ================"
cat $OUT
echo
echo "================ NEW dmesg since run start (kfd/pin/reserve/gpu_mem) ================"
# print dmesg lines after the baseline marker, filtered
dmesg | awk -v b="$DBEFORE" 'f{print} $0==b{f=1}' | grep -iE 'kfd|amdkfd|reserve|pin|limit|evict|ttm|dmabuf|out of memory|gpu_mem|oom' | tail -40
echo "[[dmesg tail total new lines above]]"
