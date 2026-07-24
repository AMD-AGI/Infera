#!/bin/bash
# run dmabuf MVP (handshake-synced) while capturing KFD accounting from dmesg.
MODE=${1:-dmabuf}
SIG=/mnt/vast/c_huggingface/sig
VMM=/sys/kernel/debug/dri/0000:75:00.0/amdgpu_vram_mm
OUT=/mnt/vast/c_huggingface/hs_$MODE.log
DM=/mnt/vast/c_huggingface/dmesg_$MODE.log
rm -f $SIG/PHASE $SIG/GO; mkdir -p $SIG; : > $OUT; : > $DM

# follow dmesg with human timestamps, filter KFD accounting keywords
( dmesg -wT 2>/dev/null | grep --line-buffered -iE 'kfd|amdkfd|reserve|pin|limit|evict|ttm|dmabuf|p2p' >> $DM ) &
DMESG=$!

# handshake sampler: bind TTM sample to each phase, also mark dmesg with phase
( for k in $(seq 1 6); do
    for w in $(seq 1 300); do [ -f $SIG/PHASE ] && break; sleep 0.1; done
    [ -f $SIG/PHASE ] || break
    ph=$(cat $SIG/PHASE)
    ttm=$(grep -m1 'total:' $VMM | sed -E 's/.*free:[[:space:]]*([0-9]+)MiB.*/\1/')
    ttmG=$(awk "BEGIN{printf \"%.2f\", $ttm/1024.0}")
    echo "=====PHASE MARK: $ph  (TTM_free=${ttmG}G) $(date +%T)" >> $DM
    echo "TTM_free=${ttmG}G  ||  $ph" >> $OUT
    touch $SIG/GO
    for w in $(seq 1 100); do [ -f $SIG/GO ] || break; sleep 0.1; done
  done ) &
SAMP=$!

docker exec -e HIP_VISIBLE_DEVICES=0 dmabuf_probe bash -lc "cd /root && ./mvp $MODE"
sleep 2
kill $SAMP $DMESG 2>/dev/null
echo "================ $MODE TTM-aligned ================"
cat $OUT
echo
echo "================ KFD / pin / reserve dmesg during run ================"
cat $DM
