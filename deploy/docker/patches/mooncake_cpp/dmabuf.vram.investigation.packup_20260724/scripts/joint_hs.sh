#!/bin/bash
MODE=$1
SIG=/mnt/vast/c_huggingface/sig
VMM=/sys/kernel/debug/dri/0000:75:00.0/amdgpu_vram_mm
OUT=/mnt/vast/c_huggingface/hs_$MODE.log
rm -f $SIG/PHASE $SIG/GO; mkdir -p $SIG; : > $OUT
# host sampler loop: for up to 6 phases
( for k in $(seq 1 6); do
    # wait for a PHASE marker (max 30s)
    for w in $(seq 1 300); do [ -f $SIG/PHASE ] && break; sleep 0.1; done
    [ -f $SIG/PHASE ] || { echo "sampler: no PHASE, exit" >> $OUT; break; }
    ph=$(cat $SIG/PHASE)
    ttm=$(grep -m1 'total:' $VMM | sed -E 's/.*free:[[:space:]]*([0-9]+)MiB.*/\1/')
    ttmG=$(awk "BEGIN{printf \"%.2f\", $ttm/1024.0}")
    echo "TTM_free=${ttmG}G  ||  $ph" >> $OUT
    touch $SIG/GO
    # wait for MVP to consume GO before looking for next PHASE
    for w in $(seq 1 100); do [ -f $SIG/GO ] || break; sleep 0.1; done
  done ) &
SAMP=$!
docker exec -e HIP_VISIBLE_DEVICES=0 dmabuf_probe bash -lc "cd /root && ./mvp $MODE"
wait $SAMP 2>/dev/null
echo "================ $MODE handshake-aligned result ================"
cat $OUT
