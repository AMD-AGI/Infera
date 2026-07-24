#!/bin/bash
MODE=$1
VMM=/sys/kernel/debug/dri/0000:75:00.0/amdgpu_vram_mm
LOG=/mnt/vast/c_huggingface/vmm_$MODE.log
: > $LOG
( for i in $(seq 1 90); do
    # the line we want contains "total:" and "free:"
    fl=$(grep -m1 'total:' $VMM 2>/dev/null)
    echo "$(date +%H:%M:%S) $fl" >> $LOG
    sleep 1
  done ) &
SAMP=$!
docker exec -e HIP_VISIBLE_DEVICES=0 dmabuf_probe bash -lc "cd /root && ./mvp $MODE"
kill $SAMP 2>/dev/null
