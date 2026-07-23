#!/bin/bash
pkill -9 -f sglang.launch_server 2>/dev/null
sleep 5
cd /mnt/vast/c_huggingface
ROLE=decode GMUS="0.80 0.85 0.88 0.90 0.92" OUT=/mnt/vast/c_huggingface/kvcache_gmu_study_20260724_084928 bash kv_gmu_sweep.sh > /mnt/vast/c_huggingface/kvcache_gmu_study_20260724_084928.nohup 2>&1
