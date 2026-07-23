#!/bin/bash
pkill -9 -f sglang.launch_server 2>/dev/null
sleep 8
cd /mnt/vast/c_huggingface
ROLE=prefill GMUS="0.85 0.90" OUT=/mnt/vast/c_huggingface/kvcache_gmu_study_20260724_084928 bash kv_gmu_sweep.sh > /mnt/vast/c_huggingface/kvcache_gmu_study_20260724_084928/prefill_run.nohup 2>&1
