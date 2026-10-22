# Task v0521cap2 round wts48 (user 2026-10-14 ~03:30 "找hi-cache=450最好的实验结果，改用selective throughput"):
# best HiCache-450 result = q1 C48 A3 22,306 (q1 config: e2g + decode max_running 96 + prefill mem 0.88, decode mem 0.95,
# HiCache 450 GB write_through). Only change: PREFILL_HICACHE_WRITE_POLICY=write_through_selective
# (sglang v0.5.21: write_through_threshold 2 instead of 1 -> a radix node is backed up to host only after its 2nd hit;
# unified_radix_cache.py:516-517, unified_tree_core.py:1078-1084). Chain: chain_cap2 48 48 52.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-089 YIHOU_IP=10.245.145.201 YIHOU_JOB=213048 YIHOU_ROUND=wts48dry
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n4.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 PREFILL_HICACHE_WRITE_POLICY=write_through_selective ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
