# Task v0521cap2 round tb56n on crsuse2-m2m-085 (oai job 211169), user 2026-10-13 ~11:55 "要": NO-PATCH CONTROL for tb56.
# Identical to tb56 (q1 config: decode max_running 96, prefill mem 0.88, decode mem 0.95; HiCache ratio 1.5 = 154 GB)
# minus the runshare v2 decode patch (no decode.py mount, no YIHOU_DECODE_RUNNING_KV_SHARE env). Chain: 56 56 64.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-085 YIHOU_IP=10.245.153.145 YIHOU_JOB=211169 YIHOU_ROUND=tb56n
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n2.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=56 PREFILL_HICACHE_SIZE=0 PREFILL_HICACHE_RATIO=1.5 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
