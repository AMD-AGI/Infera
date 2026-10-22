# Task v0521cap2 round pc48e on crsuse2-m2m-274 (oai job 211171), user 2026-10-13 ~15:00 "跑": CONTROL for pc48b.
# Same as pc48b (HiCache 370 GB write_back, chunk 512, decode mem 0.96, decode max_running 96, prefill mem 0.88, C48)
# but decode allocation policy = early (default; no prefill_complete flag, no attndp hook mount). Chain: 48 48.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-274 YIHOU_IP=10.245.157.34 YIHOU_JOB=211171 YIHOU_ROUND=pc48e
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n2.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 PREFILL_HICACHE_SIZE=370 PREFILL_HICACHE_WRITE_POLICY=write_back DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.96 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
