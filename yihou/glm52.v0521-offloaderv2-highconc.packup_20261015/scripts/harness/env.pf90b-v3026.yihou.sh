# Task v0521cap2 round pf90b on crsuse2-m2m-v3-026 (burst job 217126, partition amd-spur-v2; image 9d755dd6): pf90 config, NO offload.
# Same-partition reference for ofg3b (v3-020) at C48 and C56; q1 config + PREFILL_MEM_FRACTION=0.90, sim 3.61.
# Chain: chain_cap2 48 48 56.
# Chain: chain_cap2 48 48 52.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-v3-026 YIHOU_IP=10.245.152.176 YIHOU_JOB=217126 YIHOU_ROUND=pf90b
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n6.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
