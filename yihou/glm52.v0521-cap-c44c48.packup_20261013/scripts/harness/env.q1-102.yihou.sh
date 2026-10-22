# Task v0521cap round q1 (PLAN v2): e2g recipe (HiCache 450) + decode max_running 96 (24/DP) + prefill mem 0.88,
# CONC 44, warm protocol. Node 102 template (retarget YIHOU_NODE/IP/JOB + image config for the 2nd oai node).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=207495 YIHOU_ROUND=q1
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=44 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
