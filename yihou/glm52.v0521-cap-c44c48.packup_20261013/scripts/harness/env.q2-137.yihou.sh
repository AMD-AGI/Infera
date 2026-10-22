# Task v0521cap round q2 on 137: Q1 config (e2g + decode max_running 96 + prefill mem 0.88, decode 0.95) at CONC 48,
# warm protocol. Chain: chain_q2.yihou.sh (= chain_q1 with env/round q2).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-137 YIHOU_IP=10.245.153.247 YIHOU_JOB= YIHOU_ROUND=q2
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
