# Task v0521cap round q4 on 137: Q1 config (max_running 96, prefill 0.88) + decode mem 0.96 (+~55k tok/DP; C48 hit 0.99
# on DP1), CONC 48, warm protocol (A1 -> flush -> A2 scored). Chain: chain_q4.yihou.sh (waits for q1 A4 C52, stops q1).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-137 YIHOU_IP=10.245.153.247 YIHOU_JOB= YIHOU_ROUND=q4
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.96 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
