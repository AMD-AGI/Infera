# Task v0521cap: C52 leg (4th AgentX run) on the LIVE warm q1 stack on 137 after A3 C48 (22,306). Server args
# identical to q1. Driver: a4_q1c52.yihou.sh -> runs/q1/agentx-c52-A4.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-137 YIHOU_IP=10.245.153.247 YIHOU_JOB= YIHOU_ROUND=q1
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=52 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
