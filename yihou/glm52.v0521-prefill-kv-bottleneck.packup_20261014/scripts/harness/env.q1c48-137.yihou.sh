# Task v0521cap: C48 leg on the LIVE warm q1 stack on 137 (3rd AgentX run; server args identical to q1, CONC is
# bench-only). Driver: a3_q1c48.yihou.sh (after q1 chain done: flush -> bench BENCH_TAG=A3 -> runs/q1/agentx-c48-A3).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-137 YIHOU_IP=10.245.153.247 YIHOU_JOB= YIHOU_ROUND=q1
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
