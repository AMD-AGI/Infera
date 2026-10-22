# Task v0521cap round q1 on crsuse2-m2m-137 (user 10:5x: "135-139挑一台能用的用"; plain ssh, /shared_nfs ro there ->
# NODE_RW_ROOT default /mnt/m2m_nobackup/yihou_v0521dbg). e2g recipe (HiCache 450, decode 0.95) + decode max_running 96
# (24/DP) + prefill mem 0.88, CONC 44, warm protocol (A1 fresh -> flush -> A2 scored). Chain: chain_q1.yihou.sh.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-137 YIHOU_IP=10.245.153.247 YIHOU_JOB= YIHOU_ROUND=q1
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=44 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
