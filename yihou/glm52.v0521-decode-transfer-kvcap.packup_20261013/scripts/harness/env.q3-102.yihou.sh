# Task v0521cap round q3 on 102 (after pc40c): Q1 config (decode max_running 96 + prefill mem 0.88) + runshare 0.5
# (decode in-flight KV cap, user rule), CONC 44, warm protocol. Same node as k1/e2g -> also a same-node check of Q1.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=207495 YIHOU_ROUND=q3
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.runshare.sh
export YIHOU_OVERRIDES="RUNSHARE_BASE_CONFIG=$H/config.yihou.mc4340.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh RUNNING_KV_SHARE=0.5 CONC=44 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
