# Task v0521cap Track B round pc40d on 102: e2g recipe + runshare v2 (share 0.5, skip + 5 s anti-starvation), C40,
# burst smoke (230k tok) -> A1 -> flush -> A2 (scored vs e2g A2 22,114; pc40c v1 A2 = 21,271).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=207495 YIHOU_ROUND=pc40d
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.runshare2.sh
export YIHOU_OVERRIDES="RUNSHARE_BASE_CONFIG=$H/config.yihou.mc4340n.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh RUNNING_KV_SHARE=0.5 CONC=40 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
