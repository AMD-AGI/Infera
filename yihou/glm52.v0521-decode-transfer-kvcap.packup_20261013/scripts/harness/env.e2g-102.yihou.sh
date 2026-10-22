# Round e2g on 102 (R041): exact CONTROL REPLICATE of the 21,696 run (R030, e2c stack @C40) on a FRESH
# (cold) stack, to separate node drift from config effects (R040). = env.e2c-102.yihou.sh except
# ROUND=e2g, CONC=40, job 207495 (follow-on hold on the same node). Same image (mc4340), same mounts
# (v2-regbase via config.yihou.mc4340.sh), index_share FALSE, chunk 32768 default, no YIHOU_* transfer
# env, no THREAD_POOL override. Chain: mc4340 -> E2-target -> v0521-base.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=207495 YIHOU_ROUND=e2g
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=40 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
