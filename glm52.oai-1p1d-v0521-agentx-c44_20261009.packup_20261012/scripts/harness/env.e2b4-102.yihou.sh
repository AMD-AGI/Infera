# Round e2b4 on 102: E2b (hicache-size 450 + CHUNK_GB=512, decode mem 0.95, CONC 44, fused indexer OFF)
# on the e1c chain: mc4340 image + REGBASE mount, private PID ns + default IPC + DECODE_PID_OFFSET=30000.
# Written by leader 09:10 (harness-v2 unresponsive on this item). spur-authz denies --pid=host on 102.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=202546 YIHOU_ROUND=e2b4
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2b.sh ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
