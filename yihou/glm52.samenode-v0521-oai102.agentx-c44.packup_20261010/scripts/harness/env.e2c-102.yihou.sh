# Round e2c on 102 (prepared, NOT launched): the E2 TARGET on the e2b4 chain -- fused DSA indexer
# back to the v0521 default (flag absent + writer env absent), everything else = e2b4:
# mc4340 + REGBASE, 450 GB + CHUNK_GB=512, decode mem 0.95, CONC 44, DECODE_PID_OFFSET=30000,
# private PID ns + default IPC, legacy OFF, agent OFF.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=202546 YIHOU_ROUND=e2c
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
