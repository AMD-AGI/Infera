# Round e1c on 102: the e1b chain moved to 102 (136 contended): mc4340 image + REGBASE mount,
# MC4340_BASE_CONFIG=E1-relaxed, private PID ns + default IPC + DECODE_PID_OFFSET=30000 (R022), legacy OFF, agent OFF.
# WARNING (first-hand, e2b3 07:23): spur-authz on 102 DENIES `docker run --pid=host`.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=202546 YIHOU_ROUND=e1c
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E1-relaxed.sh ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
