# Round e1b on 136 (R020): E1 relaxed on the Mooncake #4340 image + REGBASE mount (repro's
# config.yihou.mc4340.sh; it hard-sets IMAGE and the per-node EXPECTED_IMAGE_ID), --pid host +
# default IPC, legacy OFF, debug agent OFF. Plain ssh, NODE_RW_ROOT (ro NFS). MC4340_BASE_CONFIG
# rides in YIHOU_OVERRIDES so the node-side engine.sh / agentx_bench / stop re-source the same chain.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-136 YIHOU_IP=10.245.154.168 YIHOU_JOB= YIHOU_ROUND=e1b
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E1-relaxed.sh ENGINE_PID_HOST=1 IPC_MODE_LEGACY=0 DEBUG_AGENT=0"
