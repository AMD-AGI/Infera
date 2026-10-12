# Round e2j on 102 (R050, PREPARED, NOT armed): e2g control config + decode-only DSA host-variant mounts
# (YIHOU_DSA_HOST_VARIANT=1) + --cuda-graph-bs-decode 1..16. Protocol like e2g: R1 fresh (reference) ->
# flush -> R2 warm (scored vs e2g A2). Debug-agent env: add E2J_DECODE_DEBUG_AGENT=1 to the overrides.
# Chain: e2j -> mc4340 -> E2-target -> v0521-base.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=207495 YIHOU_ROUND=e2j
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.e2j.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=40 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000 E2J_DECODE_DEBUG_AGENT=0"
