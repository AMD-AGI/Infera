# Task v0521cap round k1 on 102 (user 2026-10-12 08:2x: "hi-cache 360, decode gmu 0.95"): e2g recipe (mc4340 ->
# E2-target -> v0521-base; decode mem 0.95 already) with prefill --hicache-size 360 (was 450) and CONC=44.
# Protocol: A1 fresh (pipeline) -> flush -> A2 warm (scored). Chain driver: chain_k1.yihou.sh.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=207495 YIHOU_ROUND=k1
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=44 PREFILL_HICACHE_SIZE=360 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
