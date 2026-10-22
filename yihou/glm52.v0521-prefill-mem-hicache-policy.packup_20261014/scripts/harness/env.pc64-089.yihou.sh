# Task v0521cap2 round pc64 on crsuse2-m2m-089 (oai job 213048), user 2026-10-14 ~04:20 "在pc48b的基础上跑conc=64":
# pc48b config (decode allocation policy prefill_complete on both legs + YIHOU-PCOMPLETE-ATTNDP hook fix, prefill HiCache
# 370 GB write_back, decode mem 0.96, decode max_running 96, prefill mem 0.88) at CONC 64. Hook now via YIHOU_MOUNTS
# (config.yihou.pcomplete2.sh). Chain: 64 64 (A1 fresh C64 -> A2 scored C64), after pf90 A2 on this node.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-089 YIHOU_IP=10.245.145.201 YIHOU_JOB=213048 YIHOU_ROUND=pc64
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.pcomplete2.sh
export YIHOU_OVERRIDES="PCOMPLETE_BASE_CONFIG=$H/config.yihou.mc4340n4.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=64 PREFILL_HICACHE_SIZE=370 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.96 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
