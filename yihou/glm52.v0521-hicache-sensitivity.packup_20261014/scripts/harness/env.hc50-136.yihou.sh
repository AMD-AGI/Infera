# Task v0521cap2 round hc50 on crsuse2-m2m-136 (plain ssh; /shared_nfs ro there -> NODE_RW_ROOT /mnt/m2m_nobackup/yihou_v0521dbg),
# user 2026-10-13 ~16:00: e2g recipe (C40 best 22,114 on 102: HiCache 450, decode 0.95, max_running 64, prefill 0.85) with
# prefill HiCache 50 GB, C40, everything else unchanged. Image = 136's own mc4340 build (9872bbda, engine.so identical to 102's).
# Chain (login host): chain_cap2l 40 40.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-136 YIHOU_IP=10.245.154.168 YIHOU_JOB= YIHOU_ROUND=hc50
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=40 PREFILL_HICACHE_SIZE=50 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
