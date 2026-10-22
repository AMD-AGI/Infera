# Task v0521cap2 round pf93 on crsuse2-m2m-136 (plain ssh; image 9872bbda; NODE_RW_ROOT /mnt/m2m_nobackup/yihou_v0521dbg),
# user 2026-10-14 ~07:15: probe how far prefill mem-fraction can go above 0.92 before OOM (q1 config otherwise). 0.93 -> expected
# prefill KV ~2.758 M tok and ~16.7 GiB post-pool headroom; measured runtime peak at 0.88 was ~22-23 GiB over the pool (089) ->
# may OOM under load. If it OOMs, step down to 0.93. Chain (login host): chain_cap2l 48 48.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-136 YIHOU_IP=10.245.154.168 YIHOU_JOB= YIHOU_ROUND=pf93
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.93 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
