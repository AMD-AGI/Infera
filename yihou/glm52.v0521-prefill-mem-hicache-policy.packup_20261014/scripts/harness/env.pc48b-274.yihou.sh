# Task v0521cap2 round pc48b (= pc48 + YIHOU-PCOMPLETE-ATTNDP validation fix mounted on both legs; pc48 died at startup: "prefill_complete does not yet support attention tensor parallelism greater than one", a v0.5.21 validation bug vs the --attn-dp-size rewrite) (user 2026-10-13 ~08:00): decode allocation policy prefill_complete (both legs; forces
# prefill HiCache write_back), prefill HiCache 370 GB (chunk 512), decode mem 0.96, decode max_running 96, prefill mem
# 0.88, CONC 48. vs q1 C48 A3 22,306 (HiCache 450 write_through, decode 0.95, early). Chain: chain_cap2 48 48.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-274 YIHOU_IP=10.245.157.34 YIHOU_JOB=211171 YIHOU_ROUND=pc48b
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.pcomplete.sh
export YIHOU_OVERRIDES="PCOMPLETE_BASE_CONFIG=$H/config.yihou.mc4340n2.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 PREFILL_HICACHE_SIZE=370 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.96 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000 PREFILL_MOUNTS_ADD=$H/mounts.yihou.pcomplete-attndp.tsv DECODE_MOUNTS_ADD=$H/mounts.yihou.pcomplete-attndp.tsv"
