# Task v0521cap2 round ofg2h3 on crsuse2-m2m-v3-026 (burst job 217126, partition amd-spur-v2; image 9d755dd6): OffloaderV2 on the
# prefill leg (G=2 K=1 step=1 cpu -> 1/2 of the MoE layers' experts in pinned host memory) + YIHOU-OFFLOAD-ATTRS patch, on the pf90
# config (q1 + prefill mem 0.90). PERF pass, sim 3.61. HIGH CONC (user 10-15: offload at high conc; no new baseline). Driver chain_cap2 56 56 64. Relaunch #2 (ofg2h2 failed: cache 026 lacked 12 aiter .so + stale lock; cache re-seeded 09:2x). Decode tokenizer 4 + detokenizer 4 (rule 10-15).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-v3-026 YIHOU_IP=10.245.152.176 YIHOU_JOB=217126 YIHOU_ROUND=ofg2h3
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.offload.sh
export YIHOU_OVERRIDES="OFF_BASE_CONFIG=$H/config.yihou.tokw2.sh TOKW2_BASE_CONFIG=$H/config.yihou.mc4340n6.sh OFF_G=2 OFF_K=1 OFF_STEP=1 MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=56 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.95 DECODE_SIMULATE_ACC_LEN=3.61 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
