# Task v0521cap2 round ofg3b on crsuse2-m2m-v3-020 (burst job 217125, partition amd-spur-v2; image 9d755dd6): OffloaderV2 on the
# prefill leg (G=3 K=1 step=1 cpu -> 1/3 of the MoE layers' experts in pinned host memory) + YIHOU-OFFLOAD-ATTRS patch, on the pf90
# config (q1 + prefill mem 0.90). PERF pass, sim 3.61. Rerun of ofg3 (266 released 10-15 02:50); ofg3c 23/23 on 266. Driver chain_cap2 48 48 56.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-v3-020 YIHOU_IP=10.245.150.105 YIHOU_JOB=217125 YIHOU_ROUND=ofg3b
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.offload.sh
export YIHOU_OVERRIDES="OFF_BASE_CONFIG=$H/config.yihou.mc4340n6.sh OFF_G=3 OFF_K=1 OFF_STEP=1 MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.95 DECODE_SIMULATE_ACC_LEN=3.61 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
