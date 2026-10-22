# Task v0521cap2 round ofg2c on crsuse2-m2m-288 (oai job 215369; image 9d755dd6): OffloaderV2 on the
# prefill leg (G=2 K=1 step=1 cpu -> 1/2 of the MoE layers' experts in pinned host memory) + YIHOU-OFFLOAD-ATTRS patch, on the pf90
# config (q1 + prefill mem 0.90). CORRECTNESS pass: simulated acceptance OFF; same tests as cx90/ofs0c. Driver ofs_seq (on node), then perf round ofg2.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-288 YIHOU_IP=10.245.152.60 YIHOU_JOB=215369 YIHOU_ROUND=ofg2c
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.offload.sh
export YIHOU_OVERRIDES="OFF_BASE_CONFIG=$H/config.yihou.mc4340n5.sh OFF_G=2 OFF_K=1 OFF_STEP=1 MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.95 DECODE_SIMULATE_ACC_LEN= ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
