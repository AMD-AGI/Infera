# Task v0521cap2 round ofg2h21 on crsuse2-m2m-v3-006 (burst job 222453, amd-spur-v2; image 9d755dd6): OffloaderV2 on the
# prefill leg (G=2 K=1 step=1 cpu -> 1/2 of the MoE layers' experts in pinned host memory) + YIHOU-OFFLOAD-ATTRS patch, on the pf90
# config. PERF, sim 3.61, SHORT round C56 C56 (A1+A2; rerun of ofg2h5/17/18/19/20, lost or preempted); decode tok4/detok4 (rule 10-15). Driver chain_cap2 ON the node.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-v3-006 YIHOU_IP=10.245.144.8 YIHOU_JOB=222453 YIHOU_ROUND=ofg2h21
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.offload.sh
export YIHOU_OVERRIDES="OFF_BASE_CONFIG=$H/config.yihou.tokw2.sh TOKW2_BASE_CONFIG=$H/config.yihou.mc4340n7.sh OFF_G=2 OFF_K=1 OFF_STEP=1 MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=56 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.95 DECODE_SIMULATE_ACC_LEN=3.61 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
