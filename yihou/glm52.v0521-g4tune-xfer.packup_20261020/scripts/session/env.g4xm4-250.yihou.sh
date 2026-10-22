# Task g4tune R2-full G4 C56 SHARD=mix (xferfix2: mix/orig/LOG=1; one variable vs R1 g4xb5, same node) on crsuse2-m2m-250 (OAI job 225298, machine-pool target t-1019210042-7f5d lent to g4tune, amd-spur amd-oai-qos; image 9d755dd6): OffloaderV2 on the
# prefill leg (G=4 K=1 step=1 cpu -> 1/4 of the MoE layers' experts in pinned host memory) + YIHOU-OFFLOAD-ATTRS patch, on the pf90
# config. PERF, sim 3.61, SHORT round C56 C56 (A1+A2); decode tok4/detok4 (rule 10-15). Driver chain_cap2 ON the node.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-250 YIHOU_IP=10.245.158.155 YIHOU_JOB=225298 YIHOU_ROUND=g4xm4
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.xferfix2.sh
export YIHOU_OVERRIDES="XFERFIX_BASE_CONFIG=$H/config.yihou.offload.sh YIHOU_XFER_LOG=1 YIHOU_XFER_SHARD=mix YIHOU_NONFINAL_EVENT=orig OFF_BASE_CONFIG=$H/config.yihou.tokw2.sh TOKW2_BASE_CONFIG=$H/config.yihou.mc4340n16.sh OFF_G=4 OFF_K=1 OFF_STEP=1 MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=56 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.95 DECODE_SIMULATE_ACC_LEN=3.61 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
