# Task idxshare FIX VALIDATION replicate idxf1c on crsuse2-m2m-054 (OAI 228279): g4n1 config (new rule, index_share TRUE, custom AR on, PR42553, HiCache 260) + ONLY eager_runner.yihou-idlefix.py. No guards/debug agent. Pass = full A1+A2 with no decode MAF. Driver chain_cap2 ON the node.
# prefill leg (G=4 K=1 step=1 cpu -> 1/4 of the MoE layers' experts in pinned host memory) + YIHOU-OFFLOAD-ATTRS patch, on the pf90
# config. PERF, sim 3.61, SHORT round C56 C56 (A1+A2); decode tok4/detok4 (rule 10-15). Driver chain_cap2 ON the node.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-054 YIHOU_IP=10.245.144.125 YIHOU_JOB=228279 YIHOU_ROUND=idxf1c
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.idlefix.sh
export YIHOU_OVERRIDES="IDLEFIX_BASE_CONFIG=$H/config.yihou.g4new.sh NEW_BASE_CONFIG=$H/config.yihou.xferfix2.sh PREFILL_HICACHE_SIZE=260 XFERFIX_BASE_CONFIG=$H/config.yihou.offload.sh YIHOU_XFER_LOG=1 YIHOU_XFER_SHARD=dest YIHOU_NONFINAL_EVENT=orig OFF_BASE_CONFIG=$H/config.yihou.tokw2.sh TOKW2_BASE_CONFIG=$H/config.yihou.mc4340n18.sh OFF_G=4 OFF_K=1 OFF_STEP=1 MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=56 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.95 DECODE_SIMULATE_ACC_LEN=3.61 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
