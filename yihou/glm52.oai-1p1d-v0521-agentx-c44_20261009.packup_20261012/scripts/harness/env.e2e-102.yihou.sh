# Round e2e on 102 (R033/R034): target config (E2-target, index_share FALSE) @ CONC 40 + KV-transfer log
# (YIHOU_XFER_LOG=1, PAR=1: te_conc showed k-way submission gives no gain) + prefill --chunked-prefill-size 65536
# (16384/rank, lever L2). Chain: xferpar -> mc4340 -> E2-target -> v0521-base. PID offset 30000.
# Gate: repro's concurrent-caller te_pair test byte-exact (diag/run_te_conc.yihou.sh).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=202546 YIHOU_ROUND=e2e
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.xferpar.sh
export YIHOU_OVERRIDES="XFERPAR_BASE_CONFIG=$H/config.yihou.mc4340.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=40 YIHOU_XFER_LOG=1 YIHOU_XFER_PAR=1 XFER_THREAD_POOL=16 PREFILL_CHUNK_SIZE=65536 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
