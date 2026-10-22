# Round e2f on 102 (R036): target config (E2-target, index_share FALSE) @ CONC 40 + KV-transfer HOL fix:
# YIHOU_NONFINAL_EVENT=launch (non-final wait_event covers only batch N, not N+1) + YIHOU_XFER_SHARD=room
# + YIHOU_XFER_LOG=1 (sync/idle stats). KV_CKSUM decided after the microbench (built-in checksum cost).
# Chunked prefill stays at the e2c value (32768) unless e2e shows 65536 helps.
# Chain: xferfix -> mc4340 -> E2-target -> v0521-base. Private PID ns + default IPC + DECODE_PID_OFFSET.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=202546 YIHOU_ROUND=e2f
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.xferfix.sh
export YIHOU_OVERRIDES="XFERFIX_BASE_CONFIG=$H/config.yihou.mc4340.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=40 YIHOU_NONFINAL_EVENT=launch YIHOU_XFER_SHARD=room YIHOU_XFER_LOG=1 KV_CKSUM=0 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
