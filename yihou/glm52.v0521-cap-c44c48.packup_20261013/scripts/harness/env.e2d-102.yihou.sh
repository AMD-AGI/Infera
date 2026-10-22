# Round e2d on 102 (R029): e2c target + index_share_for_mtp_iteration=TRUE, CONC 40, seed guard=count
# (decode), lean debug agent on decode only. One run answers: MAF yes/no, faulting kernel name,
# out-of-pool seed entries, eager-draft rate, and the index_share throughput at C40 (goal > 22k/chip).
# Chain: seedguard -> mc4340 -> E2-target -> v0521-base. Private PID ns + default IPC + DECODE_PID_OFFSET.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=202546 YIHOU_ROUND=e2d
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.seedguard.sh
export YIHOU_OVERRIDES="SEEDGUARD_BASE_CONFIG=$H/config.yihou.mc4340.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh INDEX_SHARE_FOR_MTP=true CONC=40 YIHOU_DSA_SEED_GUARD=count DECODE_DEBUG_AGENT=1 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
