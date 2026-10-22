# Round e2i on 102 (R049, PREPARED): e2g control config + decode-only lean ROCm debug agent env (the
# only decode difference e2d had besides seedguard/index_share). No seedguard mounts, index_share FALSE,
# no cpusets. R1 fresh/cold C40, then R2 on the same stack after flush (chain_e2i.yihou.sh).
# Chain: dbgagent -> mc4340 -> E2-target -> v0521-base.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=207495 YIHOU_ROUND=e2i
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.dbgagent.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=40 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
