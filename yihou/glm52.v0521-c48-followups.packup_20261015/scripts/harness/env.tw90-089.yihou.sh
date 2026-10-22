# Task v0521cap2 round tw90 on crsuse2-m2m-089 (oai job 213048), user 2026-10-14 ~08:00 "decode加 --tokenizer-worker-num 4":
# pf90 config (q1 + prefill mem 0.90; A2 C48 23,127 on this same node) + decode --tokenizer-worker-num 4 (config.yihou.tokw.sh).
# Same node as pf90 -> same-node A/B. Chain: chain_cap2 48 48 (after cx90 on this node).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-089 YIHOU_IP=10.245.145.201 YIHOU_JOB=213048 YIHOU_ROUND=tw90
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.tokw.sh
export YIHOU_OVERRIDES="TOKW_BASE_CONFIG=$H/config.yihou.mc4340n4.sh DECODE_TOKENIZER_WORKERS=4 MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
