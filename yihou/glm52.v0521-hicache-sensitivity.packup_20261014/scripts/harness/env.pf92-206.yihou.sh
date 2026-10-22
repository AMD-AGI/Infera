# Task v0521cap2 round pf92 on crsuse2-m2m-206 (oai job 211763), user 2026-10-14 "测0.90和0.92 prefill kv": q1 config (e2g + decode
# max_running 96, decode mem 0.95, HiCache 450) with PREFILL_MEM_FRACTION=0.92 (q1: 0.88). Real 256-expert model.
# Question: does a bigger prefill device KV pool raise the C48 score on the real model (synthetic runs: prefill KV binds)?
# Chain: chain_cap2 48 48 52.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-206 YIHOU_IP=10.245.148.83 YIHOU_JOB=211763 YIHOU_ROUND=pf92
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n4.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.92 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
