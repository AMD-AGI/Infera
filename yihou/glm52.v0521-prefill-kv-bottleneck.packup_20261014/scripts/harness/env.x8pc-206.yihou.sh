# Task v0521cap2 round x8pc on crsuse2-m2m-206 (oai job 211763), user 2026-10-13 ~15:00 "跑": split of x8p.
# Same as x8p (prefill-only synthetic 8 experts, EP4, no shared-expert fusion; q1 config) but the prefill KV pool is capped
# back to the 256-expert prefill size: --max-total-tokens 2433088 (x8p: 3670016). Isolates expert8 compute/weights from
# prefill KV capacity. Chain: 48 48.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-206 YIHOU_IP=10.245.148.83 YIHOU_JOB=211763 YIHOU_ROUND=x8pc
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.expert8leg.sh
export YIHOU_OVERRIDES="E8_LEG=prefill E8_MAX_TOTAL_TOKENS=2433088 E8_BASE_CONFIG=$H/config.yihou.mc4340n3.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
