# Task v0521cap2 round x8p on crsuse2-m2m-274 (oai job 211171): q1 config (e2g + decode max_running 96 + prefill mem
# 0.88; HiCache 450, decode mem 0.95) with the synthetic 8-expert model on the PREFILL leg only (EP4, no shared-expert
# fusion, --max-total-tokens 3670016). Question: does more prefill KV raise the C48/C52 score? Chain: chain_cap2 48 48 52.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-274 YIHOU_IP=10.245.157.34 YIHOU_JOB=211171 YIHOU_ROUND=x8p
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.expert8leg.sh
export YIHOU_OVERRIDES="E8_LEG=prefill E8_BASE_CONFIG=$H/config.yihou.mc4340n2.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
