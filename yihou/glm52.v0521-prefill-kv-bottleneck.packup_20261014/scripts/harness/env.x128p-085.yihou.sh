# Task v0521cap2 round x128p on crsuse2-m2m-085 (oai job 211169), user 2026-10-13 ~16:00: prefill leg with HALF the routed
# experts (256 -> 128, truncated checkpoint, own top-8 routing, EP1, no shared-expert fusion); decode unchanged.
# q1 config (decode max_running 96, prefill mem 0.88, decode mem 0.95, HiCache 450). Compare x8p (8 experts) 25,905 and q1 22,306.
# Chain: 48 48 (after tb56n on the same node).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-085 YIHOU_IP=10.245.153.145 YIHOU_JOB=211169 YIHOU_ROUND=x128p
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.expertnleg.sh
export YIHOU_OVERRIDES="EN_LEG=prefill EN_BASE_CONFIG=$H/config.yihou.mc4340n2.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
