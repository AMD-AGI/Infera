# Task v0521cap2 round cx90 on crsuse2-m2m-089 (oai job 213048), user 2026-10-14 ~07:15: correctness tests "在我们现在最高的吞吐的配置上"
# = pf90 config (q1 + prefill mem 0.90; A2 C48 23,127, best real-model score) with SIMULATED ACCEPTANCE OFF
# (DECODE_SIMULATE_ACC_LEN= empty -> real EAGLE acceptance), everything else identical. Tests: smoke (greedy + chat template with
# generation_config sampling T=1.0 top_p=0.95), needle-in-a-haystack 8k..256k at 3 depths, long-context multi-fact retrieval.
# Driver: chain_cx.yihou.sh (up -> census -> smoke -> audit -> correctness_tests.yihou.py -> stop).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-089 YIHOU_IP=10.245.145.201 YIHOU_JOB=213048 YIHOU_ROUND=cx90
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340n4.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.95 DECODE_SIMULATE_ACC_LEN= ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
