# Task v0521cap2 Track B round tb56 on crsuse2-m2m-085 (oai job 211169), user 2026-10-13: decode in-flight KV cap
# (runshare v2, share 0.5, skip + 5 s anti-starvation) at high load with the ORIGINAL C40 HiCache capacity
# (e1c: --hicache-ratio 1.5, no --hicache-size = 154 GB/DP, same as the t2f sweep). Rest = q1 config (decode
# max_running 96 so the 16/DP running cap does not bind at C56/C64; prefill mem 0.88; decode mem 0.95).
# Compare with t2f (C40 19,798 -> C56 14,675, -25.9 %). Chain: chain_cap2 56 56 64.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-085 YIHOU_IP=10.245.153.145 YIHOU_JOB=211169 YIHOU_ROUND=tb56
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.runshare2.sh
export YIHOU_OVERRIDES="RUNSHARE_BASE_CONFIG=$H/config.yihou.mc4340n2.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh RUNNING_KV_SHARE=0.5 CONC=56 PREFILL_HICACHE_SIZE=0 PREFILL_HICACHE_RATIO=1.5 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
