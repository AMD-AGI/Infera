# Task v0521cap Track B round pc40 on oai node 102 (job 207495), after k1: e2g recipe (mc4340 image, HiCache 450, decode 0.95,
# max_running 64) + decode runshare patch YIHOU_DECODE_RUNNING_KV_SHARE=0.5. Chain: chain_pc40.yihou.sh (bring-up,
# long-prompt burst smoke, AgentX C40 A1 fresh -> flush -> A2 warm, scored vs e2g A2 22,114).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=207495 YIHOU_ROUND=pc40c
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.runshare.sh
export YIHOU_OVERRIDES="RUNSHARE_BASE_CONFIG=$H/config.yihou.mc4340n.sh MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh RUNNING_KV_SHARE=0.5 CONC=40 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
