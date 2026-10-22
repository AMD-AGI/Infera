# Task v0521cap2 round ofg4h3 on crsuse2-m2m-136 (plain ssh, image 9872bbda; offloader.py sha == image-9d755dd6 source): OffloaderV2 on the
# prefill leg (G=4 K=1 step=1 cpu -> 1/4 of the MoE layers' experts in pinned host memory) + YIHOU-OFFLOAD-ATTRS patch, on the pf90
# config (q1 + prefill mem 0.90). PERF pass, sim 3.61. HIGH CONC (user 10-15). Driver after_ofg4b136l2: chain_cap2l 56 56 64 after ofg4b. Decode tokenizer 4 + detokenizer 4 (user 10-15 rule for all new rounds). Relaunch #2 (ofg4h2 STOP at census: login-host sync_back hang); driver uses chain_cap2ls (streaming sync_back).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-136 YIHOU_IP=10.245.154.168 YIHOU_JOB= YIHOU_ROUND=ofg4h3
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.offload.sh
export YIHOU_OVERRIDES="OFF_BASE_CONFIG=$H/config.yihou.tokw2.sh TOKW2_BASE_CONFIG=$H/config.yihou.mc4340n.sh OFF_G=4 OFF_K=1 OFF_STEP=1 MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=56 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.95 DECODE_SIMULATE_ACC_LEN=3.61 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
