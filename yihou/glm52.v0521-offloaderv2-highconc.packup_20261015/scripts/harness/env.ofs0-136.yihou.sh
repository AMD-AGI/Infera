# Task v0521cap2 round ofs0 on crsuse2-m2m-136 (plain ssh, image 9872bbda; offloader.py sha == image-9d755dd6 source): OffloaderV2 S0 on the
# prefill leg (G=8 K=1 step=1 cpu -> ~1/8 of the MoE layers' experts in pinned host memory) + YIHOU-OFFLOAD-ATTRS patch, on the pf90
# config (q1 + prefill mem 0.90). PERF pass: simulated acceptance 3.61 (as all perf rounds); compare with pf90 A2 23,127 (089) / A1 21,859. Driver chain_cap2l 48 48.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-136 YIHOU_IP=10.245.154.168 YIHOU_JOB= YIHOU_ROUND=ofs0
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.offload.sh
export YIHOU_OVERRIDES="OFF_BASE_CONFIG=$H/config.yihou.mc4340n.sh OFF_G=8 OFF_K=1 OFF_STEP=1 MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=48 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.95 DECODE_SIMULATE_ACC_LEN=3.61 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
