# Round e2h on 102 (R045, PREPARED, NOT launched): e2g control config + NUMA-local CPU placement.
# Prefill (GPU0-3, NUMA node0) --cpuset-cpus=0-117, decode (GPU4-7, NUMA node1) --cpuset-cpus=118-235,
# SGLANG_SET_CPU_AFFINITY=0 on both legs (image default 1 pins ranks by LOCAL gpu_id -> decode DPs
# landed on node0 halves, identical to prefill DPk). --cpuset-mems NOT set: prefill HiCache host pool is
# 4 x 450 GB = 1.8 TB > one NUMA node (1.38 TB) -> PREFILL_CPUSET_MEMS=0 cannot fit (see R046).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=207495 YIHOU_ROUND=e2h
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.mc4340.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=40 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000 PREFILL_CPUSET_CPUS=0-117 DECODE_CPUSET_CPUS=118-235 ENGINE_SET_CPU_AFFINITY=0"
