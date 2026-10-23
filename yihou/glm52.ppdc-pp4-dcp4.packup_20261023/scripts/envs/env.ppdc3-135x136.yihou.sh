# env.ppdc3-135x136.yihou.sh (2026-10-23, user) = env.ppdc1 with prefill HiCache write_back -> write_through ONLY (ablation; C56).
# Task dcpx 2026-10-22: g3x1 with ONLY the decode leg replaced by the DCP4 decode (r5b). Differences vs
# env.g3x1-135x136.yihou.sh: CONFIG -> config.yihou.dcpx.sh (DCPX_BASE_CONFIG = xnode), round tag, decode
# mem 0.85 / max-running 64 / index_share false (r5b values; graph bs 64 is the harness default on both), image IDs per node.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-135 YIHOU_IP=10.245.148.209 YIHOU_JOB= YIHOU_ROUND=ppdc3
export XNODE_DECODE_NODE=crsuse2-m2m-136 XNODE_DECODE_IP=10.245.154.168
unset EXPECTED_IMAGE_ID
export CONFIG=$H/config.yihou.dcpx.sh
# Fill both IDs from `docker image inspect --format '{{.Id}}' infera-sglang:v0522-1020-yihou-dcp-mc4340` on 135 and 136.
_P_ID=sha256:1b78f7ccce06ef9fce39818a6d6536213db290f64fb5864eeb269956f89f27e6 _D_ID=sha256:1b78f7ccce06ef9fce39818a6d6536213db290f64fb5864eeb269956f89f27e6
export YIHOU_OVERRIDES="DCPX_BASE_CONFIG=$H/config.yihou.xnode.sh DCPX_PREFILL_IMAGE_ID=$_P_ID DCPX_DECODE_IMAGE_ID=$_D_ID DCPX_DCP=4 DCPX_SHARD=xstats DCPX_PMTP=1 DCPX_PACKNOWAIT=2 DCPX_PP=4 DCPX_SPF=1 DCPX_NOOFFLOAD=1 DCPX_DECODE_EC=1 PREFILL_HICACHE_WRITE_POLICY=write_through PREFILL_MAX_RUNNING=256 DCPX_IMAGE=infera-sglang:v0522-1020-yihou-dcp-mc4340-destpin DCPX_RAIL=destlocal DCPX_DECODE_RDMA=list DCPX_SPLIT=1 DCPX_SDMA0=0 XNODE_BASE_CONFIG=$H/config.yihou.idlefix.sh XNODE_DECODE_NODE=crsuse2-m2m-136 XNODE_DECODE_IP=10.245.154.168 NODE_RW_ROOT= IDLEFIX_BASE_CONFIG=$H/config.yihou.g4new.sh NEW_BASE_CONFIG=$H/config.yihou.xferfix2.sh PREFILL_HICACHE_SIZE=260 XFERFIX_BASE_CONFIG=$H/config.yihou.offload.sh YIHOU_XFER_LOG=1 YIHOU_XFER_SHARD=dest YIHOU_NONFINAL_EVENT=orig OFF_BASE_CONFIG=$H/config.yihou.tokw2.sh TOKW2_BASE_CONFIG=$H/config.yihou.mc4340n19.sh OFF_G=3 OFF_K=1 OFF_STEP=1 MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=56 DECODE_MAX_RUNNING=96 DECODE_GRAPH_MAX_BS=96 DCPX_GRAPH_BS=1,2,4,8,12,16,24,32,40,48,64,80,96 PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.88 DECODE_SIMULATE_ACC_LEN=3.61 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000 G4NEW_IDXSHARE=true DECODE_EXTRA_ENV_BASE=MC_LOG_LEVEL=INFO"
unset _P_ID _D_ID
# Run (ON 135, as g3x1): setsid nohup bash $H/chain_x.yihou.sh <this env> 56 56 > $R/dcpx7.chain.log 2>&1 < /dev/null &
