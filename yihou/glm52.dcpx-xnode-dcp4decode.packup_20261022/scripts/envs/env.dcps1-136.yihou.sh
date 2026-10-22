# env.dcps1-136.yihou.sh (task dcpx 2026-10-22, user 07:4xZ: free 135, continue SAME-NODE on 136) = dcpx5 placed on one node:
# prefill GPU0-3 + DCP4 decode GPU4-7 on crsuse2-m2m-136, KV over Mooncake HIP IPC (DCPX_RAIL=hip, base per-GPU map + no-such-device
# filter, as the DCP kit r5b). Everything else as dcpx5: image -destpin, overlay 93034eba91, g3x1 prefill (G3, GTT 260, mem 0.90),
# decode DCP4 + index_share true + max_running 128 + graph bs 128 + mem 0.85. Driver chain_cap2 ON 136.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-136 YIHOU_IP=10.245.154.168 YIHOU_JOB= YIHOU_ROUND=dcps1
unset EXPECTED_IMAGE_ID
_P_ID=sha256:1b78f7ccce06ef9fce39818a6d6536213db290f64fb5864eeb269956f89f27e6 _D_ID=sha256:1b78f7ccce06ef9fce39818a6d6536213db290f64fb5864eeb269956f89f27e6
export CONFIG=$H/config.yihou.dcpx.sh
export YIHOU_OVERRIDES="DCPX_BASE_CONFIG=$H/config.yihou.idlefix.sh DCPX_PREFILL_IMAGE_ID=$_P_ID DCPX_DECODE_IMAGE_ID=$_D_ID DCPX_DCP=4 DCPX_IMAGE=infera-sglang:v0522-1020-yihou-dcp-mc4340-destpin DCPX_RAIL=hip DCPX_DECODE_RDMA=list DCPX_SPLIT=1 DCPX_SDMA0=0 NODE_RW_ROOT= IDLEFIX_BASE_CONFIG=$H/config.yihou.g4new.sh NEW_BASE_CONFIG=$H/config.yihou.xferfix2.sh PREFILL_HICACHE_SIZE=260 XFERFIX_BASE_CONFIG=$H/config.yihou.offload.sh YIHOU_XFER_LOG=1 YIHOU_XFER_SHARD=dest YIHOU_NONFINAL_EVENT=orig OFF_BASE_CONFIG=$H/config.yihou.tokw2.sh TOKW2_BASE_CONFIG=$H/config.yihou.mc4340n19.sh OFF_G=3 OFF_K=1 OFF_STEP=1 MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=56 DECODE_MAX_RUNNING=128 DECODE_GRAPH_MAX_BS=128 PREFILL_MEM_FRACTION=0.90 DECODE_MEM_FRACTION=0.85 DECODE_SIMULATE_ACC_LEN=3.61 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000 G4NEW_IDXSHARE=true DECODE_EXTRA_ENV_BASE=MC_LOG_LEVEL=INFO"
unset _P_ID _D_ID
