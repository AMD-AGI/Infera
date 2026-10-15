# GLM-5.2-MXFP4 aggregated (no PD) AgentX run on one MI355X node: one TP4 + DP4-attention
# worker on GPUs 0-3 of crsuse2-m2m-138, after InferenceX's B300 Dynamo+SGLang aggregated
# recipe (../inferencex-glm52-b300-dynamo-agg-20261015/evidence/c48/config_c48.yaml).
# The image, model, AgentX client and harness are the 3P1D kit's; this file only replaces the
# topology, node, ports, run directory and the worker's settings.
# Required env: ROUND (new tag per bring-up). Usage: source config/config.agg.sh
AGG=/home/liyingli/bench_agentx/baseline/Infera/llying/agg-tp4dpa-138-v0521-20261015
source /home/liyingli/bench_agentx/baseline/Infera/llying/3p1d-tp4-tp8-v0521-20261012/config/config.3p1d.sh

TOPOLOGY="$AGG/config/topology.agg.tsv"
CONTROL_NODE=crsuse2-m2m-138          # etcd + router + AgentX client + the worker
BUILDER_NODE="$CONTROL_NODE"
YIHOU_NODE="$CONTROL_NODE"
CONTAINER_PREFIX="glm52-llying-agg-$ROUND"
RUNS="$AGG/runtime/runs/$ROUND"

ETCD_PORT=24601 ETCD_PEER_PORT=24602
ROUTER_PORT=28200
ENGINE_PORT_BASE=27101 ENGINE_PORT_STRIDE=256
BOOTSTRAP_PORT_BASE=28298             # unused (no PD); agentx_env.py requires every base
KV_EVENT_PORT_BASE=27657              # DP rank r publishes on base + r
SNAPSHOT_PORT_BASE=28951

# The worker (B300 recipe values unless noted).
GPUS_mixed_0=0,1,2,3
MIXED_TP=4 MIXED_EP=1 MIXED_DP=4 MIXED_DPA=1 MIXED_DP_LM_HEAD=1
MIXED_MTP=1 MIXED_SIMULATE_ACC_LEN=3.61
MIXED_MEM_FRACTION=0.85
MIXED_CHUNK_SIZE=32768 MIXED_MAX_PREFILL_TOKENS=8192
MIXED_MAX_RUNNING=128 MIXED_GRAPH_MAX_BS=64
MIXED_PREFILL_DELAYER_MAX_PASSES=60
# HiCache: 270 GB host pool per rank, write_back as on B300. B300 uses io-backend direct with
# page_first_direct; here kernel + page_first, the combination the 3P1D prefills run on ROCm.
MIXED_HICACHE=1 MIXED_HICACHE_SIZE=270 MIXED_HICACHE_RATIO=1.5
MIXED_HICACHE_WRITE_POLICY=write_back MIXED_HICACHE_IO_BACKEND=kernel MIXED_HICACHE_MEM_LAYOUT=page_first
# Host pools above 256 GiB need registration in chunks (3P1D prefill-0, 400 GB).
HICACHE_HOST_REGISTER_CHUNK_GB=512
MIXED_TOKENIZER_WORKERS=1             # B300 default; the 3P1D decode needed 4 at ~5k tok/s
MIXED_EXTRA_ARGS=""
MIXED_EXTRA_ENV=""
PID_OFFSET_mixed_0=30000
HSA_NO_SCRATCH_RECLAIM=1

# Router: kv-aware with session affinity on mixed workers (prefill-only affinity skips them).
# Mixed workers are scored with --kv-overlap-weight (the PD weights do not apply); 1.0 is the
# infera and Dynamo KV-router default, which the B300 run keeps. An explicit ROUTER_ENV wins.
ROUTER_ENV="${ROUTER_ENV-INFERA_SESSION_AFFINITY=both INFERA_SESSION_AFFINITY_TTL_SECS=3600}"
KV_OVERLAP_WEIGHT="${KV_OVERLAP_WEIGHT:-1.0}"
AITER_JIT_SEED="$NODE_RW_ROOT/cache/aiter/decode-0"
# Per-bring-up deltas (what differs from the above in that round).
if [[ -r "$AGG/config/round.$ROUND.sh" ]]; then source "$AGG/config/round.$ROUND.sh"; fi
