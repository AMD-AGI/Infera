#!/usr/bin/env bash
# One aggregated GLM-5.2 engine with the P4D4 decode config (DP4 TP1, fp8 KV, MTP3, block 64), on this node.
#   agg_up.sh NAME GPUS PORT RPC_PORT MODE [extra vllm serve args...]
#   MODE: features joined by "+":
#         stock   v0.34.0 as shipped
#         gluon   #47897 overlay v2 ($R/overlay: rocm_aiter_mla_sparse.py, rocm_gluon_paged_mqa_logits.py)
#                 bind-mounted, VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1, VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS=${LPT_ROWS:-4}
#         gluon3  #47897 overlay v3 ($R/overlay-v3: LPT group from the indexer decode metadata, no env for it)
#         comm    cuda_communicator.py with the AITER custom AG/RS gate accepting FULL graphs captured with
#                 batch_descriptor=None ($R/overlay-comm)
# Env: TRACE_DIR (host dir for torch traces; adds --profiler-config), ENV_EXTRA / MOUNT_EXTRA / IMAGE passed through
#      to vllm_up.sh.
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackA-20261118}
name=$1 gpus=$2 port=$3 rpc=$4 mode=$5; shift 5
P=/usr/local/lib/python3.12/dist-packages/vllm
V=$P/v1/attention/ops
a=(--data-parallel-size 4 --tensor-parallel-size 1 --kv-cache-dtype fp8_e4m3 --enable-prefix-caching
   --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --max-num-seqs 64 --max-num-batched-tokens 8192
   --block-size 64 --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
   --data-parallel-rpc-port "$rpc")
env_extra=${ENV_EXTRA:-}
mount_extra=${MOUNT_EXTRA:-}
for f in ${mode//+/ }; do
  case $f in
    stock) ;;
    gluon|gluon3)
      O=$R/overlay/vllm/v1/attention/ops
      if [[ $f == gluon3 ]]; then
        O=$R/overlay-v3/vllm/v1/attention/ops
        env_extra+=" VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1"
      else
        env_extra+=" VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1 VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS=${LPT_ROWS:-4}"
      fi
      mount_extra+=" $O/rocm_aiter_mla_sparse.py:$V/rocm_aiter_mla_sparse.py:ro"
      mount_extra+=" $O/rocm_gluon_paged_mqa_logits.py:$V/rocm_gluon_paged_mqa_logits.py:ro" ;;
    comm)
      C=$R/overlay-comm/vllm/distributed/device_communicators/cuda_communicator.py
      mount_extra+=" $C:$P/distributed/device_communicators/cuda_communicator.py:ro" ;;
    *) echo "unknown mode feature $f"; exit 1 ;;
  esac
done
if [[ -n "${TRACE_DIR:-}" ]]; then
  mkdir -p "$TRACE_DIR" && chmod 777 "$TRACE_DIR"
  mount_extra+=" $TRACE_DIR:/trace"
  a+=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/trace","torch_profiler_with_stack":false}')
fi
ENV_EXTRA="$env_extra" MOUNT_EXTRA="$mount_extra" exec bash "$R/scripts/vllm_up.sh" "$name" "$gpus" "$port" "${a[@]}" "$@"
