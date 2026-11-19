# Mooncake P/D settings shared by the prefill and decode legs (after Infera dev/pd_opt/vllm_glm52_agentx
# scripts/pd1_up.sh, GLM-5.2 P4D4 on vLLM 0.34). Source after agg_tp4.sh with KV_ROLE (kv_producer | kv_consumer)
# and BOOT_PORT set. MR_MODE (peermem | gem, from lib.sh rdma_mr_mode) and GEM_SO come from the item.
#   MC_DISABLE_HIP=1         same-host transfers go over RDMA loopback, like the cross-node deployment
#   MC_MAX_MR_SIZE=1 GiB     ionic memory-region size limit; Mooncake registers large pools in 1 GiB pieces
#   MC_RDMA_NOTIFY_ENABLED=0 the vLLM connector signals completion over its own ZMQ channel; Mooncake's notify QPs
#                            failed to initialise (-302) at P8D8 scale (U27)
#   VLLM_HOST_IP             the advertised node address, so the bootstrap server and the router agree
#   VLLM_MOONCAKE_BOOTSTRAP_PORT one per worker on a shared node
MR_MODE=${MR_MODE:-peermem}
WORKER_ENV+=(
    MC_GID_INDEX=1
    MC_DISABLE_HIP=1
    MC_MAX_MR_SIZE=1073741824
    MC_RDMA_NOTIFY_ENABLED=0
    VLLM_HOST_IP="$(node_ip)"
    VLLM_MOONCAKE_BOOTSTRAP_PORT="$BOOT_PORT"
    VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1800
)
mc_mod=""
if [ "$MR_MODE" = gem ]; then
    # amdgpu 6.19 (or no peer-memory module): KV pools in amdgpu GEM BOs, registered by dma-buf and charged once by
    # KFD. GEM BOs must never go through peer memory (kfd_peerdirect dereferences the missing KFD BO).
    WORKER_ENV+=(INFERA_AMD_GEM_ALLOCATOR="${GEM_SO:?GEM_SO}" LD_PRELOAD="$GEM_SO" MOONCAKE_DISABLE_HIP_DMABUF=0 MC_HIP_KFD_PIN_FIX=0)
    mc_mod=',"kv_connector_module_path":"infera.engine.vllm.gem_mooncake"'
else
    # amdgpu 6.16: peer memory (ibv_reg_mr) is charged once, dma-buf twice.
    WORKER_ENV+=(MOONCAKE_DISABLE_HIP_DMABUF=1)
fi
# MC_TOPO=pergpu: GPU k uses only ionic_k (its PCIe neighbour; one rail and one /24 subnet per NIC on AUS). Unset,
# infera 0.3.3 pins every GPU to ionic_0 because the NICs span several subnets, so all KV crosses one NIC.
if [ "${MC_TOPO:-}" = pergpu ]; then
    topo=/tmp/mc-topo-$PORT.json
    python3 - "$GPUS" "$topo" <<'PY'
import json, sys
gpus = [int(g) for g in sys.argv[1].split(",")]
t = {f"cuda:{i}": [[f"ionic_{g}"], []] for i, g in enumerate(gpus)}
t.update({"cpu:0": [[f"ionic_{k}" for k in range(4)], []], "cpu:1": [[f"ionic_{k}" for k in range(4, 8)], []]})
json.dump(t, open(sys.argv[2], "w"))
PY
    WORKER_ENV+=(MC_CUSTOM_TOPO_JSON="$topo")
fi
VLLM_ARGS+=(--kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\"$mc_mod,\"kv_role\":\"$KV_ROLE\",\"kv_connector_extra_config\":{\"num_workers\":10,\"device_name\":\"ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7\"}}")
