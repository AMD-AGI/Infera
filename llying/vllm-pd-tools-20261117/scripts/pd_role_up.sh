#!/usr/bin/env bash
# Start one PD role on THIS node for a two-node deployment (P8D8): an 8-GPU TP1 x PP8 prefill or an 8-GPU DP8 decode,
# registering with the etcd on ETCD_HOST. Same engine flags and Mooncake settings as pd1_up.sh; the Mooncake topology
# maps each visible GPU k to ionic_k (same-named NICs are on the same rail across nodes, and ionic_k only reaches
# ionic_k). The router and etcd run on the prefill node (pd2_drive.sh).
#   pd_role_up.sh prefill|decode RUN_ID ETCD_HOST
# Env as pd1_up.sh: IMAGE, MODEL, INFERA_SRC, GEM, P_ARGS / D_ARGS, P_ENV / D_ENV, P_KV_EXTRA, D_PREFIX_CACHE,
# D_KV_EVENTS, D_MOUNT_EXTRA, SPEC, SIM_ACCEPT_LEN, PP_LAYER_PARTITION (10,10,10,10,10,10,10,8).
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
role=$1 run=$2 etcd_host=$3
RUN_DIR=$R/runs/$run
cfg=$RUN_DIR/config-$role
mkdir -p "$cfg"
export IMAGE=${IMAGE:-infera-vllm:v0340-0c5e5e89}
export MODEL=${MODEL:-/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4}
SERVED=glm5.2-mxfp4
GEM=${GEM:-1}
SPEC=${SPEC-'{"method":"mtp","num_speculative_tokens":3}'}
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
ETCD_PORT=24701 P_PORT=27201 D_PORT=27301 BOOT_PORT=28398
NICS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
src_mnt=${INFERA_SRC:+$INFERA_SRC:/opt/infera/infera:ro}
python3 - "$cfg/topo.json" <<'EOF'
import json, sys
t = {f"cuda:{i}": [[f"ionic_{i}"], []] for i in range(8)}
t.update({"cpu:0": [[f"ionic_{k}" for k in range(4)], []], "cpu:1": [[f"ionic_{k}" for k in range(4, 8)], []]})
json.dump(t, open(sys.argv[1], "w"), indent=1)
EOF
mc_env="MC_GID_INDEX=1 MC_DISABLE_HIP=1 MC_RDMA_WRITE_PREFER_DEST_NIC=1 MC_IB_TIMEOUT=18 MC_MAX_MR_SIZE=1073741824
MOONCAKE_DISABLE_HIP_DMABUF=0 MC_HIP_KFD_PIN_FIX=0 VLLM_HOST_IP=$IP VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOT_PORT
VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1800 INFERA_ENGINE_READY_TIMEOUT=3600 MC_CUSTOM_TOPO_JSON=$cfg/topo.json"
[[ "$GEM" == 1 ]] && mc_env+=" INFERA_AMD_GEM_ALLOCATOR=/opt/infera-gem/libhip_gem_alloc.so LD_PRELOAD=/opt/infera-gem/libhip_gem_alloc.so"
mc_mod=""
[[ "$GEM" == 1 ]] && mc_mod=',"kv_connector_module_path":"infera.engine.vllm.gem_mooncake"'
mc_extra="{\"num_workers\":10,\"device_name\":\"$NICS\"}"
infera="python3 -m infera.engine.vllm --discovery-backend etcd --etcd-endpoint $etcd_host:$ETCD_PORT --request-transport http --kv-event-transport zmq --advertise-host $IP"
common="--model $MODEL --served-model-name $SERVED --trust-remote-code --host 0.0.0.0 --kv-cache-dtype fp8_e4m3 --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice"
if [[ "$role" == prefill ]]; then
    p_kv="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"MooncakeConnector\"$mc_mod,\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":$mc_extra}${P_KV_EXTRA:+,$P_KV_EXTRA}]}}"
    spec=""; [[ -n "$SPEC" ]] && spec="--speculative-config '$SPEC'"
    cmd="$infera $common --port $P_PORT --tensor-parallel-size 1 --pipeline-parallel-size 8 --enable-prefix-caching --max-num-batched-tokens 8192 --max-num-seqs 128 $spec --kv-transfer-config '$p_kv' ${P_ARGS:-}"
    envx="VLLM_PP_LAYER_PARTITION=${PP_LAYER_PARTITION:-10,10,10,10,10,10,10,8} ${P_ENV:-}"
    name=vpd-prefill port=$P_PORT mnt="$cfg:$cfg:ro $src_mnt"
else
    d_kv="{\"kv_connector\":\"MooncakeConnector\"$mc_mod,\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":$mc_extra}"
    spec=""
    if [[ -n "$SPEC" ]]; then
        s=$SPEC
        [[ -n "${SIM_ACCEPT_LEN:-}" ]] && s="${SPEC%\}},\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$SIM_ACCEPT_LEN}"
        spec="--speculative-config '$s'"
    fi
    d_pc="--no-enable-prefix-caching"; [[ "${D_PREFIX_CACHE:-0}" == 1 ]] && d_pc="--enable-prefix-caching"
    d_ev="--no-enable-kv-events --kv-events off"; [[ "${D_KV_EVENTS:-0}" == 1 ]] && d_ev=""
    cmd="$infera $d_ev $common --port $D_PORT --data-parallel-size 8 --tensor-parallel-size 1 --data-parallel-rpc-port 29620 $d_pc --max-num-seqs 64 --max-num-batched-tokens 8192 --stream-interval 30 $spec --kv-transfer-config '$d_kv' ${D_ARGS:-}"
    envx="${D_ENV:-}"
    name=vpd-decode port=$D_PORT mnt="$cfg:$cfg:ro $src_mnt ${D_MOUNT_EXTRA:-}"
fi
echo "$role on $(hostname -s) $IP: $cmd" > "$RUN_DIR/engine-$role.cmd"
docker rm -f "$name" > /dev/null 2>&1
MOUNT_EXTRA="$mnt" ENV_EXTRA="$(echo $mc_env) $envx" ENTRY="exec $cmd" R=$R bash "$R/scripts/vllm_up.sh" "$name" 0,1,2,3,4,5,6,7 "$port"
