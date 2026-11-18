#!/usr/bin/env bash
# One-node P4D4 with Infera + vLLM: etcd, a TP1 x PP4 prefill on GPUs 0-3, a DP4 (TP1, MoE flattened to TP4) decode on
# GPUs 4-7, and the Infera router with the sglang recipe's router flags. KV over Mooncake RDMA loopback on the decode
# GPU's NIC (MC_DISABLE_HIP=1, MC_RDMA_WRITE_PREFER_DEST_NIC=1 + one preferred NIC per GPU); KV pools in GEM BOs
# when GEM=1 (crusoe: amdgpu 6.16 charges dma-buf-pinned VRAM twice).
#   pd1_up.sh RUN_ID
# Env: IMAGE, MODEL, INFERA_SRC, GEM (1; P_GEM / D_GEM per role), P_ARGS / D_ARGS (extra vllm args per role), P_ENV / D_ENV (extra K=V), P_KV_EXTRA
#      (JSON objects appended to the prefill MultiConnector list, e.g. an OffloadingConnector), D_PREFIX_CACHE (0),
#      D_KV_EVENTS (0; 1 = the decode publishes KV events, so the router can steer sessions to the rank that holds them),
#      P_MOUNT_EXTRA / D_MOUNT_EXTRA (prefill- / decode-only src:dst mounts), ROUTER_BACKEND (rust), ROUTER_ENV, ROUTER_EXTRA, SPEC (MTP config JSON; empty = no MTP), SIM_ACCEPT_LEN (empty = real acceptance).
# Writes $R/runs/RUN_ID/{engine-config.env,workers.json}. Exit 0 when the router lists 1 prefill + 1 decode.
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
run=${1:?run id}
RUN_DIR=$R/runs/$run
mkdir -p "$RUN_DIR"
export IMAGE=${IMAGE:-infera-vllm:v0340-0c5e5e89}
export MODEL=${MODEL:-/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4}
SERVED=glm5.2-mxfp4
GEM=${GEM:-1}
SPEC=${SPEC-'{"method":"mtp","num_speculative_tokens":3}'}
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
ETCD_PORT=24701 ROUTER_PORT=28300 P_PORT=27201 D_PORT=27301 BOOT_PORT=28398
NICS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
cfg=$RUN_DIR/config
# INFERA_SRC: an infera/ source tree mounted over the image's (python imports /opt/infera/infera first).
src_mnt=${INFERA_SRC:+$INFERA_SRC:/opt/infera/infera:ro}
mkdir -p "$cfg"
# One preferred NIC per visible GPU (logical index inside the container), no fallback NIC: on crusoe ionic_k only
# reaches ionic_k, so a fallback to another NIC can never connect.
topo() { python3 - "$@" <<'EOF'
import json, sys
out, first = sys.argv[1], int(sys.argv[2])
t = {f"cuda:{i}": [[f"ionic_{first + i}"], []] for i in range(4)}
t.update({"cpu:0": [[f"ionic_{k}" for k in range(4)], []], "cpu:1": [[f"ionic_{k}" for k in range(4, 8)], []]})
json.dump(t, open(out, "w"), indent=1)
EOF
}
topo "$cfg/topo-prefill.json" 0
topo "$cfg/topo-decode.json" 4
mc_env="MC_GID_INDEX=1 MC_DISABLE_HIP=1 MC_RDMA_WRITE_PREFER_DEST_NIC=1 MC_IB_TIMEOUT=18 MC_MAX_MR_SIZE=1073741824
MOONCAKE_DISABLE_HIP_DMABUF=0 MC_HIP_KFD_PIN_FIX=0 VLLM_HOST_IP=$IP VLLM_MOONCAKE_BOOTSTRAP_PORT=$BOOT_PORT
VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1800 INFERA_ENGINE_READY_TIMEOUT=3600"
# GEM per role (P_GEM / D_GEM, default GEM): the KV pools of that role in GEM BOs.
P_GEM=${P_GEM:-$GEM} D_GEM=${D_GEM:-$GEM}
gem_env="INFERA_AMD_GEM_ALLOCATOR=/opt/infera-gem/libhip_gem_alloc.so LD_PRELOAD=/opt/infera-gem/libhip_gem_alloc.so"
gem_mod=',"kv_connector_module_path":"infera.engine.vllm.gem_mooncake"'
p_mc_env=$mc_env d_mc_env=$mc_env p_mod="" d_mod=""
[[ "$P_GEM" == 1 ]] && p_mc_env+=" $gem_env" && p_mod=$gem_mod
[[ "$D_GEM" == 1 ]] && d_mc_env+=" $gem_env" && d_mod=$gem_mod
mc_extra="{\"num_workers\":10,\"device_name\":\"$NICS\"}"
p_kv="{\"kv_connector\":\"MultiConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"connectors\":[{\"kv_connector\":\"MooncakeConnector\"$p_mod,\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":$mc_extra}${P_KV_EXTRA:+,$P_KV_EXTRA}]}}"
d_kv="{\"kv_connector\":\"MooncakeConnector\"$d_mod,\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":$mc_extra}"
infera="python3 -m infera.engine.vllm --discovery-backend etcd --etcd-endpoint $IP:$ETCD_PORT --request-transport http --kv-event-transport zmq --advertise-host $IP"
common="--model $MODEL --served-model-name $SERVED --trust-remote-code --host 0.0.0.0 --kv-cache-dtype fp8_e4m3 --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice"
spec_args=""
if [[ -n "$SPEC" ]]; then
    s=$SPEC
    [[ -n "${SIM_ACCEPT_LEN:-}" ]] && s="${SPEC%\}},\"rejection_sample_method\":\"synthetic\",\"synthetic_acceptance_length\":$SIM_ACCEPT_LEN}"
    spec_args="--speculative-config '$s'"
    p_spec="--speculative-config '$SPEC'"
fi
d_pc="--no-enable-prefix-caching"
[[ "${D_PREFIX_CACHE:-0}" == 1 ]] && d_pc="--enable-prefix-caching"
p_cmd="$infera $common --port $P_PORT --tensor-parallel-size 1 --pipeline-parallel-size 4 --enable-prefix-caching --max-num-batched-tokens 8192 --max-num-seqs 64 ${p_spec:-} --kv-transfer-config '$p_kv' ${P_ARGS:-}"
d_ev="--no-enable-kv-events --kv-events off"
[[ "${D_KV_EVENTS:-0}" == 1 ]] && d_ev=""
d_cmd="$infera $d_ev $common --port $D_PORT --data-parallel-size 4 --tensor-parallel-size 1 --data-parallel-rpc-port 29620 $d_pc --max-num-seqs 64 --max-num-batched-tokens 8192 --stream-interval 30 $spec_args --kv-transfer-config '$d_kv' ${D_ARGS:-}"
{
    echo "RUN=$run IMAGE=$IMAGE INFERA_SRC=${INFERA_SRC:-} IP=$IP P_GEM=$P_GEM D_GEM=$D_GEM SIM_ACCEPT_LEN=${SIM_ACCEPT_LEN:-real}"
    echo "P_ENV=${P_ENV:-}"; echo "D_ENV=${D_ENV:-}"; echo "ROUTER=${ROUTER_BACKEND:-rust} ${ROUTER_ENV:-} ${ROUTER_EXTRA:-}"; echo "MC_ENV=$mc_env"
    echo "PREFILL: $p_cmd"; echo "DECODE: $d_cmd"
} > "$RUN_DIR/engine-config.env"
docker rm -f vpd-etcd vpd-router vpd-prefill vpd-decode > /dev/null 2>&1
docker image inspect quay.io/coreos/etcd:v3.5.14 > /dev/null 2>&1 || docker pull -q quay.io/coreos/etcd:v3.5.14
docker run -d --name vpd-etcd --network host --entrypoint /usr/local/bin/etcd quay.io/coreos/etcd:v3.5.14 \
    --advertise-client-urls "http://$IP:$ETCD_PORT" --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
    --listen-peer-urls http://127.0.0.1:24702 --initial-advertise-peer-urls http://127.0.0.1:24702 \
    --initial-cluster default=http://127.0.0.1:24702 > /dev/null || exit 1
MOUNT_EXTRA="$cfg:$cfg:ro $src_mnt ${P_MOUNT_EXTRA:-}" ENV_EXTRA="$(echo $p_mc_env) MC_CUSTOM_TOPO_JSON=$cfg/topo-prefill.json VLLM_PP_LAYER_PARTITION=20,20,20,18 ${P_ENV:-}" \
    ENTRY="exec $p_cmd" R=$R bash "$R/scripts/vllm_up.sh" vpd-prefill 0,1,2,3 $P_PORT &
p_pid=$!
sleep 30
MOUNT_EXTRA="$cfg:$cfg:ro $src_mnt ${D_MOUNT_EXTRA:-}" ENV_EXTRA="$(echo $d_mc_env) MC_CUSTOM_TOPO_JSON=$cfg/topo-decode.json ${D_ENV:-}" \
    ENTRY="exec $d_cmd" R=$R bash "$R/scripts/vllm_up.sh" vpd-decode 4,5,6,7 $D_PORT &
d_pid=$!
wait $p_pid; p_rc=$?
wait $d_pid; d_rc=$?
if (( p_rc != 0 || d_rc != 0 )); then echo "engine start failed: prefill rc=$p_rc decode rc=$d_rc"; exit 1; fi
# ROUTER_BACKEND=python (with ROUTER_ENV=INFERA_VLLM_MOONCAKE_SERIAL=1 for serial PD legs) drops the two rust-only flags.
ROUTER_BACKEND=${ROUTER_BACKEND:-rust}
# ROUTER_POLICY_ARGS: the sglang recipe's kv-aware policy by default. The Python router tokenizes every prompt for it in
# its event loop (87% of its CPU in P4D4 s8), so with ROUTER_BACKEND=python use "--router-policy round-robin".
# --router-tokenizer-path is a required router argument whatever the policy. ROUTER_MOUNT_EXTRA: extra src:dst mounts for
# the router (e.g. a rebuilt Rust router over /usr/local/bin/infera-router).
ROUTER_POLICY_ARGS=${ROUTER_POLICY_ARGS:---router-policy kv-aware --kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2}
rmounts=(); for mx in ${ROUTER_MOUNT_EXTRA:-}; do rmounts+=(-v "$mx"); done
rflags="--session-affinity prefill"
[[ "$ROUTER_BACKEND" == rust ]] && rflags+=" --pd-prefill-guard-release completion --decode-input-demand on"
renv=(); for kv in ${ROUTER_ENV:-}; do renv+=(-e "$kv"); done
docker run -d --name vpd-router --network host --ulimit nofile=1048576:1048576 --cap-add SYS_PTRACE -v "$MODEL:$MODEL:ro" ${src_mnt:+-v $src_mnt} "${rmounts[@]}" "${renv[@]}" --entrypoint python3 "$IMAGE" -m infera.server \
    --host 0.0.0.0 --port $ROUTER_PORT --router-backend "$ROUTER_BACKEND" --discovery-backend etcd --etcd-endpoint "$IP:$ETCD_PORT" \
    --request-transport http --kv-event-transport zmq --router-tokenizer-path "$MODEL" $ROUTER_POLICY_ARGS \
    $rflags ${ROUTER_EXTRA:-} > /dev/null || exit 1
nohup docker logs -f vpd-router > "$R/logs/engines/vpd-router.log" 2>&1 &
for i in $(seq 120); do
    if [[ "$(docker inspect -f '{{.State.Running}}' vpd-router 2>/dev/null)" != true ]]; then
        echo "vpd-router exited; last log lines:"; docker logs --tail 20 vpd-router 2>&1; exit 1
    fi
    modes=$(curl -sf -m 10 "http://$IP:$ROUTER_PORT/v1/workers" | python3 -c 'import json, sys; print(*sorted(w["disagg_mode"] for w in json.load(sys.stdin)["workers"]))' 2>/dev/null)
    [[ "$modes" == "decode prefill" ]] && break
    sleep 10
done
curl -sf -m 10 "http://$IP:$ROUTER_PORT/v1/workers" > "$RUN_DIR/workers.json"
echo "router workers: ${modes:-none} ($(date -u +%T)); ROUTER_URL=http://$IP:$ROUTER_PORT"
echo "http://$IP:$ROUTER_PORT" > "$RUN_DIR/router_url"
[[ "$modes" == "decode prefill" ]]
