#!/usr/bin/env bash
# From the login node: a two-node P8D8 deployment. etcd, the PP8 prefill and the router on the node of q.sh (job
# 294199), the DP8 decode on the node of q2.sh (hold_n2.sbatch). Waits until the router lists 1 prefill + 1 decode.
#   pd2_drive.sh RUN_ID   (role env vars are forwarded to both nodes: see pd_role_up.sh)
set -uo pipefail
R=/home/liyingli/bench_agentx/vllm-pd-20261117
run=${1:?run id}
mkdir -p "$R/runs/$run"
fwd=""
for v in IMAGE MODEL INFERA_SRC GEM P_ARGS D_ARGS P_ENV D_ENV P_KV_EXTRA D_PREFIX_CACHE D_KV_EVENTS D_MOUNT_EXTRA SPEC SIM_ACCEPT_LEN PP_LAYER_PARTITION; do
    [[ -n "${!v+x}" ]] && fwd+="$v=$(printf %q "${!v}") "
done
p_ip=$("$R/q.sh" -w ip 'ip -o -4 route get 8.8.8.8 | awk "{for (i = 1; i < NF; i++) if (\$i == \"src\") print \$(i + 1)}"' | grep -E '^[0-9.]+$' | head -1)
[[ -n "$p_ip" ]] || { echo "no prefill node IP"; exit 1; }
echo "prefill node IP $p_ip; forwarded: $fwd"
"$R/q.sh" -w etcd "docker rm -f vpd-etcd vpd-router vpd-prefill > /dev/null 2>&1; docker image inspect quay.io/coreos/etcd:v3.5.14 > /dev/null 2>&1 || docker pull -q quay.io/coreos/etcd:v3.5.14; docker run -d --name vpd-etcd --network host --entrypoint /usr/local/bin/etcd quay.io/coreos/etcd:v3.5.14 --advertise-client-urls http://$p_ip:24701 --listen-client-urls http://0.0.0.0:24701 --listen-peer-urls http://127.0.0.1:24702 --initial-advertise-peer-urls http://127.0.0.1:24702 --initial-cluster default=http://127.0.0.1:24702" || exit 1
"$R/q.sh" -w "p-$run" "$fwd bash scripts/pd_role_up.sh prefill $run $p_ip" > "$R/runs/$run/up-prefill.log" 2>&1 &
pp=$!
"$R/q2.sh" -w "d-$run" "$fwd bash scripts/pd_role_up.sh decode $run $p_ip" > "$R/runs/$run/up-decode.log" 2>&1 &
dp=$!
wait $pp; prc=$?; wait $dp; drc=$?
tail -2 "$R/runs/$run/up-prefill.log" "$R/runs/$run/up-decode.log"
(( prc == 0 && drc == 0 )) || { echo "engine start failed: prefill rc=$prc decode rc=$drc"; exit 1; }
src=""; [[ -n "${INFERA_SRC:-}" ]] && src="-v $INFERA_SRC:/opt/infera/infera:ro"
img=${IMAGE:-infera-vllm:v0340-0c5e5e89} model=${MODEL:-/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4}
"$R/q.sh" -w router "docker run -d --name vpd-router --network host --ulimit nofile=1048576:1048576 -v $model:$model:ro $src --entrypoint python3 $img -m infera.server --host 0.0.0.0 --port 28300 --router-backend rust --discovery-backend etcd --etcd-endpoint $p_ip:24701 --request-transport http --kv-event-transport zmq --router-tokenizer-path $model --router-policy kv-aware --kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on ${ROUTER_EXTRA:-} && (nohup docker logs -f vpd-router > $R/logs/engines/vpd-router.log 2>&1 &)" || exit 1
for i in $(seq 120); do
    modes=$(curl -sf -m 10 "http://$p_ip:28300/v1/workers" | python3 -c 'import json, sys; print(*sorted(w["disagg_mode"] for w in json.load(sys.stdin)["workers"]))' 2>/dev/null)
    [[ "$modes" == "decode prefill" ]] && break
    sleep 10
done
curl -sf -m 10 "http://$p_ip:28300/v1/workers" > "$R/runs/$run/workers.json"
echo "http://$p_ip:28300" > "$R/runs/$run/router_url"
echo "router workers: ${modes:-none} ($(date -u +%T)); ROUTER_URL=http://$p_ip:28300"
[[ "$modes" == "decode prefill" ]]
