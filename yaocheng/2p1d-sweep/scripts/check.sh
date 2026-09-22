#!/usr/bin/env bash
# Node/path/RDMA checks plus CPU-only image inspection. No existing containers are stopped.
source "$(dirname "$0")/common.sh"
bash "$HARNESS_DIR/check_nodes.sh" "${nodes[@]}"
expected_image=""
expected_rails=""
while IFS=$'\t' read -r index instance role node ip; do
    say "checking $instance on $node"
    ssh_run "$node" test -r "$CONFIG"
    ssh_run "$node" test -r "$TOPOLOGY"
    ssh_run "$node" test -r "$REMOTE_BENCH_DIR/engine.sh"
    ssh_run "$node" test -r "$MODEL/tokenizer_config.json"
    ssh_run "$node" test -r "$HOST_RDMA_LIB"
    image_id="$(ssh_run "$node" docker image inspect --format '{{.Id}}' "$IMAGE")"
    say "$node image=$image_id"
    [[ -z "$EXPECTED_IMAGE_ID" || "$image_id" == "$EXPECTED_IMAGE_ID" ]] || {
        echo "$node: image differs from the pinned reference: $EXPECTED_IMAGE_ID" >&2; exit 1;
    }
    [[ -z "$expected_image" || "$image_id" == "$expected_image" ]] || {
        echo "image IDs differ across nodes" >&2; exit 1;
    }
    expected_image="$image_id"
    containers="$(ssh_run "$node" docker ps -a --format '{{.Names}}')"
    while read -r name; do
        [[ "$name" != "$CONTAINER_PREFIX-"* ]] || {
            echo "existing campaign container on $node: $name; stop the previous run first" >&2; exit 1;
        }
    done <<< "$containers"
    ports=("$((ENGINE_PORT_BASE + index))" "$((BOOTSTRAP_PORT_BASE + index))" \
        "$((KV_EVENT_PORT_BASE + index))" "$((SNAPSHOT_PORT_BASE + index))")
    [[ "$node" != "$CONTROL_NODE" ]] || ports+=("$ROUTER_PORT" "$ETCD_PORT" "$ETCD_PEER_PORT")
    rail_report="$(ssh_run "$node" python3 -c '
import pathlib, socket, sys
sockets=[]
rails=[]
for port in sys.argv[1:]:
    s=socket.socket(); s.bind(("0.0.0.0", int(port))); sockets.append(s)
for i in range(8):
    port=pathlib.Path(f"/sys/class/infiniband/ionic_{i}/ports/1")
    state=(port/"state").read_text().strip()
    gid=(port/"gids/1").read_text().strip()
    if "ACTIVE" not in state or not int(gid.replace(":", ""), 16):
        raise SystemExit(f"ionic_{i}: unusable rail: {state} {gid}")
    print(f"ionic_{i}: {state}, GID[1]={gid}")
    rails.append(gid.split(":")[1])
print("RAILS=" + ",".join(rails))
' "${ports[@]}")"
    printf '%s\n' "$rail_report"
    rails="${rail_report##*$'\n'}"
    [[ -z "$expected_rails" || "$rails" == "$expected_rails" ]] || {
        echo "$node: ionic_i rail IDs differ across nodes; adapt the GPU/NIC map before launching" >&2; exit 1;
    }
    expected_rails="$rails"
done <<< "$topology_rows"
bash "$SCRIPT_DIR/verify_pd_fixes.sh" "CONFIG=$CONFIG" "TOPOLOGY=$TOPOLOGY"
say "PASS: nodes, image patches, shared paths, ports and RDMA rails"
