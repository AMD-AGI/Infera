#!/usr/bin/env bash
# Capture immutable software/hardware provenance before any RCA workload starts.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/config/config.rca.p8d8.sh}"
source "$CONFIG"

OUT="${1:?usage: $0 OUT_DIR}"
mkdir -p "$OUT/local" "$OUT/$PREFILL_NODE" "$OUT/$DECODE_NODE"

git -C /home/liyingli/bench_agentx/baseline/Infera rev-parse HEAD \
    >"$OUT/local/git-head.txt"
git -C /home/liyingli/bench_agentx/baseline/Infera status --short --branch \
    >"$OUT/local/git-status.txt"
git -C /home/liyingli/bench_agentx/baseline/Infera log -1 --format=fuller \
    >"$OUT/local/git-log.txt"
sha256sum \
    "$BENCH_DIR/launch.sh" "$BENCH_DIR/engine.sh" \
    "$BENCH_DIR/agentx_bench.sh" "$BENCH_DIR/stop.sh" \
    "$BENCH_DIR/config.full.sh" "$CONFIG" \
    >"$OUT/local/harness-sha256.txt"

for node in "$PREFILL_NODE" "$DECODE_NODE"; do
    node_out="$OUT/$node"
    image_id="$(
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
            docker image inspect "$IMAGE" --format '{{.Id}}'
    )"
    if [[ "$image_id" != "$EXPECTED_IMAGE_ID" ]]; then
        echo "$node image mismatch: $image_id expected $EXPECTED_IMAGE_ID" >&2
        exit 1
    fi
    printf '%s\n' "$image_id" >"$node_out/image-id.txt"

    ssh -o BatchMode=yes "$node" bash -s >"$node_out/host.txt" 2>&1 <<'HOST'
set +e
echo "=== timestamp"; date -u --iso-8601=ns
echo "=== hostname"; hostname -f
echo "=== uname"; uname -a
echo "=== os-release"; cat /etc/os-release
echo "=== cmdline"; cat /proc/cmdline
echo "=== modules"; cat /proc/modules
echo "=== iommu-groups"; ls -l /sys/kernel/iommu_groups 2>&1
echo "=== numactl"; numactl --hardware 2>&1
echo "=== lspci-tree"; lspci -tv 2>&1
echo "=== lspci-gpu-rdma"; lspci -Dnnk 2>&1
echo "=== rocm-smi-version"; rocm-smi --showdriverversion --showproductname 2>&1
echo "=== rocm-smi-firmware"; rocm-smi --showfw 2>&1
echo "=== rocm-smi-topology"; rocm-smi --showtoponuma --showtopo 2>&1
echo "=== rocminfo"; rocminfo 2>&1
echo "=== ibv-devinfo"; ibv_devinfo -v 2>&1
echo "=== rdma-link"; rdma link show 2>&1
echo "=== ip-address"; ip -details address show 2>&1
echo "=== ip-route"; ip route show table all 2>&1
echo "=== libionic"
readlink -f /lib/x86_64-linux-gnu/libionic.so 2>&1
sha256sum /lib/x86_64-linux-gnu/libionic.so 2>&1
dpkg-query -W 'libionic*' 2>&1
echo "=== uverbs-abi"
for f in /sys/class/infiniband_verbs/uverbs*/abi_version; do
  printf '%s=' "$f"; cat "$f" 2>&1
done
echo "=== gpu-kfd-topology"
for d in /sys/class/kfd/kfd/topology/nodes/[0-9]*; do
  echo "--- $d"; cat "$d"/properties 2>&1
done
echo "=== ionic-topology"
for ib in /sys/class/infiniband/ionic_*; do
  [ -e "$ib" ] || continue
  name=${ib##*/}
  dev=$(readlink -f "$ib/device")
  echo "--- $name device=$dev"
  printf 'numa='; cat "$dev/numa_node" 2>&1
  printf 'state='; cat "$ib/ports/1/state" 2>&1
  printf 'active_mtu='; cat "$ib/ports/1/active_mtu" 2>&1
  printf 'gid1='; cat "$ib/ports/1/gids/1" 2>&1
  echo "netdevs=$(ls "$ib/device/net" 2>/dev/null | tr '\n' ' ')"
done
echo "=== docker-ps"; docker ps -a --no-trunc 2>&1
echo "=== rocm-processes"; rocm-smi --showmemuse --showpids 2>&1
echo "=== processes"; ps -eo user,pid,ppid,lstart,cmd --sort=pid 2>&1
HOST

    ssh -o BatchMode=yes "$node" \
        docker image inspect "$IMAGE" >"$node_out/image-inspect.json"
    ssh -o BatchMode=yes "$node" \
        docker history --no-trunc "$IMAGE" >"$node_out/image-history.txt"
    ssh -o BatchMode=yes "$node" \
        'ids=$(docker ps -aq); if [ -n "$ids" ]; then docker inspect $ids; else echo "[]"; fi' \
        >"$node_out/containers-before.json"
    ssh -o BatchMode=yes "$node" \
        'journalctl -k --no-pager -n 5000 2>&1 || dmesg 2>&1' \
        >"$node_out/kernel-before.log" || true

    ssh -o BatchMode=yes "$node" bash -s -- "$IMAGE" \
        >"$node_out/image-runtime.txt" 2>&1 <<'HOST_IMAGE'
set -euo pipefail
image="$1"
docker run --rm -i --entrypoint bash "$image" -s <<'CONTAINER'
set +e
echo "=== versions"
python3 -c "import sglang; print(\"sglang\", sglang.__version__)"
python3 -c "import torch; print(\"torch\", torch.__version__, \"hip\", torch.version.hip)"
python3 -c "import importlib.metadata as m, importlib.util as u; print(\"aiter\", u.find_spec(\"aiter\").origin); [print(\"aiter-dist\", d, m.version(d)) for d in (\"amd-aiter\", \"aiter\") if any(x.metadata[\"Name\"] == d for x in m.distributions())]"
echo "=== mooncake-package"
python3 - <<'PY'
import hashlib
import importlib.metadata
import pathlib
import mooncake
import mooncake.engine

so = pathlib.Path(mooncake.engine.__file__).resolve()
print("package", pathlib.Path(mooncake.__file__).resolve().parent)
print("engine", so)
print("sha256", hashlib.sha256(so.read_bytes()).hexdigest())
for dist in ("mooncake-transfer-engine", "mooncake"):
    try:
        print("distribution", dist, importlib.metadata.version(dist))
    except importlib.metadata.PackageNotFoundError:
        pass
PY
so=$(python3 -c "import mooncake.engine as e; print(e.__file__)")
echo "=== engine-build-id"; readelf -n "$so" 2>&1
echo "=== engine-ldd"; ldd "$so" 2>&1
echo "=== engine-capabilities"
nm -D "$so" 2>/dev/null | grep -E "ibv_reg_dmabuf_mr|hsa_amd_portable_export_dmabuf" || true
strings "$so" | grep -E "MC_(ENABLE|DISABLE|CUSTOM|NIC|GID)" | sort -u || true
echo "=== ionic-packages"; dpkg-query -W "libionic*" 2>&1
echo "=== build-env"; env | grep -E "^(AITER|TRITON|BUILD_MOONCAKE|ROCM)" | sort
true
CONTAINER
HOST_IMAGE

    ssh -o BatchMode=yes "$node" \
        /home/liyingli/bench_agentx/baseline/Infera/deploy/docker/scripts/verify_image_mooncake.sh \
        "$IMAGE" >"$node_out/verify-image-mooncake.txt" 2>&1 || {
            cat "$node_out/verify-image-mooncake.txt" >&2
            exit 1
        }
done

date -u --iso-8601=ns >"$OUT/captured-at.txt"
echo "provenance captured at $OUT"
