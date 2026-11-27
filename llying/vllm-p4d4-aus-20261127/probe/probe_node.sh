#!/usr/bin/env bash
# Node check for the P4D4 runs, run once per node by holder.sh before any item. Writes results/probe-<node>/:
#   host.txt     amdgpu / kernel / peer-memory modules / docker
#   nics.txt     RDMA devices, port state, GID 1, PCI path; gpus.txt GPU PCI paths; gpu-nic.txt nearest NIC per GPU
#   pin-*.log    kfd_pin_repro.py (../kfd619/bench) in the nightly base image, one GPU per case, 150 GiB each:
#                kfd+dmabuf, kfd+peermem, gem+dmabuf; with the image's own libionic and with the host's
#   gem          the decision: 0 = KV in plain hipMalloc registered through peer memory (MOONCAKE_DISABLE_HIP_DMABUF=1),
#                1 = KV in GEM BOs (dma-buf, the crusoe path)
#   libionic     0 = the image's libionic works, 1 = mount the host's
#   summary.txt  one line per check
set -uo pipefail
. /perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/site/env.sh
node=$(hostname -s)
O=$W/results/probe-$node
mkdir -p "$O"
K=/perf_apps/liyingli/bench_agentx/kfd619/bench
{
    echo "node $node $(ts)"
    echo "amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null) kernel $(uname -r)"
    echo "peer-memory modules: $(lsmod | awk '{print $1}' | grep -E 'peer' | paste -sd, -)"
    ls /sys/kernel/mm/memory_peers 2>/dev/null | sed 's/^/memory_peers: /'
    echo "docker $(docker version -f '{{.Server.Version}}' 2>&1)"
    grep -E '^(MemTotal|MemAvailable|Shmem|HugePages_Total):' /proc/meminfo
    echo "THP $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
    df -h / /var/lib/docker /tmp 2>/dev/null
} > "$O/host.txt" 2>&1
for d in /sys/class/infiniband/*; do
    n=$(basename "$d")
    echo "$n state=$(cat "$d/ports/1/state" 2>/dev/null) gid1=$(cat "$d/ports/1/gids/1" 2>/dev/null) pci=$(readlink -f "$d/device")"
done > "$O/nics.txt"
for d in /sys/bus/pci/devices/*; do
    [[ "$(cat "$d/vendor")" == 0x1002 && "$(cat "$d/class")" == 0x12* ]] && echo "$(basename "$d") $(readlink -f "$d")"
done > "$O/gpus.txt"
# Nearest NIC per GPU: the longest common PCI path prefix (same switch); GPU order = PCI bus order = HIP device order.
python3 - "$O/gpus.txt" "$O/nics.txt" > "$O/gpu-nic.txt" <<'EOF'
import os, sys
gpus = [l.split()[1] for l in open(sys.argv[1]) if l.strip()]
nics = [(l.split()[0], l.split("pci=")[1].strip()) for l in open(sys.argv[2]) if "pci=" in l]
for i, g in enumerate(gpus):
    best = max(nics, key=lambda n: len(os.path.commonpath([g, n[1]])))
    print(f"cuda:{i} {best[0]} common={os.path.commonpath([g, best[1]])}")
EOF
pull_image "$NIGHTLY_BASE" > /dev/null || echo "pull of $NIGHTLY_BASE FAILED" >> "$O/summary.txt"
run_case() {   # gpu alloc reg libionic(0/1)
    local g=$1 alloc=$2 reg=$3 hl=$4 tag="pin-g$1-$2-$3-hostlib$4"
    local lib=()
    # Host libionic: mounted at /host-libionic and copied over every libionic in the image (the sglang images' entrypoint
    # does the same on this cluster).
    local swap=""
    if [[ "$hl" == 1 ]]; then
        lib=(-v "$(readlink -f /lib/x86_64-linux-gnu/libionic.so):/host-libionic/libionic.so:ro")
        swap='for f in $(find / -xdev -name "libionic*.so*" -not -path "/host-libionic/*" 2>/dev/null); do cp -f /host-libionic/libionic.so "$f"; done; '
    fi
    timeout 1200 docker run --rm --name "vpd-probe-g$g" --network host --ipc host --device /dev/kfd --device /dev/dri \
        --device /dev/infiniband --group-add video --group-add render --cap-add IPC_LOCK --security-opt seccomp=unconfined \
        --ulimit memlock=-1:-1 -v "$K:$K:ro" "${lib[@]}" -e ROCR_VISIBLE_DEVICES="$g" -e RDMAV_FORK_SAFE=1 \
        --entrypoint bash "$NIGHTLY_BASE" -c "$swap ibv_devices 2>&1 | head -12; python3 $K/kfd_pin_repro.py --gpu 0 --alloc $alloc \
        --reg $reg --gib 150 --nics ionic_$g --tag $tag; echo \"exit \$?\"" > "$O/$tag.log" 2>&1
}
# Round 1: image libionic; round 2: host libionic. Three GPUs per round in parallel.
for hl in 0 1; do
    run_case 0 kfd dmabuf $hl & run_case 1 kfd peermem $hl & run_case 2 gem dmabuf $hl & wait
done
python3 - "$O" > "$O/pin-summary.txt" <<'EOF'
import glob, json, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "pin-*.log"))):
    st, tail = {}, ""
    for l in open(f, errors="replace"):
        l = l.strip()
        if l.startswith("{"):
            try: d = json.loads(l); st[d["step"]] = d
            except ValueError: pass
        elif l: tail = l
    r = st.get("registered", {})
    print(os.path.basename(f)[:-4], "before", st.get("before", {}).get("headroom_gib"), "after_reg", r.get("headroom_gib"),
          "real_free", r.get("real_free_gib"), "mrs", r.get("mrs"), "err", st.get("error", {}).get("what", ""), tail)
EOF
# Decision. GEM is not needed when hipMalloc + peer memory registers 150 GiB and leaves ~the real free VRAM allocatable
# (no double count); a driver that charges it twice (6.19) or a node without peer memory needs GEM.
python3 - "$O" <<'EOF'
import os, sys
O = sys.argv[1]
rows = {l.split()[0]: l.split() for l in open(os.path.join(O, "pin-summary.txt"))}
def ok(tag):
    r = rows.get(tag)
    if not r or r[-1] != "0" or r[4] in ("None", ""): return False
    return float(r[4]) > 100 and abs(float(r[4]) - float(r[6])) < 10   # headroom after reg ~ real free
for hl in ("0", "1"):
    if any(ok(f"pin-g{g}-{a}-{r}-hostlib{hl}") for g, a, r in (("1", "kfd", "peermem"), ("2", "gem", "dmabuf"), ("0", "kfd", "dmabuf"))):
        break
else:
    hl = "0"
gem = "0" if ok(f"pin-g1-kfd-peermem-hostlib{hl}") else "1"
open(os.path.join(O, "gem"), "w").write(gem + "\n")
open(os.path.join(O, "libionic"), "w").write(hl + "\n")
print(f"gem={gem} libionic_host={hl} peermem_ok={ok(f'pin-g1-kfd-peermem-hostlib{hl}')} gem_ok={ok(f'pin-g2-gem-dmabuf-hostlib{hl}')} dmabuf_ok={ok(f'pin-g0-kfd-dmabuf-hostlib{hl}')}")
EOF
{ head -3 "$O/host.txt"; cat "$O/gpu-nic.txt"; cat "$O/pin-summary.txt"; echo "decision gem=$(cat "$O/gem") libionic_host=$(cat "$O/libionic")"; } >> "$O/summary.txt"
touch "$O/done"
cat "$O/summary.txt"
