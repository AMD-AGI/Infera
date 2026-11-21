#!/usr/bin/env bash
# E0 preflight: collect node facts (read-only) -> raw text files + facts.json + facts.md on NFS. Run ON the GPU node.
# Usage: preflight.sh [--out DIR] [--skip-docker] [--skip-gpu] [--in-image IMAGE ...|--in-image-all]
#   default out: $DSV41_NFS_ROOT/preflight/<hostname>-<UTC timestamp>/
#   --in-image IMAGE   run the in-image check list in IMAGE (docker run --rm, no GPU devices, network none)
#   --in-image-all     same for every image in configs/*.env
# Never stops/changes anything: foreign containers on /dev/kfd are REPORTED only.
set -uo pipefail
KIT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WS=$(cd "$KIT_DIR/../.." && pwd)
NFS_ROOT=${DSV41_NFS_ROOT:-/perf_apps/yihou/dsv41_vllm_pd_20261119}
LOCAL_ROOT=${DSV41_LOCAL_ROOT:-/data/yihou/dsv41_vllm_pd_20261119}
HOST=$(hostname -s)
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=""; SKIP_DOCKER=0; SKIP_GPU=0; IMGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=${2:?}; shift ;;
    --skip-docker) SKIP_DOCKER=1 ;;
    --skip-gpu) SKIP_GPU=1 ;;
    --in-image) IMGS+=("${2:?}"); shift ;;
    --in-image-all)
      for f in "$WS"/configs/*.env; do
        i=$(bash -c 'source "$1"; echo "$IMAGE"' _ "$f")
        [[ " ${IMGS[*]:-} " == *" $i "* ]] || IMGS+=("$i")
      done ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
  shift
done
OUT=${OUT:-$NFS_ROOT/preflight/$HOST-$TS}
case "$OUT" in */yihou/*) ;; *) echo "out dir must contain /yihou/" >&2; exit 2;; esac
mkdir -p "$OUT/raw"
R=$OUT/raw
have() { command -v "$1" >/dev/null 2>&1; }
cap() {  # cap <name> <cmd...> : run, save stdout+stderr, never fail
  local n=$1; shift
  { echo "\$ $*"; timeout 120 "$@" 2>&1; echo "[rc=$?]"; } > "$R/$n.txt"
}
note() { echo "$*" >> "$R/notes.txt"; }
gpu_containers() {  # every running container with /dev/kfd or --privileged; OURS = name contains yihou (report only)
  local id name priv devs img started who
  for id in $(docker ps -q); do
    IFS='|' read -r name priv devs img started < <(docker inspect -f '{{.Name}}|{{.HostConfig.Privileged}}|{{range .HostConfig.Devices}}{{.PathOnHost}},{{end}}|{{.Config.Image}}|{{.State.StartedAt}}' "$id")
    name=${name#/}
    if [ "$priv" = true ] || [[ "$devs" == */dev/kfd* ]]; then
      who=FOREIGN; [[ "$name" == *yihou* ]] && who=OURS
      echo "$who $name privileged=$priv devices=$devs image=$img started=$started"
    fi
  done
}

# ---- host --------------------------------------------------------------------------------------------------------
cap hostname bash -c 'hostname -f; hostname -I; uname -a; date -u'
cap os cat /etc/os-release
cap cpu bash -c 'lscpu; echo; nproc'
cap meminfo cat /proc/meminfo
cap free free -b
cap shm df -B1 --output=source,fstype,size,used,avail,target /dev/shm
cap disk_data df -B1 --output=source,fstype,size,used,avail,target /data
cap disk_nfs df -B1 --output=source,fstype,size,used,avail,target "$(dirname "$NFS_ROOT")"
cap numa bash -c 'numactl -H 2>/dev/null || lscpu | grep -i numa'
cap hugepages bash -c 'grep -i huge /proc/meminfo; cat /sys/kernel/mm/transparent_hugepage/enabled'
cap ulimit bash -c 'ulimit -a'
cap slurm bash -c 'echo SLURM_JOB_ID=${SLURM_JOB_ID:-}; squeue -h -w "$(hostname -s)" -o "%i %u %q %T %M %N" 2>/dev/null'

# ---- GPUs --------------------------------------------------------------------------------------------------------
if [ "$SKIP_GPU" = 0 ]; then
  if have rocm-smi; then
    cap gpu_product rocm-smi --showproductname
    cap gpu_vram rocm-smi --showmeminfo vram --json
    cap gpu_use rocm-smi --showuse --showpower --showtemp
    cap gpu_bus rocm-smi --showbus
    cap gpu_topo rocm-smi --showtopo
    cap gpu_pids rocm-smi --showpids
  else note "rocm-smi not on host PATH"; fi
  if have amd-smi; then
    cap amdsmi_list amd-smi list
    cap amdsmi_topo amd-smi topology
  fi
  cap gpu_pci_numa bash -c 'for d in /sys/bus/pci/devices/*; do c=$(cat $d/class 2>/dev/null); case $c in 0x038000|0x030000|0x120000) v=$(cat $d/vendor); [ "$v" = 0x1002 ] && echo "$(basename $d) class=$c numa=$(cat $d/numa_node) dev=$(cat $d/device)";; esac; done'
  cap kfd bash -c 'ls -l /dev/kfd /dev/dri; getent group video render'
else note "GPU queries skipped (--skip-gpu)"; fi

# ---- RDMA / NICs -------------------------------------------------------------------------------------------------
if have ibv_devices; then
  cap ibv_devices ibv_devices
  cap ibv_devinfo ibv_devinfo -v
else note "ibv_devices not on host PATH (try inside an image)"; fi
cap ib_sysfs bash -c 'for d in /sys/class/infiniband/*; do [ -e "$d" ] || continue; n=$(basename $d); pci=$(basename $(readlink -f $d/device)); echo "$n pci=$pci numa=$(cat $d/device/numa_node 2>/dev/null) netdev=$(ls $d/device/net 2>/dev/null | tr "\n" " ")"; for p in $d/ports/*; do echo "  port $(basename $p) state=$(cat $p/state) phys=$(cat $p/phys_state) rate=$(cat $p/rate) link_layer=$(cat $p/link_layer 2>/dev/null)"; done; done'
cap netdev bash -c 'ip -br addr; echo; ip -br link'
cap lspci_net bash -c 'lspci -D | grep -i -E "ethernet|infiniband|network"'
cap lspci_tree bash -c 'lspci -tv'

# ---- docker ------------------------------------------------------------------------------------------------------
if [ "$SKIP_DOCKER" = 0 ] && have docker; then
  cap docker_info docker info
  cap docker_images docker images --digests --format '{{.Repository}}:{{.Tag}} {{.Digest}} {{.Size}} {{.CreatedSince}}'
  cap docker_ps docker ps -a --format '{{.Names}}|{{.Image}}|{{.Status}}|{{.RunningFor}}'
  cap docker_root_df bash -c 'df -B1 --output=size,used,avail,target "$(docker info -f "{{.DockerRootDir}}")"'
  # foreign containers holding GPUs (report only)
  gpu_containers > "$R/foreign_gpu_containers.txt" 2>&1
  # in-image checks
  for img in "${IMGS[@]}"; do
    safe=$(echo "$img" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-80)
    if ! docker image inspect "$img" >/dev/null 2>&1; then note "in-image: $img not present (pull_images.sh first)"; continue; fi
    cap "inimage_$safe" docker run --rm --name "yihou-dsv41-preflight-$$" --network none --entrypoint bash "$img" -c '
      echo "## pip list (filtered)"; pip list 2>/dev/null | grep -i -E "vllm|aiter|nixl|mooncake|mori|umbp"
      echo "## vllm version"; python3 -c "import vllm; print(vllm.__version__)"
      echo "## mori / umbp server binary"; python3 -c "import mori, os; d=os.path.dirname(mori.__file__); print(d); print([f for f in os.listdir(d) if \"umbp\" in f])"
      echo "## connectors importable"; for m in mooncake nixl mori; do python3 -c "import $m; print(\"$m ok\", getattr($m, \"__file__\", \"\"))" 2>&1 | tail -1; done
      echo "## rdma tools in image"; which ibv_devices ibv_devinfo ib_write_bw 2>&1
      echo "## python / torch"; python3 -c "import sys, torch; print(sys.version); print(torch.__version__, torch.version.hip)"
      echo "## vllm-router"; pip show vllm-router 2>/dev/null | head -2'
  done
else note "docker queries skipped"; fi
# GPU index map (HIP order vs rocm-smi order; chaox saw them differ). Needs GPUs in a container -> run by hand:
note "HIP-vs-smi index map (manual, needs GPU container): docker run --rm --device /dev/kfd --device /dev/dri --group-add video --entrypoint python3 <image> -c 'import torch; [print(i, torch.cuda.get_device_properties(i).pci_bus_id if hasattr(torch.cuda.get_device_properties(i),\"pci_bus_id\") else torch.cuda.get_device_properties(i)) for i in range(torch.cuda.device_count())]'"
note "per-rail RDMA bandwidth probe (SYNTHESIS E0/M6) is NOT run by this script."
# NFS writability as container root (root squash) is checked by serve.sh's entry (.write_test); record host side here
cap nfs_write bash -c "mkdir -p '$NFS_ROOT/runs' && touch '$NFS_ROOT/runs/.preflight_touch_$HOST' && ls -ld '$NFS_ROOT/runs' && mount | grep -E ' $(dirname "$NFS_ROOT") | /perf_apps ' "
cap local_root bash -c "mkdir -p '$LOCAL_ROOT' && df -B1 '$LOCAL_ROOT'"

# ---- summarize ---------------------------------------------------------------------------------------------------
if have python3; then
  python3 - "$OUT" "$HOST" "$TS" <<'PY'
import json, re, sys, pathlib
out, host, ts = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
R = out / "raw"
def rd(n):
    p = R / f"{n}.txt"
    return p.read_text(errors="replace") if p.exists() else ""
f = {"host": host, "utc": ts, "raw_dir": str(R)}
m = rd("meminfo")
def kb(k):
    x = re.search(rf"^{k}:\s+(\d+) kB", m, re.M); return int(x.group(1)) * 1024 if x else None
f["host_dram_total_B"], f["host_dram_available_B"] = kb("MemTotal"), kb("MemAvailable")
def dfline(n):
    lines = [l for l in rd(n).splitlines() if l and not l.startswith(("$", "[rc", "Filesystem"))]
    if not lines: return None
    p = lines[-1].split()
    try: return {"source": p[0], "fstype": p[1], "size_B": int(p[2]), "used_B": int(p[3]), "avail_B": int(p[4]), "target": p[5]}
    except Exception: return {"raw": lines[-1]}
f["dev_shm"], f["data"], f["nfs"] = dfline("shm"), dfline("disk_data"), dfline("disk_nfs")
try:
    v = json.loads(rd("gpu_vram").split("\n", 1)[1].rsplit("[rc", 1)[0])
    f["gpu_vram"] = {k: {kk: vv for kk, vv in d.items() if "VRAM" in kk} for k, d in v.items() if k.startswith("card")}
except Exception: f["gpu_vram"] = "unparsed; see raw/gpu_vram.txt"
f["gpu_count_smi"] = rd("gpu_product").count("Card Series") or rd("gpu_product").count("Card series")
f["gpu_pci_numa"] = [l for l in rd("gpu_pci_numa").splitlines() if "numa=" in l]
f["ib"] = [l for l in rd("ib_sysfs").splitlines() if l.strip() and not l.startswith(("$", "[rc"))]
f["ibv_devices"] = [l.split()[0] for l in rd("ibv_devices").splitlines()[3:] if l.strip() and not l.startswith("[rc")]
f["foreign_gpu_containers"] = [l for l in rd("foreign_gpu_containers").splitlines() if l.startswith("FOREIGN")]
f["our_gpu_containers"] = [l for l in rd("foreign_gpu_containers").splitlines() if l.startswith("OURS")]
f["docker_images"] = [l for l in rd("docker_images").splitlines() if l and not l.startswith(("$", "[rc"))]
f["in_image"] = {p.stem: p.read_text(errors="replace")[:4000] for p in R.glob("inimage_*.txt")}
f["notes"] = rd("notes").splitlines()
umbp_gb = 449e9
if f["host_dram_available_B"]:
    f["umbp_449GB_instances_fit_by_MemAvailable"] = int(f["host_dram_available_B"] // umbp_gb)
(out / "facts.json").write_text(json.dumps(f, indent=1) + "\n")
G = lambda b: f"{b/2**30:,.0f} GiB" if isinstance(b, int) else "?"
md = [f"# Preflight {host} {ts}", "",
      f"- host DRAM total {G(f['host_dram_total_B'])}, available {G(f['host_dram_available_B'])}; "
      f"449 GB UMBP pools that fit (MemAvailable only, no headroom): {f.get('umbp_449GB_instances_fit_by_MemAvailable','?')}",
      f"- /dev/shm: {f['dev_shm']}", f"- /data: {f['data']}", f"- NFS: {f['nfs']}",
      f"- GPUs (rocm-smi Card Series lines): {f['gpu_count_smi']}", "- GPU VRAM:", "```", json.dumps(f["gpu_vram"], indent=1), "```",
      "- GPU PCI/NUMA:", *[f"  - {l}" for l in f["gpu_pci_numa"]],
      "- RDMA (sysfs):", *[f"  - {l}" for l in f["ib"]],
      f"- ibv_devices: {f['ibv_devices']}",
      "- FOREIGN GPU containers (report only):", *([f"  - {l}" for l in f["foreign_gpu_containers"]] or ["  - none"]),
      "- our GPU containers:", *([f"  - {l}" for l in f["our_gpu_containers"]] or ["  - none"]),
      "- docker images:", *[f"  - {l}" for l in f["docker_images"]],
      "- in-image checks:", *[f"### {k}\n```\n{v}\n```" for k, v in f["in_image"].items()],
      "- notes:", *[f"  - {l}" for l in f["notes"]],
      "", f"Raw outputs: {R} (topology: raw/gpu_topo.txt, raw/lspci_tree.txt, raw/ibv_devinfo.txt)"]
(out / "facts.md").write_text("\n".join(md) + "\n")
print(out / "facts.md")
PY
else
  echo "python3 missing on host: raw files only in $R"
fi
echo "preflight: $OUT"
