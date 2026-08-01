#!/usr/bin/env bash
# Environment snapshot for the merged-image validation. Runs ON a node.
set -u
echo "=== host ==="
hostname; uname -r; date -u +"%Y-%m-%dT%H:%M:%SZ"
echo "=== cpu / ram ==="
lscpu | grep -E "^Model name|^CPU\(s\):|^Socket" | sed 's/  */ /g'
free -g | head -2
echo "=== gpu ==="
rocm-smi --showproductname 2>/dev/null | grep -E "Card Series|GPU\[" | head -9
echo "  amdgpu: $(cat /sys/module/amdgpu/version 2>/dev/null)"
echo "=== rdma fabric ==="
echo "  ionic_rdma: $(modinfo ionic_rdma 2>/dev/null | awk '/^version/{print $2}')"
ibv_devices 2>/dev/null | tail -n +3
for d in $(ibv_devices 2>/dev/null | tail -n +3 | awk '{print $1}'); do
  s=$(ibv_devinfo -d "$d" 2>/dev/null | awk '/state:/{print $2}' | head -1)
  r=$(ibv_devinfo -d "$d" 2>/dev/null | awk '/active_speed|rate:/{print $0}' | head -1)
  echo "  $d state=$s $r"
done
echo "  host libionic: $(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1)"
echo "=== images ==="
for t in infera/engine-sglang:merged lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x; do
  echo "  $t -> $(docker image inspect "$t" --format '{{.Id}}' 2>/dev/null || echo ABSENT)"
done
echo "  base RepoDigests: $(docker image inspect lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x --format '{{json .RepoDigests}}' 2>/dev/null)"
echo "=== versions inside the merged image ==="
docker run --rm --entrypoint python3 infera/engine-sglang:merged -c \
  "import sglang,torch;print('  sglang',sglang.__version__);print('  torch',torch.__version__)" 2>/dev/null
echo "=== disk ==="
df -h /var/lib/docker /mnt/vast 2>/dev/null | tail -3
