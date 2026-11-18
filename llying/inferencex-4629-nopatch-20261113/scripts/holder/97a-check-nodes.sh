#!/usr/bin/env bash
# All nodes on amdgpu 6.16: run the PR's configuration as queued. Any other driver (6.19.14 double-charges
# KFD VRAM that the NIC pins): switch the queue to the TEST-ONLY GEM KV variant (alt-gem/: KV pools in GEM BOs,
# dma-buf registration), building libhip_gem_alloc.so for the image first if it is missing. If that build
# fails, fall back to excluding the node and asking resubmit_watch.sh for a new holder.
H=/apps/liyingli/holder-infx
EX=/apps/liyingli/jobs/exclude_nodes.txt
SQ=/apps/liyingli/squash/lmsysorg_sglang-rocm_v0.5.23-rocm720-mi35x-20261117_sha256_e016f33423250be30bde9e0888d159b67b01e00944de1edc11f36dd1f3dd0269.sqsh
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
bad=()
for n in "${nodes[@]}"; do
    v="$(srun --overlap -N1 -n1 -w "$n" cat /sys/module/amdgpu/version 2>/dev/null)"
    echo "$n amdgpu $v"
    case "$v" in 6.16.*) ;; *) bad+=("$n") ;; esac
done
[ ${#bad[@]} -eq 0 ] && { echo "all nodes on 6.16: PR configuration"; exit 0; }
echo "not 6.16: ${bad[*]}: switching to the GEM KV variant"
if [ ! -s "$H/gem/libhip_gem_alloc.so" ]; then
    srun --overlap -N1 -n1 -w "${nodes[0]}" --container-image="$SQ" --no-container-mount-home \
        --container-mounts="$H/gem:/gem" bash -c 'gcc -O2 -Wall -shared -fPIC -D__HIP_PLATFORM_AMD__ \
        -I/opt/rocm/include /gem/src/hip_gem_alloc.c -o /gem/libhip_gem_alloc.so.tmp -L/opt/rocm/lib \
        -lamdhip64 -lhsa-runtime64 -libverbs -lpthread -ldl && mv /gem/libhip_gem_alloc.so.tmp /gem/libhip_gem_alloc.so'
fi
if [ -s "$H/gem/libhip_gem_alloc.so" ]; then
    w="$H/withdrawn/pr-$(date -u +%m%dT%H%M)"; mkdir -p "$w"
    for f in "$H"/queue/97[b-i]-*.sh "$H"/queue/99-stop.sh; do [ -e "$f" ] && mv "$f" "$w/"; done
    cp "$H"/alt-gem/*.sh "$H/queue/"
    echo "queue now: $(ls "$H/queue" | paste -sd' ')"
    exit 0
fi
for n in "${bad[@]}"; do grep -qx "$n" "$EX" || echo "$n" >> "$EX"; done
echo "GEM allocator build failed; excluded ${bad[*]}, asking for a new holder"
cp "$H/running/97a-check-nodes.sh" "$H/queue/97a-check-nodes.sh"
touch "$H/NEED_RESUBMIT" "$H/STOP"
exit 1
