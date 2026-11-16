#!/bin/bash
# Node facts for the reproduction: driver and its limit check, peer-memory client, ionic, scratch, GPU/NIC layout.
h=$(hostname -s)
echo "== $h $(date -u +%FT%TZ)"
echo "amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null) kernel $(uname -r)"
echo "-- limit check in the DKMS source (affected if no '+ kfd.vram_pinned' after vram_pin_size)"
grep -n -A4 "uint64_t vram_available =" /usr/src/amdgpu-*/amd/amdgpu/amdgpu_amdkfd_gpuvm.c 2>/dev/null | head -8
ls -d /usr/src/amdgpu-* 2>/dev/null
echo "vram_pin_size_xcp (upstream fix present if > 0): $(grep -c vram_pin_size_xcp /usr/src/amdgpu-*/amd/amdgpu/amdgpu.h 2>/dev/null)"
echo "-- peer memory client / ib_core peer support"
lsmod | grep -iE "amdp2p|peer|ib_core|ionic|ib_uverbs" | awk '{print $1, $3, $4}'
grep -l -i "peer" /sys/module/*/parameters/* 2>/dev/null | head -5
ls /sys/kernel/mm/memory_peers 2>/dev/null && for d in /sys/kernel/mm/memory_peers/*; do echo "$d: $(cat $d/version 2>/dev/null)"; done
echo "-- ionic"
modinfo ionic 2>/dev/null | grep -E "^version|^filename" ; ls -la /lib/x86_64-linux-gnu/libionic* 2>/dev/null
for d in /sys/class/infiniband/*; do n=$(basename $d); printf "%s numa=%s pci=%s " $n "$(cat $d/device/numa_node 2>/dev/null)" "$(basename $(readlink -f $d/device))"; cat $d/ports/1/state 2>/dev/null | tr -d '\n'; echo " gid1=$(cat $d/ports/1/gids/1 2>/dev/null)"; done
echo "-- GPUs (bdf / numa)"
for c in /sys/class/drm/card*/device; do [ -e $c/mem_info_vram_total ] || continue; echo "$(basename $(readlink -f $c)) numa=$(cat $c/numa_node) vram_used=$(( $(cat $c/mem_info_vram_used) >> 20 )) MiB"; done
rocm-smi --showmemuse 2>/dev/null | grep VRAM% | awk '{printf "%s ", $NF} END {print "(VRAM%)"}'
rocm-smi --showpids 2>/dev/null | grep -A4 "KFD process" | tail -3
echo "-- scratch, memory, docker"
for d in /mnt/m2m_nobackup /scratch /tmp /raid /nvme /local; do [ -d $d ] && df -h $d | tail -1; done
free -g | head -2; nproc
docker ps -a --format '{{.Names}} {{.Image}} {{.Status}}' 2>&1 | head
docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' 2>&1 | grep -E "infera|etcd" | head
ip -o -4 route get 1.1.1.1 2>/dev/null
ip -o -4 addr show | awk '{print $2, $4}' | grep -v "^lo" | head
