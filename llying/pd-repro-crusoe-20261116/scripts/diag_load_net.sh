#!/bin/bash
# (1) why docker load is slow: the load client's read rate, dockerd's writes, the disks; (2) VM or not, NIC identity,
# addressing / routes / GIDs per ionic port, GPU <-> NIC PCI layout.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH
T=/shared_nfs/liyingli/images/infera-sglang-v0522-fdbafb60.tar
echo "===== (1) docker load"
pid=$(pgrep -f "docker load -i" | head -1)
dpid=$(pgrep -x dockerd | head -1)
echo "load pid=$pid etime=$(ps -o etime= -p $pid 2>/dev/null) dockerd=$dpid"
docker info 2>/dev/null | grep -E "Storage Driver|Docker Root Dir|Server Version|Images"
df -h /var/lib/docker 2>/dev/null | tail -1
r0=$(awk '/^rchar/ {print $2}' /proc/$pid/io 2>/dev/null); w0=$(awk '/^write_bytes/ {print $2}' /proc/$dpid/io 2>/dev/null)
d0=$(awk '$3 ~ /^(vda|nvme0n1|dm-0)$/ {print $3, $6, $10}' /proc/diskstats)
sleep 20
r1=$(awk '/^rchar/ {print $2}' /proc/$pid/io 2>/dev/null); w1=$(awk '/^write_bytes/ {print $2}' /proc/$dpid/io 2>/dev/null)
d1=$(awk '$3 ~ /^(vda|nvme0n1|dm-0)$/ {print $3, $6, $10}' /proc/diskstats)
echo "load client read: $(( ${r1:-0} / 1048576 )) MiB so far of $(( $(stat -c %s $T) / 1048576 )) MiB, rate $(( (${r1:-0} - ${r0:-0}) / 20 / 1048576 )) MiB/s"
echo "dockerd disk writes: $(( (${w1:-0} - ${w0:-0}) / 20 / 1048576 )) MiB/s"
paste <(echo "$d0") <(echo "$d1") | awk '{printf "disk %s: read %d MiB/s, write %d MiB/s\n", $1, ($5 - $2) * 512 / 20 / 1048576, ($6 - $3) * 512 / 20 / 1048576}'
top -b -n1 -o %CPU 2>/dev/null | sed -n 7,13p
echo "-- NFS: mount options and a 2 GiB read of another region of the tar (O_DIRECT, bypasses the page cache)"
grep " /shared_nfs " /proc/mounts
timeout 120 dd if=$T of=/dev/null bs=16M count=128 skip=1200 iflag=direct 2>&1 | tail -1
echo "-- local disks: 2 GiB direct write + read on /mnt/m2m_nobackup and on the root disk (/var/tmp)"
for d in /mnt/m2m_nobackup/liyingli_pdr /var/tmp; do
    mkdir -p $d; f=$d/.ddtest.$$
    printf "%s write: " $d; timeout 120 dd if=/dev/zero of=$f bs=16M count=128 oflag=direct 2>&1 | tail -1
    printf "%s read : " $d; timeout 120 dd if=$f of=/dev/null bs=16M iflag=direct 2>&1 | tail -1
    rm -f $f
done
lsblk -o NAME,SIZE,TYPE,ROTA,MOUNTPOINT 2>/dev/null | head -12
echo "===== (2) virtualization"
systemd-detect-virt 2>&1
for f in sys_vendor product_name product_version bios_vendor bios_version board_vendor; do echo "dmi $f: $(cat /sys/class/dmi/id/$f 2>/dev/null)"; done
lscpu | grep -iE "hypervisor|virtualization|model name|^cpu\(s\)|numa node"
grep -m1 -o -w hypervisor /proc/cpuinfo
lspci -nn 2>/dev/null | grep -iE "virtio|qemu|red hat|bochs" | head -8
echo "-- PCI devices: GPUs and NICs (vendor:device, PF/VF, link)"
lspci -nn -D 2>/dev/null | grep -iE "1dd8|Pensando|Display|Processing accelerators|1002:75|Mellanox" | head -20
for d in /sys/class/infiniband/ionic_*; do
    b=$(basename $(readlink -f $d/device))
    echo "$(basename $d) $b vendor=$(cat $d/device/vendor) device=$(cat $d/device/device) sriov_numvfs=$(cat $d/device/sriov_numvfs 2>/dev/null || echo -) physfn=$( [ -e $d/device/physfn ] && echo yes || echo no) link=$(cat $d/device/current_link_speed 2>/dev/null)/x$(cat $d/device/current_link_width 2>/dev/null) fw=$(cat $d/fw_ver 2>/dev/null) node_guid=$(cat $d/node_guid 2>/dev/null)"
done
lspci -tv 2>/dev/null | head -40
echo "-- GPU index -> PCI bus (rocm-smi) and the XGMI / PCIe topology rocm-smi reports"
rocm-smi --showbus 2>/dev/null | grep -E "GPU\[" | head -8
rocm-smi --showtopotype 2>/dev/null | sed -n 1,14p
rocm-smi --showtoponuma 2>/dev/null | grep -E "GPU\[" | head -8
echo "-- ionic netdevs: addresses, MTU, routes, speed"
for d in /sys/class/infiniband/ionic_*; do
    n=$(ls $d/device/net 2>/dev/null | head -1); ib=$(basename $d)
    echo "## $ib netdev=$n mtu=$(cat /sys/class/net/$n/mtu 2>/dev/null) mac=$(cat /sys/class/net/$n/address 2>/dev/null) speed=$(cat /sys/class/net/$n/speed 2>/dev/null) oper=$(cat /sys/class/net/$n/operstate 2>/dev/null)"
    ip -o addr show dev $n 2>/dev/null | awk '{print "   addr", $3, $4}'
    ip -o -4 route show dev $n 2>/dev/null | sed 's/^/   r4 /' | head -4
    ip -o -6 route show dev $n 2>/dev/null | grep -v "^fe80\|^multicast\|^ff00" | sed 's/^/   r6 /' | head -4
    for g in 0 1 2 3; do v=$(cat $d/ports/1/gids/$g 2>/dev/null); t=$(cat $d/ports/1/gid_attrs/types/$g 2>/dev/null); [ -n "$v" ] && [ "$v" != "0000:0000:0000:0000:0000:0000:0000:0000" ] && echo "   gid$g $v $t"; done
done
ip rule show 2>/dev/null | head -12
ip -6 rule show 2>/dev/null | head -12
command -v lldpctl >/dev/null && lldpctl 2>/dev/null | grep -E "Interface|SysName|PortDescr|PortID" | head -40 || echo "no lldpctl"
command -v rdma >/dev/null && rdma link show 2>/dev/null | head -12
