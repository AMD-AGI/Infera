GPU=0000:75:00.0
echo "===GPU $GPU (amdgpu, card1) BARs via lspci==="
lspci -v -s $GPU 2>/dev/null | grep -iE 'Region|Memory at|prefetchable|size='
echo
echo "===GPU BAR sizes via sysfs resource (start end flags)==="
awk 'NR<=6{sz=($2-$1+1); printf "BAR%d: %.2f GiB  (flags %s)\n",NR-1, sz/1073741824.0, $3}' /sys/bus/pci/devices/$GPU/resource
echo
echo "===find ionic_0 NIC PCI addr==="
NIC=$(basename $(readlink -f /sys/class/infiniband/ionic_0/device))
echo "ionic_0 -> $NIC"
echo "===ionic NIC BARs==="
lspci -v -s $NIC 2>/dev/null | grep -iE 'Region|Memory at|prefetchable|size='
awk 'NR<=6{sz=($2-$1+1); printf "BAR%d: %.2f MiB  (flags %s)\n",NR-1, sz/1048576.0, $3}' /sys/bus/pci/devices/$NIC/resource 2>/dev/null
echo
echo "===amdgpu VRAM total for context==="
cat /sys/class/drm/card1/device/mem_info_vram_total
