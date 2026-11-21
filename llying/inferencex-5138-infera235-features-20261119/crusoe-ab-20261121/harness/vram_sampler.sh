#!/usr/bin/env bash
# First line "total <MiB per GPU>", then every 20 s: epoch seconds and VRAM in use per GPU in MiB (sysfs card order;
# sysfs counts the GEM KV pools, which hipMemGetInfo does not). Runs in a container so it outlives the spur step that
# started it on a remote node. Usage: vram_sampler.sh OUT_FILE
cards=()
for d in /sys/class/drm/card*/device; do
    [ -r "$d/mem_info_vram_used" ] && cards+=("$d")
done
line=total
for d in "${cards[@]}"; do line+=" $(($(cat "$d/mem_info_vram_total") >> 20))"; done
echo "$line" >>"$1"
while :; do
    line=$(date +%s)
    for d in "${cards[@]}"; do line+=" $(($(cat "$d/mem_info_vram_used") >> 20))"; done
    echo "$line" >>"$1"
    sleep 20
done
