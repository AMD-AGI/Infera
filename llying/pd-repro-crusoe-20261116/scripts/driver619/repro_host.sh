#!/bin/bash
# repro_node.sh without docker (the node's containerd is hung): kfd_pin_repro.py on the host ROCm 7.2.0 / MOFED verbs.
OUT=$1
CASES=${2:-"0:kfd:dmabuf:40 1:kfd:peermem:40 2:kfd:dmabuf:150 3:kfd:peermem:150 4:gem:dmabuf:40 5:gem:dmabuf:150 6:kfd:peermem:40:all 7:kfd:none:40"}
K=/home/liyingli/bench_agentx/pd-repro-20261116/driver619
mkdir -p "$OUT"; h=$(hostname -s)
echo "$(date -u +%T) $h amdgpu $(cat /sys/module/amdgpu/version) kernel $(uname -r) (host, no docker)" | tee "$OUT/$h.node"
pids=()
for c in $CASES; do
    IFS=: read -r g alloc reg gib nics <<<"$c"
    [ "$nics" = all ] && nics=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
    nics=${nics:-ionic_$g}; tag="$h-g$g-$alloc-$reg-$gib"
    ( ROCR_VISIBLE_DEVICES=$g RDMAV_FORK_SAFE=1 LD_LIBRARY_PATH=/opt/rocm/lib timeout 1200 \
        python3 $K/${REPRO:-kfd_pin_repro.py} --gpu 0 --alloc $alloc --reg $reg --gib $gib --nics $nics --tag $tag; echo "exit $?" ) > "$OUT/$tag.log" 2>&1 &
    pids+=($!)
done
wait "${pids[@]}"
python3 $K/summarize.py "$OUT" > "$OUT/$h.summary.md" 2>&1; cat "$OUT/$h.summary.md"
echo "$(date -u +%T) $h done"
