#!/usr/bin/env bash
# On one holder node: peer-memory registration at the decode's scale, outside SGLang and Mooncake. hipMalloc'd VRAM
# registered with ibv_reg_mr in 1 GiB MRs on all eight ionic NICs, as Mooncake does for each decode rank's KV.
#   R1  one GPU, 145 GiB, relaxed ordering (Mooncake's access flags)
#   R4  GPUs 0-3 at once, 145 GiB each, relaxed ordering (the 4-rank decode)
#   R4n GPUs 0-3 at once, 145 GiB each, no relaxed ordering
#   F1  one GPU, 120 GiB first filled in 4 MiB pieces with every other one freed, then 145 GiB (fragmented VRAM)
#   F4  as F1 on GPUs 0-3 at once
# Output: one JSON line per step in out/<run>-gpu<N>.jsonl; kernel ionic messages after each run.
D=/apps/liyingli/holder-infx/diag
O=$D/out/$(hostname -s)-$(date -u +%m%dT%H%M)
mkdir -p "$O"
IMG=""
for i in lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261112 lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261113; do
    docker image inspect "$i" >/dev/null 2>&1 && { IMG=$i; break; }
done
if [ -z "$IMG" ]; then
    IMG=lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261112
    anon=$(mktemp -d); echo '{}' > "$anon/config.json"; DOCKER_CONFIG=$anon docker pull -q "$IMG"; rm -rf "$anon"
fi
NICS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
echo "node $(hostname -s) amdgpu $(cat /sys/module/amdgpu/version) image $IMG out $O"
run() {  # RUN_NAME RELAXED "EXTRA REPRO ARGS" GPU...
    local name=$1 relaxed=$2 extra=$3; shift 3
    local gpus="$*"
    docker run --rm --name "llying-regdiag-$name" --network host --ipc host \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video --group-add render \
        --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
        -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
        -v "$D:$D" --entrypoint bash "$IMG" -c "
            for g in $gpus; do
                python3 $D/kfd_pin_repro.py --gpu \$g --alloc kfd --reg peermem --gib 145 --nics $NICS --chunk-gib 1 \
                    --headroom 0 --relaxed $relaxed $extra --tag $name > $O/$name-gpu\$g.jsonl 2>&1 &
            done
            wait"
    echo "== $name ($gpus, relaxed=$relaxed $extra)"
    for g in $gpus; do
        echo "  gpu$g: $(grep -E '"step": "(fragmented|registered|error)"' "$O/$name-gpu$g.jsonl" | cut -c1-300 | tr '\n' ' ')"
        grep -v '^{' "$O/$name-gpu$g.jsonl" | tail -2 | sed 's/^/    /'
    done
    { dmesg -T 2>/dev/null || sudo -n dmesg -T 2>/dev/null; } | grep -E -i 'ionic|peer|ib_umem|enomem' | tail -8 | sed 's/^/  kernel: /'
}
run R1 1 "" 0
run R4 1 "" 0 1 2 3
run R4n 0 "" 0 1 2 3
run F1 1 "--fragment-gib 120 --fragment-mib 4" 0
run F4 1 "--fragment-gib 120 --fragment-mib 4" 0 1 2 3
