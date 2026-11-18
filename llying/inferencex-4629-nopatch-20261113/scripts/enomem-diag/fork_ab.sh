#!/usr/bin/env bash
# ibv_fork_init (Mooncake always calls it) x the decode's buffer layout: 84 separate 1.73 GB hipMallocs per GPU,
# 1 GiB chunks on all eight NICs from 16 threads, GPUs 0-3 at once.
D=/apps/liyingli/holder-infx/diag; IMG=lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261112
docker image inspect $IMG >/dev/null 2>&1 || IMG=lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261113
NICS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
O=$D/out/$(hostname -s)-fork-$(date -u +%m%dT%H%M); mkdir -p $O
t() {  # NAME FORK_INIT "EXTRA"
    echo "== $1 fork_init=$2 $3"
    docker run --rm --network host --ipc host --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
        -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
        -v $D:$D --entrypoint bash $IMG -c "
            for g in 0 1 2 3; do
                python3 $D/kfd_pin_repro.py --gpu \$g --alloc kfd --reg peermem --buffers 84 --nics $NICS --chunk-gib 1 \
                    --threads $THREADS --fork-init $2 $3 --relaxed 1 --headroom 0 --tag $1 > $O/$1-gpu\$g.jsonl 2>&1 &
            done; wait"
    for g in 0 1 2 3; do echo "  gpu$g: $(grep -E '"step": "(ipc_exported|registered_buffers|error)"' $O/$1-gpu$g.jsonl | cut -c1-260 | tr '\n' ' ')"; done
}
THREADS=128 t P128 1 "--filler-gib 115 --ipc-export 1"
THREADS=256 t P256 1 "--filler-gib 115 --ipc-export 1"
