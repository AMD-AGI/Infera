#!/usr/bin/env bash
# RLIMIT_MEMLOCK x concurrent peer-memory registration, no CAP_IPC_LOCK (like the pyxis worker containers).
D=/apps/liyingli/holder-infx/diag; IMG=lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261112
docker image inspect $IMG >/dev/null 2>&1 || IMG=lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261113
NICS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
t() {  # NAME MEMLOCK THREADS
    echo "== $1 memlock=$2 threads=$3"
    docker run --rm --network host --ipc host --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render --ulimit memlock=$2:$2 \
        -v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro \
        -v $D:$D --entrypoint bash $IMG -c "grep -i locked /proc/self/limits; python3 $D/kfd_pin_repro.py --gpu 0 --alloc kfd --reg peermem --gib 32 --nics $NICS --threads $3 --relaxed 1 --headroom 0 --tag $1" 2>&1 \
        | grep -E 'locked|"step": "(registered|error)"' | cut -c1-330
}
t L8T16 8424685568 16
t L8T1 8424685568 1
t UNLT16 -1 16
