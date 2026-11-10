#!/bin/bash
# REPORT-20261110 section 6: run mc_stream_repro.py on all 8 GPUs at once (one process each, GPU i on ionic_i) inside
# the infera image with the kit's Mooncake env, under a few variants. Usage on the node: mc_stream_repro.sh [VARIANT...]
#   base       kit env (MOONCAKE_DISABLE_HIP_DMABUF=0, MC_MAX_MR_SIZE=1 GiB), single register_memory of 40 GiB
#   batch      batch_register_memory in 1 GiB chunks
#   nodmabuf   MOONCAKE_DISABLE_HIP_DMABUF=1
#   small      4 GiB only
OUT=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench/out/mc_stream_repro
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
mkdir -p $OUT
ip=$(ip -4 -o addr show dev fenic | awk '{print $4}' | cut -d/ -f1)
for v in "${@:-base batch nodmabuf small}"; do
  for var in $v; do
    extra="-e MOONCAKE_DISABLE_HIP_DMABUF=0"; gb=40; batch=0
    case $var in batch) batch=1 ;; nodmabuf) extra="-e MOONCAKE_DISABLE_HIP_DMABUF=1" ;; small) gb=4 ;; esac
    docker rm -f llying-mc-repro >/dev/null 2>&1
    docker run --rm --name llying-mc-repro --network host --ipc host --shm-size 16g \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        --group-add video --group-add render --cap-add SYS_PTRACE --cap-add IPC_LOCK \
        --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 --ulimit core=0 \
        -v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro -v /perf_apps:/perf_apps \
        $extra -e MC_GID_INDEX=1 -e MC_MAX_MR_SIZE=1073741824 -e MC_IB_TIMEOUT=18 -e RDMAV_FORK_SAFE=1 \
        -e MC_TE_FILTERS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 -e IP=$ip \
        infera-sglang:v0522-fdbafb60 bash -c "
for g in 0 1 2 3 4 5 6 7; do GPU=\$g NIC=ionic_\$g GB=$gb BATCH=$batch POLL=10 timeout 300 python3 $B/mc_stream_repro.py > /tmp/r\$g.log 2>&1 & done
wait
for g in 0 1 2 3 4 5 6 7; do echo \"-- gpu \$g: \$(grep -a -c 'ok\$' /tmp/r\$g.log) ok, \$(grep -a -c -i 'segment\|fault\|Traceback' /tmp/r\$g.log) errors\"; grep -a -E 'register|initialize|Segment|fault|Error' /tmp/r\$g.log | head -4; done
" > $OUT/$var.log 2>&1
    echo "== $var"; grep -a -E "^-- gpu|register|Segment|fault|Error" $OUT/$var.log | cut -c1-200 | head -30
  done
done
