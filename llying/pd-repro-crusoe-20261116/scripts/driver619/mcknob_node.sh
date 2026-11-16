#!/bin/bash
# Mooncake same-NIC loopback (ionic_4) with knob variants: initiator pinned host memory -> target host or GEM GPU 4 memory.
OUT=$1; R=/home/liyingli/bench_agentx/pd-repro-20261116; K=$R/driver619; SO=$K/gemalloc/build/libhip_gem_alloc.so; mkdir -p $OUT
lib=(-v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro)
common=(--network host --ipc host --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video --group-add render
  --cap-add IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 "${lib[@]}" -v $R:$R -v /tmp:/tmp -e RDMAV_FORK_SAFE=1
  -e MC_GID_INDEX=1 -e MC_MAX_MR_SIZE=1073741824 -e MC_IB_TIMEOUT=18 -e MC_DISABLE_HIP=1 -e MOONCAKE_DISABLE_HIP_DMABUF=0 -e MC_HIP_KFD_PIN_FIX=0 -e MC_BUF_GIB=6)
for v in default MC_SLICE_SIZE=1048576 MC_WORKERS_PER_CTX=8 MC_NUM_QP_PER_EP=8 "MC_SLICE_SIZE=1048576 MC_WORKERS_PER_CTX=8 MC_NUM_QP_PER_EP=8"; do
    ev=(); [ "$v" = default ] || for kv in $v; do ev+=(-e $kv); done
    for tp in host gem; do
        pre=(); [ $tp = gem ] && pre=(-e LD_PRELOAD=$SO -e SGLANG_AMD_GEM_ALLOCATOR=$SO)
        sig=$(mktemp -d /tmp/llying-pdr-kn.XXXX); docker rm -f llying-pdr-kn-t llying-pdr-kn-i >/dev/null 2>&1
        timeout 200 docker run --rm --name llying-pdr-kn-t "${common[@]}" "${ev[@]}" "${pre[@]}" -e ROCR_VISIBLE_DEVICES=4 infera-sglang:v0522-fdbafb60 \
            python3 $K/mc_loop_sweep.py --pool $tp --role target --gib 12 --nic ionic_4 --sig $sig > $OUT/t.log 2>&1 & t=$!
        timeout 200 docker run --rm --name llying-pdr-kn-i "${common[@]}" "${ev[@]}" -e ROCR_VISIBLE_DEVICES=0 infera-sglang:v0522-fdbafb60 \
            python3 $K/mc_loop_sweep.py --pool host --role initiator --nic ionic_4 --sig $sig > $OUT/i.log 2>&1
        wait $t; rm -rf $sig
        echo "[$v] host->$tp: $(grep -a '"step": "sweep"' $OUT/i.log | sed -E 's/.*"big_gbs": ([0-9.]+).*"small_gbs": ([0-9.]+).*/big \1 small \2/' | tr '\n' ';') $(grep -a 'reg_rc": -1\|rror' $OUT/i.log $OUT/t.log | head -1 | cut -c1-80)"
    done
done | tee $OUT/summary.txt
