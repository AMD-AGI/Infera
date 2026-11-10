#!/bin/bash
# REPORT-20261110 section 6, option 1: SGLang PD start-up (prefill role, Mooncake over the ionic rails, TP 8) on an
# amdgpu 6.19 node with our image (ROCm 7.2) vs the upstream ROCm 10 image, on a small model (DeepSeek-V3-5layer, FP8,
# same DeepseekV2 model code as GLM-5.2) so one start-up takes minutes. Our image's PD schedulers segfault at start-up on
# 6.19 nodes in c10::hip::HIPStream::stream() (scheduler.py run_event_loop); a non-PD server does not.
# Usage on the node: pd_startup_test.sh infera|rocm10 [TP] [MINUTES] [graph|nograph]  (nograph: the ROCm 10 image's sglang hits an ImportError (smallm_moe_gfx950 _check) in decode graph capture)
which=$1 tp=${2:-8} mins=${3:-15} mode=${4:-graph}
B=/perf_apps/liyingli/bench_agentx/kopt-20261109/bench
OUT=$B/out/pd_startup
mkdir -p $OUT
log=$OUT/$which-tp$tp-$mode-$(date -u +%m%dT%H%M).log
name=llying-pd-test
mounts=(-v /perf_apps:/perf_apps); extra_env=()
case $which in
infera)
    IMAGE=infera-sglang:v0522-fdbafb60
    mounts+=(-v /lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro)   # its entrypoint swaps it in
    entry=() ;;
rocm10)
    IMAGE=lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261021
    # the nodes' docker credentials hold an expired token; the image is public, so pull it anonymously
    docker image inspect $IMAGE >/dev/null 2>&1 || DOCKER_CONFIG=$(mktemp -d) timeout 2400 docker pull $IMAGE | tail -2
    # no swap entrypoint: put the host's ionic provider over the image's copies (host kernel ABI)
    for f in $(docker run --rm --entrypoint bash $IMAGE -c 'ls /usr/lib/x86_64-linux-gnu/libionic.so* /usr/lib/x86_64-linux-gnu/libibverbs/libionic* 2>/dev/null'); do
        h=/lib/x86_64-linux-gnu/$(basename $f); [ -e $h ] || h=$(ls /lib/x86_64-linux-gnu/libibverbs/$(basename $f) 2>/dev/null)
        [ -n "$h" ] && [ -e "$h" ] && mounts+=(-v $(readlink -f $h):$f:ro)
    done
    # its sglang imports a missing _check from kernels/ops/moe/smallm_moe_gfx950 on the MoE path (ImportError at start-up)
    extra_env=(-e SGLANG_ROCM_SMALLM_MOE=0)
    entry=(--entrypoint bash) ;;
*) echo "usage: $0 infera|rocm10 [TP] [MINUTES]"; exit 2 ;;
esac
envs=(-e MC_TE_FILTERS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7 -e MC_GID_INDEX=1
      -e MOONCAKE_DISABLE_HIP_DMABUF=0 -e MC_DISABLE_HIP_TRANSPORT=1 -e MC_ENABLE_DEST_DEVICE_AFFINITY=1
      -e MC_ENABLE_DEST_LOCAL_RAIL=1 -e MC_IB_TIMEOUT=18 -e RDMAV_FORK_SAFE=1 -e MC_MAX_MR_SIZE=1073741824
      -e SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1800 "${extra_env[@]}")
cmd="python3 -m sglang.launch_server --model-path /perf_apps/data/models/DeepSeek-V3-5layer --trust-remote-code
     --tp $tp --host 127.0.0.1 --port 30071 --disaggregation-mode prefill --disaggregation-transfer-backend mooncake
     --disaggregation-ib-device ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
     --disable-custom-all-reduce --mem-fraction-static 0.8"
[ "$mode" = nograph ] && cmd="$cmd --disable-cuda-graph"
# One line: `bash -c` runs each line as its own command (until 11-10 18:55 every run started a TP1 non-PD server from the
# first line only).
cmd="$(tr -s '\n ' '  ' <<<"$cmd")"
docker rm -f $name >/dev/null 2>&1
echo "$(date -u +%T) $which tp$tp image $IMAGE mounts ${mounts[*]}" | tee $log
docker run -d --name $name --network host --ipc host --shm-size 32g \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video --group-add render \
    --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 "${mounts[@]}" "${envs[@]}" "${entry[@]}" $IMAGE -c "$cmd" >/dev/null \
    || { echo "docker run failed" | tee -a $log; exit 1; }
res=timeout
for i in $(seq 1 $((mins * 6))); do
    sleep 10
    L="$(docker logs $name 2>&1)"
    if echo "$L" | grep -aq "crashed with exit code -11\|Segmentation fault"; then res=segfault; break; fi
    if echo "$L" | grep -aq "fired up and ready to roll"; then sleep 60; L="$(docker logs $name 2>&1)"
        echo "$L" | grep -aq "crashed with exit code -11\|Segmentation fault" && res=segfault || res=ready; break; fi
    [ "$(docker inspect -f '{{.State.Running}}' $name 2>/dev/null)" = true ] || { res=exited; break; }
done
docker logs $name > $log.server 2>&1
echo "$(date -u +%T) $which tp$tp: $res" | tee -a $log
echo "server args: $(grep -a -o "'tp_size': [0-9]*\|'disaggregation_mode': '[a-z]*'\|'disable_cuda_graph': [A-Za-z]*" $log.server | head -3 | tr '\n' ' ')" | tee -a $log
grep -a -o "Subprocess scheduler_[0-9]* (pid=[0-9]*) crashed with exit code -\?[0-9]*" $log.server | sort | uniq -c | tee -a $log
grep -a -E "rdma_transport|Relaxed|dmabuf|ionic|HIPStream|Error|error" $log.server | grep -v -E "Ignore import" | sort | uniq -c | sort -rn | head -8 | cut -c1-200 | tee -a $log
docker rm -f $name >/dev/null 2>&1
