#!/usr/bin/env bash
# swap_dev_test.py (or SCRIPT) cases in one container on GPUS (default 0,1,2,3), one process per case; a fault shows
# as a non-zero rc.   swap_dev_cases.sh OUT "ARGS" ...
R=${R:-/home/liyingli/bench_agentx/vllm-trackB-20261118}
out=${1:?out}; shift
IMAGE=${IMAGE:-infera-vllm:v0340-0c5e5e89}
gids=()
for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
cases=""
for c in "$@"; do cases+="echo \"== $c\"; python3 /t/${SCRIPT:-swap_dev_test.py} $c 2>&1 | grep -vE \"^\\[aiter\\]|^WARNING|^INFO\" | tail -4; echo \"rc=\$?/\${PIPESTATUS[0]}\"; "; done
docker rm -f vpd-swapdev > /dev/null 2>&1
timeout "${TIMEOUT:-600}" docker run --rm --name vpd-swapdev --network host --ipc host \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband "${gids[@]}" --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 -v "$R/scripts:/t:ro" ${MOUNT_ADD:-} \
    -e HIP_VISIBLE_DEVICES="${GPUS:-0,1,2,3}" -e MC_DISABLE_HIP=1 -e MOONCAKE_DISABLE_HIP_DMABUF=0 -e MC_HIP_KFD_PIN_FIX=0 \
    -e INFERA_AMD_GEM_ALLOCATOR=/opt/infera-gem/libhip_gem_alloc.so -e LD_PRELOAD=/opt/infera-gem/libhip_gem_alloc.so \
    ${ENV_ADD:-} --entrypoint bash "$IMAGE" -c "$cases" > "$out" 2>&1
echo "[docker rc=$?]" >> "$out"
cat "$out"
exit 0
