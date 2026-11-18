#!/usr/bin/env bash
# Build libhip_gem_alloc.so from a hip_gem_alloc.c in the image (same command as the Dockerfile) into $R/gem/.
#   build_gem.sh [SRC]   (default: the track-B worktree's llying/vllm-image-v0340-20261117/gem/hip_gem_alloc.c)
R=${R:-/home/liyingli/bench_agentx/vllm-trackB-20261118}
src=${1:-/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52-trackB/llying/vllm-image-v0340-20261117/gem/hip_gem_alloc.c}
IMAGE=${IMAGE:-infera-vllm:v0340-0c5e5e89}
mkdir -p "$R/gem"
cp "$src" "$R/gem/hip_gem_alloc.c"
docker run --rm -v "$R/gem:/g" --entrypoint bash "$IMAGE" -c 'cd /g && gcc -O2 -Wall -shared -fPIC -D__HIP_PLATFORM_AMD__ \
    -I/opt/rocm/include hip_gem_alloc.c -o libhip_gem_alloc.so.new -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -libverbs \
    -lpthread -ldl && test -s libhip_gem_alloc.so.new && mv libhip_gem_alloc.so.new libhip_gem_alloc.so && echo built'
md5sum "$R/gem/hip_gem_alloc.c" "$R/gem/libhip_gem_alloc.so"
