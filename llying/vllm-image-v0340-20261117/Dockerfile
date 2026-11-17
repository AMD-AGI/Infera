# Infera + vLLM v0.34.0 (ROCm 10.0) engine image for the GLM-5.2 AgentX PD experiments on this branch.
# Build from the repository root:  docker build -f llying/vllm-image-v0340-20261117/Dockerfile -t <tag> .
# Experimental counterpart of deploy/docker/Dockerfile.vllm (still on v0.25.1): the base's aiter (v0.1.28.post1) is
# kept (Dockerfile.vllm would rebuild v0.1.16), no vLLM source patch is applied yet (each patches/vllm/ file is
# judged against v0.34.0 first), Mooncake is v0.3.14 + Mooncake#4540/#4541 (build_mooncake.sh), and the amdgpu GEM
# KV allocator for hosts that charge dma-buf-pinned VRAM twice (crusoe, amdgpu-dkms 6.16) is built in.
ARG VLLM_BASE_IMAGE=vllm/vllm-openai-rocm:v0.34.0@sha256:3cb773d23f063afa86291ef75773cdd486154ddea94c213300f4ed7cc42b7af1
FROM ${VLLM_BASE_IMAGE}

# ROCm ships as wheels under site-packages here; tools and CMake configs that look in /opt/rocm find it through this.
RUN [ -e /opt/rocm ] || ln -s "${ROCM_PATH}" /opt/rocm

# ---- Mooncake build dependencies (the base has libibverbs-dev, libnuma-dev, libmsgpack-dev, libfmt-dev) ----
RUN apt-get update -y && apt-get install -y --no-install-recommends \
        libgoogle-glog-dev libgflags-dev libjsoncpp-dev libunwind-dev libboost-all-dev libssl-dev \
        libyaml-cpp-dev libcurl4-openssl-dev libzstd-dev liburing-dev libjemalloc-dev libzmq3-dev pybind11-dev \
        binutils git \
    && rm -rf /var/lib/apt/lists/*

COPY llying/vllm-image-v0340-20261117/patches/ /tmp/mc-patches/
COPY llying/vllm-image-v0340-20261117/build_mooncake.sh /tmp/build_mooncake.sh
RUN bash /tmp/build_mooncake.sh && rm -rf /tmp/mc-patches /tmp/build_mooncake.sh

# ---- amdgpu GEM KV allocator (torch CUDAPluggableAllocator + LD_PRELOAD shim), with the device-number fix ----
COPY llying/vllm-image-v0340-20261117/gem/hip_gem_alloc.c /opt/infera-gem/hip_gem_alloc.c
RUN cd /opt/infera-gem && gcc -O2 -Wall -shared -fPIC -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include hip_gem_alloc.c \
        -o libhip_gem_alloc.so -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64 -libverbs -lpthread -ldl \
    && test -s libhip_gem_alloc.so

# ---- infera (engine wrapper, router, kvd); keep every package the base already pins ----
WORKDIR /opt/infera
COPY pyproject.toml README.md ./
COPY infera ./infera
RUN pip freeze | grep -v -E '^(-e |infera)' > /tmp/base-constraints.txt \
    && pip install --no-cache-dir -c /tmp/base-constraints.txt ".[vllm]" \
    && rm -f /tmp/base-constraints.txt \
    && python3 -c "import infera.engine.vllm.args, infera.server; import vllm; print('infera ok, vllm', vllm.__version__)"

# ---- Rust router (--router-backend rust); toolchain removed in the same layer ----
COPY rust ./rust
RUN set -eu; \
    export LIBCLANG_PATH="$(dirname "$(find /opt/rocm/ /usr/lib -name 'libclang.so*' 2>/dev/null | head -1)")"; \
    case "$LIBCLANG_PATH" in ""|".") apt-get update && apt-get install -y --no-install-recommends libclang-dev \
        && rm -rf /var/lib/apt/lists/* && export LIBCLANG_PATH="$(dirname "$(find /usr/lib -name 'libclang.so*' | head -1)")";; esac; \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal; \
    . "$HOME/.cargo/env"; \
    ( cd rust && cargo build --release --bin infera-router ); \
    cp rust/target/release/infera-router /usr/local/bin/infera-router; \
    rm -rf rust "$HOME/.cargo" "$HOME/.rustup"; \
    sed -i '/\.cargo\/env/d' "$HOME/.profile" "$HOME/.bashrc" 2>/dev/null || true; \
    /usr/local/bin/infera-router --help > /dev/null

ENTRYPOINT []
CMD ["/bin/bash"]
