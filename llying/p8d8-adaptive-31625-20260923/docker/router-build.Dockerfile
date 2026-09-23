FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential ca-certificates curl pkg-config libclang-dev && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh && sh /tmp/rustup-init.sh -y --profile minimal --no-modify-path && rm /tmp/rustup-init.sh
ENV PATH=/opt/cargo/bin:$PATH LIBCLANG_PATH=/usr/lib/llvm-14/lib CARGO_BUILD_JOBS=2
WORKDIR /work/rust
