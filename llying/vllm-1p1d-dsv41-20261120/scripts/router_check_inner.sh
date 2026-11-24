#!/usr/bin/env bash
# Runs inside rust:1-bookworm as root: install libclang (bindgen), then drop to the host user so cargo fmt writes
# files the user owns.
set -euo pipefail
apt-get update -qq && apt-get install -y -qq --no-install-recommends libclang-dev > /dev/null
for i in 1 2 3; do timeout 300 rustup component add clippy rustfmt && break; ((i < 3)) || exit 1; sleep 10; done
LIBCLANG_PATH="$(dirname "$(find /usr/lib -name 'libclang.so*' | head -1)")"
export LIBCLANG_PATH HOME=/tmp CARGO_HOME=/tmp/ch CARGO_TARGET_DIR=/tmp/target
mkdir -p /tmp/ch && chown "$HOST_UID:$HOST_GID" /tmp/ch
exec setpriv --reuid="$HOST_UID" --regid="$HOST_GID" --clear-groups bash -c '
    cd /work/rust && cargo --version && cargo clippy --version
    cargo fmt --all && cargo clippy --locked --all-targets -- -D warnings && cargo test --locked'
