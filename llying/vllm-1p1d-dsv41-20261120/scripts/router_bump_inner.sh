#!/usr/bin/env bash
# Runs inside rust:1-bookworm as root: toolchain bits, then as the host user bump hyper past the v1.10 chunked-
# terminator regression (hyperium/hyper#4143) and run the CI gate.
set -euo pipefail
apt-get update -qq && apt-get install -y -qq --no-install-recommends libclang-dev > /dev/null
for i in 1 2 3; do timeout 300 rustup component add clippy rustfmt && break; ((i < 3)) || exit 1; sleep 10; done
LIBCLANG_PATH="$(dirname "$(find /usr/lib -name 'libclang.so*' | head -1)")"
export LIBCLANG_PATH HOME=/tmp CARGO_HOME=/tmp/ch CARGO_TARGET_DIR=/tmp/target HYPER_TO
mkdir -p /tmp/ch && chown "$HOST_UID:$HOST_GID" /tmp/ch
exec setpriv --reuid="$HOST_UID" --regid="$HOST_GID" --clear-groups bash -c '
    set -o pipefail
    cd /work/rust
    echo "== before: $(grep -A1 "^name = \"hyper\"$" Cargo.lock | tail -1)"
    cargo update -p hyper --precise "$HYPER_TO" 2>&1 | tail -20 || exit 1
    echo "== after: $(grep -A1 "^name = \"hyper\"$" Cargo.lock | tail -1)"
    cargo tree -i hyper -e normal --depth 1 2>/dev/null | head -12
    cargo fmt --all && cargo clippy --locked --all-targets -- -D warnings && cargo test --locked'
