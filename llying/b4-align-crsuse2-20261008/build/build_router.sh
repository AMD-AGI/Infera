#!/usr/bin/env bash
# Build infera-router at a commit (default: the aus B4 router, 19a6c1d2) inside
# an Ubuntu 22.04 engine image, so the binary links against the glibc of the
# containers that mount it. TEST_FILTER=<name> also runs matching unit tests.
# Usage: build_router.sh <image> [rev]    Output: ../artifacts/infera-router-<rev:8>
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
IMAGE="$1"
REV="$(git -C "$REPO" rev-parse "${2:-19a6c1d250521e1f698e5cafad83cd014dbcbfd9}")"
OUT="$HERE/../artifacts/infera-router-${REV:0:8}"

src="$(mktemp -d /tmp/b4-router.XXXXXX)"
trap 'docker run --rm -v "$src:/src" --entrypoint rm "$IMAGE" -rf /src/rust; rm -rf "$src"' EXIT
git -C "$REPO" archive "$REV" rust | tar -xf - -C "$src"
docker run --rm --network host -v "$src:/src" -e "TEST_FILTER=${TEST_FILTER:-}" --entrypoint bash "$IMAGE" -c '
    set -eu
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
    . "$HOME/.cargo/env"
    # bindgen needs libclang; as in Dockerfile.sglang, use the one ROCm ships.
    export LIBCLANG_PATH="$(dirname "$(find /opt/rocm* /usr/lib -name "libclang.so*" 2>/dev/null | head -1)")"
    rustc --version
    echo "LIBCLANG_PATH=$LIBCLANG_PATH"
    cd /src/rust
    if [ -n "$TEST_FILTER" ]; then cargo test --release --locked -p infera-router "$TEST_FILTER"; fi
    cargo build --release --locked --bin infera-router
    cp target/release/infera-router /src/infera-router'
mkdir -p "$(dirname "$OUT")"
cp "$src/infera-router" "$OUT"
sha256sum "$OUT"
