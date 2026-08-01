#!/usr/bin/env bash
# Full regression sweep in the container after all three patches.
set -u
export PATH="$HOME/.cargo/bin:$PATH"
LIBCLANG_PATH="$(dirname "$(find /opt/rocm* /usr/lib /usr/lib64 -name 'libclang.so*' 2>/dev/null | head -1)")"
case "$LIBCLANG_PATH" in ""|".") LIBCLANG_PATH=/opt/rocm/llvm/lib;; esac
export LIBCLANG_PATH
echo "############ cargo: WHOLE router crate ############"
cd /opt/infera/rust
cargo test --offline --release -p infera-router 2>&1 | grep -E "^test result|running [0-9]+ tests|error|FAILED" | head -20
echo "############ pytest: whole unit + engine tree ############"
cd /opt/infera
python3 -m pytest tests/unit tests/engine -p no:cacheprovider -q 2>&1 | tail -12
