#!/usr/bin/env bash
# Runs INSIDE the container. Applies the three remaining PR56 (non-gfx942)
# patches and tests them where the interpreter and the router binary actually
# live. Verification is against BYTECODE and the REBUILT BINARY, never a source
# grep: a stale .pyc silently answers for a patched .py, and infera-router is a
# compiled artifact that ignores an edited .rs until it is rebuilt.
set -u
fail=0
cd /opt/infera || exit 1

echo "=== 1. apply python + rust sources ==="
tar xzf /tmp/liying_rest.tgz -C /opt/infera
# The image deletes rust/ after building the binary; restore the tree so cargo
# can rebuild, then overlay the patched kv_event.rs from the same tarball.
rm -rf /opt/infera/rust
tar xzf /tmp/rust_src.tgz -C /opt/infera
tar xzf /tmp/liying_rest.tgz -C /opt/infera rust/router/src/kv_event.rs
echo "  as_u32_any occurrences in kv_event.rs: $(grep -c as_u32_any /opt/infera/rust/router/src/kv_event.rs)"

echo "=== 2. recompile the touched python, check BYTECODE ==="
for d in /opt/infera/infera/common /opt/infera/infera/engine/sglang; do
  rm -rf "$d/__pycache__"; python3 -m compileall -q "$d" >/dev/null 2>&1
done
ck() { # label glob identifier
  n=$(grep -rl "$3" $2 2>/dev/null | wc -l | head -1); n=${n:-0}
  printf '  %-46s pyc=%s ' "$1" "$n"
  if [ "$n" -gt 0 ]; then echo OK; else echo FAIL; fail=1; fi
}
ck "net::_reserved_nodeport_range" "/opt/infera/infera/common/__pycache__/net*.pyc" _reserved_nodeport_range
ck "worker::INFERA_SGLANG_READY_TIMEOUT" "/opt/infera/infera/engine/sglang/__pycache__/worker*.pyc" INFERA_SGLANG_READY_TIMEOUT

echo "=== 3. python behavioural checks ==="
python3 - <<'PY' || fail=1
import os
from infera.common.net import free_tcp_port_block, _reserved_nodeport_range
LO, HI = 30000, 32767
assert _reserved_nodeport_range() == (LO, HI), _reserved_nodeport_range()
bases = [free_tcp_port_block(8) for _ in range(40)]
bad = [b for b in bases if b + 7 >= LO and b <= HI]
assert not bad, f"blocks inside the NodePort range: {bad}"
assert len(set(bases)) > 1, "randomised scan start regressed -- all bases identical"
os.environ["INFERA_NODEPORT_RANGE"] = "none"
assert _reserved_nodeport_range() is None
os.environ["INFERA_NODEPORT_RANGE"] = "not-a-range"
assert _reserved_nodeport_range() == (LO, HI), "a typo must keep the guard"
del os.environ["INFERA_NODEPORT_RANGE"]
print("  nodeport guard + randomisation both hold        OK")
PY

echo "=== 4. pytest on the touched suites ==="
ls -la tests/unit/common/test_net_port_block.py tests/engine/sglang/test_ready_timeout.py 2>&1
python3 -m pytest tests/unit/common/test_net_port_block.py \
                  tests/engine/sglang/test_ready_timeout.py \
                  tests/unit/router/test_kv_event_bigram.py \
                  tests/unit/router/test_kv_event_e2e.py \
                  -p no:cacheprovider -q 2>&1 | tail -25
echo "pytest_rc=${PIPESTATUS[0]}"

echo "=== 5. cargo test the rust decoder ==="
export PATH="$HOME/.cargo/bin:$PATH"
# Same libclang discovery the Dockerfile does for onig_sys/bindgen; without it
# the build dies in a build script, not in our code.
LIBCLANG_PATH="$(dirname "$(find /opt/rocm* /usr/lib /usr/lib64 -name 'libclang.so*' 2>/dev/null | head -1)")"
case "$LIBCLANG_PATH" in ""|".") LIBCLANG_PATH=/opt/rocm/llvm/lib;; esac
export LIBCLANG_PATH
echo "  LIBCLANG_PATH=$LIBCLANG_PATH"
ls "$LIBCLANG_PATH"/libclang.so* 2>&1 | head -3
cd /opt/infera/rust
cargo test --offline -p infera-router kv_event 2>&1 | tail -30
echo "cargo_test_rc=${PIPESTATUS[0]}"

echo
if [ "$fail" -eq 0 ]; then echo "=== PY CHECKS PASSED ==="; else echo "=== PY CHECKS FAILED ==="; exit 1; fi
