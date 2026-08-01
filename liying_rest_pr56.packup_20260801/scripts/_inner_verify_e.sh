#!/usr/bin/env bash
# Runs INSIDE a container started from the BUILT merged-e image.
#
# Verifies group E landed in the ARTIFACT, not the source tree: freshly-compiled
# bytecode for the Python half, and the shipped `infera-router` binary for the
# Rust half. A build log saying a file was copied is not the same as the
# interpreter running it, and the Rust source is deleted after the build -- only
# the binary remains, so it is what must be checked.
set -u
fail=0

echo "=== python: bytecode ==="
for d in /opt/infera/infera/common /opt/infera/infera/engine/sglang; do
  rm -rf "$d/__pycache__" 2>/dev/null; python3 -m compileall -q "$d" >/dev/null 2>&1
done
ck() { n=$(grep -rl "$3" $2 2>/dev/null | wc -l | head -1); n=${n:-0}
  printf '  %-46s pyc=%s ' "$1" "$n"
  if [ "$n" -gt 0 ]; then echo OK; else echo FAIL; fail=1; fi; }
ck "net::_reserved_nodeport_range" "/opt/infera/infera/common/__pycache__/net*.pyc" _reserved_nodeport_range
ck "worker::INFERA_SGLANG_READY_TIMEOUT" "/opt/infera/infera/engine/sglang/__pycache__/worker*.pyc" INFERA_SGLANG_READY_TIMEOUT

echo "=== rust: the SHIPPED binary ==="
# LIMITATION, stated rather than papered over: `as_u32_any` is a small private
# fn that the release profile inlines, and its doc comment is not in the
# artifact, so the binary cannot be grepped the way bytecode can. What is
# checked here is that a router binary exists and runs; the *evidence* that the
# decode is fixed is the control run in round 4 (revert -> the real-socket
# bigram test fails 0 vs 2; restore -> passes), plus the source check the
# node-side driver does against /root/merged_e_src, which is byte-for-byte what
# this build consumed.
ls -la /usr/local/bin/infera-router
/usr/local/bin/infera-router --help >/dev/null 2>&1 && echo "  binary runs                                   OK" \
  || { echo "  binary does not run                          FAIL"; fail=1; }
[ -d /opt/infera/rust ] && echo "  NOTE: rust/ still present (unexpected)" || echo "  rust/ removed by the build, as designed"

echo "=== python behaviour ==="
python3 - <<'PY' || fail=1
from infera.common.net import free_tcp_port_block, _reserved_nodeport_range
LO, HI = 30000, 32767
assert _reserved_nodeport_range() == (LO, HI)
b = [free_tcp_port_block(8) for _ in range(40)]
assert not [x for x in b if x + 7 >= LO and x <= HI], "block inside the NodePort range"
assert len(set(b)) > 1, "randomised scan start regressed"
print("  nodeport guard + randomisation                OK")
import inspect
from infera.engine.sglang.worker import SglangEngine
sig = inspect.signature(SglangEngine._wait_ready)
assert sig.parameters["timeout"].default is None, sig
print("  _wait_ready(timeout=None) -> env-resolved     OK")
PY

echo
if [ "$fail" -eq 0 ]; then echo "=== GROUP E VERIFIED IN THE BUILT IMAGE ==="; else echo "=== VERIFICATION FAILED ==="; exit 1; fi
