#!/usr/bin/env bash
# Extra checks for infera-sglang:v0521-yihou-1010-mc4340 (run inside the image, CPU-only, read-only).
set -uo pipefail
fail=0; ok(){ echo "  OK   $*"; }; bad(){ echo "  FAIL $*"; fail=1; }
so="$(python3 -c 'import mooncake.engine as e; print(e.__file__)')"
echo "== mooncake engine.so: $so"; sha256sum "$so"
s="$(strings "$so")"
for needle in "extends past its allocation" "HipTransport: hipMemGetAddressRange failed" "HIP transport installed" "isHipReachableTarget" "MC_DISABLE_HIP" "Mismatched HIP data transfer method"; do
  n=$(grep -c -- "$needle" <<<"$s"); [ "$n" -gt 0 ] && ok "strings: '$needle' x$n" || bad "strings: '$needle' missing"
done
nm -D "$so" | grep -q ibv_reg_dmabuf_mr && ok "dyn sym ibv_reg_dmabuf_mr" || bad "ibv_reg_dmabuf_mr"
ldd "$so" | grep -qi hsa-runtime64 && ok "links hsa-runtime64" || bad "hsa-runtime64"
python3 -c "from mooncake.engine import TransferEngine; TransferEngine()" && ok "TransferEngine import+ctor" || bad "TransferEngine"
echo "VERIFY_MC4340 fail=$fail"; exit $fail
