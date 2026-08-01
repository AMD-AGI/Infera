#!/usr/bin/env bash
# Node-side driver: verify group E in the BUILT infera/engine-sglang:merged-e.
#
# Runs a throwaway container from the image (never the live merged_run, whose
# in-place patches would answer for the image's own content), and separately
# checks the source the build consumed, since the Rust half cannot be read back
# out of the stripped binary.
set -u
IMAGE="${IMAGE:-infera/engine-sglang:merged-e}"
CTR="${CTR:-vrfy_e}"
SRC="${SRC:-/root/merged_e_src}"

echo "=== source the build consumed ($SRC) ==="
for pat in 'fn as_u32_any' 'subscriber_decodes_bigram_tokens_under_mtp' \
           'decodes_sglang_bigram_batch_under_mtp'; do
  n=$(grep -rl "$pat" "$SRC/rust" 2>/dev/null | wc -l)
  printf '  %-46s src=%s ' "$pat" "$n"
  [ "$n" -gt 0 ] && echo OK || echo FAIL
done

echo "=== built image ==="
docker rm -f "$CTR" >/dev/null 2>&1
docker run -d --name "$CTR" --entrypoint sleep "$IMAGE" infinity >/dev/null || {
  echo "could not start a container from $IMAGE" >&2; exit 1; }
trap 'docker rm -f "$CTR" >/dev/null 2>&1' EXIT
sleep 3
docker cp /tmp/_inner_verify_e.sh "$CTR":/tmp/ >/dev/null
docker exec "$CTR" bash /tmp/_inner_verify_e.sh
