#!/usr/bin/env bash
# Stand in for the kernel-optimization stage: produce a `kernel_patch` against
# the engine image that is actually on the node.
#
# Generated rather than shipped. `analyze-demo`'s mock ships a recorded CSV,
# which is right for a file whose content is the point; a patch's content is
# only meaningful against one exact file, and a recorded diff would stop
# applying the first time the image is rebuilt. Cutting it here against the
# image in front of us means the mock always applies, and that the base hash it
# records is the real one.
set -eu

PKG="${AGENT_SYS_TASK_PACKAGE:?AGENT_SYS_TASK_PACKAGE is unset}"
OUT="${AGENT_SYS_OUTPUT_KERNEL_PATCH:?AGENT_SYS_OUTPUT_KERNEL_PATCH is unset}"

. "$PKG/assets/lib/remote.sh"

CONTAINER_PATH="${IT_SGLANG_ROOT:?}/${IT_PATCH_FILE:?}"
STAGE="$(pwd)/stage"
rm -rf "$STAGE"; mkdir -p "$STAGE"

echo "[seed] image=$IT_IMAGE"
echo "[seed] target=$CONTAINER_PATH symbol=$IT_PATCH_SYMBOL"

# The zone has to be reachable from the compute node, because `docker cp` writes
# there and this body reads it back. $HOME is NFS from the same server here, so
# it is; asserting it turns a wrong --demo-root into a named failure instead of
# "No such file or directory" from a docker subcommand.
require_visible_on_node "$STAGE" "the attempt zone" || exit 1

# `docker create` and not `docker run`: no process starts, no GPU is touched, and
# the image's entrypoint never executes. Extracting one file out of a 64 GB image
# this way takes about a second.
echo "[seed] extracting the stock file from the image"
on "set -e
    CID=\$(docker create '$IT_IMAGE' true)
    trap 'docker rm -f \$CID >/dev/null 2>&1' EXIT
    docker cp \"\$CID:$CONTAINER_PATH\" '$STAGE/stock.py'" || {
  echo "[seed] ABORT: could not extract $CONTAINER_PATH from $IT_IMAGE on $IT_NODE" >&2
  exit 1
}
[ -s "$STAGE/stock.py" ] || { echo "[seed] ABORT: extracted file is empty" >&2; exit 1; }
echo "[seed] extracted $(wc -c < "$STAGE/stock.py") bytes"

exec python3 "$PKG/assets/seed_patch.task/seed.py" \
  --stock "$STAGE/stock.py" \
  --container-path "$CONTAINER_PATH" \
  --symbol "$IT_PATCH_SYMBOL" \
  --image "$IT_IMAGE" \
  --out "$OUT" \
  --package "$PKG"
