#!/usr/bin/env bash
# Select and VERIFY the sglang arm inside both legs' containers, for the #33970 A/B.
#
# The image's /sgl-workspace/sglang is a git working tree with the infera patches
# applied as uncommitted modifications, so:
#   ARM=stock    -> `git checkout --` the three files PR #33970 touches
#   ARM=patched  -> restore those modifications
#
# "patched = leave them as built" only holds the FIRST time. Once the stock arm has
# run, `git checkout --` has discarded the uncommitted patch, and there is nothing
# left to leave alone -- the guard then correctly refuses with wait_event=0. So the
# patched arm has to actively restore, which means the patch must be saved BEFORE
# the first stock checkout. save_patch() below does that, once, per container.
# The two arms then differ by exactly those files and nothing else, with no rebuild.
#
# The guard is the point of this script, not the checkout. It counts `wait_event`
# in mooncake/conn.py and REFUSES to continue on a mismatch (stock=0, patched=9).
# Carried over from the previous session's up_singlenode.sh: without it, a
# mislabelled arm gets recorded as a result, which is worse than no result.
#
# Run AFTER the containers exist (cluster.spur.sh up creates them) and BEFORE the
# legs are launched.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="${ARM:?ARM=stock|patched}"
SSH_CMD="${SSH_CMD:-$HERE/spur_ssh.sh}"
CTR="${CTR:-glm52}"
PREFILL_NODE="${PREFILL_NODE:-crsuse2-m2m-237}"
DECODE_NODE="${DECODE_NODE:-crsuse2-m2m-106}"

PR_FILES="python/sglang/srt/disaggregation/common/utils.py \
python/sglang/srt/disaggregation/mooncake/conn.py \
python/sglang/srt/disaggregation/prefill.py"

# The patched arm is restored by re-running infera's own anchor script, NOT by
# replaying a saved `git diff`. Two reasons, both learned the hard way:
#   * "patched = leave it as built" only works until the stock arm runs once; after
#     `git checkout --` the uncommitted patch is gone and there is nothing to leave.
#   * the UPSTREAM PR diff does not apply to this image. The container's sglang is
#     the v0.5.17 base, which predates upstream's `staging_counted` refactor, so
#     `git apply --3way` fails with "patch does not apply" at conn.py:1626 and
#     leaves utils.py conflicted. infera's anchor script targets exactly this tree.
# The two are equivalent in effect for this experiment -- validate_B.py is what
# establishes that, and it is the same relationship #33968 relies on.
PATCH_SCRIPT="${PATCH_SCRIPT:-/patch_mc.py}"
PATCH_SRC="${PATCH_SRC:-/home/yihou/dev/git/infera.upstream.pr.verify/deploy/docker/patches/sglang_disagg/patch_mooncake_early_send_wait_event.py}"

rc=0
for node in "$PREFILL_NODE" "$DECODE_NODE"; do
  if [ "$ARM" = "stock" ]; then
    $SSH_CMD "$node" "docker exec $CTR bash -c 'cd /sgl-workspace/sglang && git checkout -- $PR_FILES'" \
      || { echo "arm: could not revert the PR files on $node" >&2; rc=1; continue; }
  else
    # checkout first so the anchor script always starts from stock: it is idempotent
    # in the sense of refusing to double-apply, not of re-applying over itself.
    $SSH_CMD "$node" "docker cp $PATCH_SRC $CTR:$PATCH_SCRIPT >/dev/null 2>&1; docker exec $CTR bash -c 'cd /sgl-workspace/sglang && git checkout -- $PR_FILES && python3 $PATCH_SCRIPT >/dev/null'" \
      || { echo "arm: could not restore the patch on $node" >&2; rc=1; continue; }
  fi
  n=$($SSH_CMD "$node" "docker exec $CTR bash -c \"grep -c wait_event /sgl-workspace/sglang/python/sglang/srt/disaggregation/mooncake/conn.py\"" 2>/dev/null | tr -d '\r\n ')
  echo "  $node: wait_event occurs ${n}x in mooncake/conn.py  (stock=0, patched=9)"
  case "$ARM:$n" in
    stock:0|patched:9) ;;
    *) echo "  REFUSING: $node is not in the '$ARM' arm (wait_event=${n}) -- a mislabelled A/B is worse than none" >&2; rc=1 ;;
  esac
done

[ $rc -eq 0 ] && echo "arm=$ARM verified on both legs"
exit $rc
