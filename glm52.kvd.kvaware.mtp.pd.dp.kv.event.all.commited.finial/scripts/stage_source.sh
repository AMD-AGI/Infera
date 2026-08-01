#!/usr/bin/env bash
# Export the merged branch and stage it on both nodes for an on-node build.
#
# WHY EXPORT SOURCE RATHER THAN SHIP AN IMAGE
# -------------------------------------------
# The claim under test is that deploy/docker/Dockerfile.sglang reproduces the
# experiment. Building once and `docker save`/`load`-ing a 28 GB tarball to the
# second node would prove only that the tarball survived the trip -- and the
# tarball would carry whatever local state the build machine had. Each node
# builds from the same source instead.
#
# `git archive` (not a tar of the worktree) so only committed content ships: an
# uncommitted edit that changed the result would otherwise be invisible.
#
# Run from a checkout of the merged branch.
#   REF=yihou.dev.glm52.merged.experiment bash stage_source.sh
set -eu
REF="${REF:-HEAD}"
JUMP="${JUMP:-root@149.28.124.225}"
NODES="${NODES:-chi2879 chi2867}"

echo "=== exporting $REF ($(git rev-parse --short "$REF")) ==="
git archive --format=tar "$REF" | gzip > /tmp/merged_src.tgz
ls -lh /tmp/merged_src.tgz

echo "=== staging via $JUMP ==="
scp -q /tmp/merged_src.tgz "$JUMP":/tmp/
for n in $NODES; do
  ssh -T "$JUMP" "scp -q /tmp/merged_src.tgz $n:/tmp/"
  echo "  staged source on $n"
done

# Stage the node-side scripts too, so REPRODUCE's `bash /tmp/<x>.sh` steps have
# something to run. Pushed over stdin rather than scp'd: scp through the jump
# host fails intermittently under its load, and a silently-missing script shows
# up much later as a confusing "No such file".
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in build_merged.sh reset_merged.sh start_leg.sh cleanlegs.sh \
         wait_ready.sh start_router.sh cache_view.sh restart_replay.sh \
         verify_built_image.sh envsnap.sh stage_probes.sh \
         glm52_leg.sh probe.py prefix_reuse.py needle.py stress_capture.py; do
  for n in $NODES; do
    ssh -T "$JUMP" "ssh -T $n 'cat > /tmp/$f'" < "$KIT/$f"
  done
done
for n in $NODES; do
  got=$(ssh -T "$JUMP" "ssh -T $n 'ls /tmp/*.sh 2>/dev/null | wc -l'")
  echo "  staged scripts on $n (/tmp/*.sh count: $got)"
done

echo "=== staged; now run 'bash /tmp/build_merged.sh' on each node ==="
