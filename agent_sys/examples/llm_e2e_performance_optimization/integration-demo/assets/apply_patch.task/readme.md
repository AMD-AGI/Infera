# apply_patch

Turn a patch set into a set of per-file read-only bind mounts, and prove each one
applies before anything expensive starts.

## Per file, not per tree

The sglang python tree is 87 MB and the AITER tree is 6.9 GB, and a kernel patch
lands in AITER as often as not. Copying either per attempt is waste; copying
AITER is prohibitive. A bind mount of a single file works because sglang is
installed into the image in editable mode, so the interpreter reads the image's
source tree directly rather than a copy under site-packages.

## Pinned by hash, not by commit

Both `/sgl-workspace/sglang` and `/sgl-workspace/aiter` are git repositories, and
neither is usable as a pin: the sglang working tree is dirty relative to its own
HEAD because the image build replaces `python/sglang` wholesale with a PR
overlay, so a `git diff` would carry changes nobody in this pipeline made. What
pins a patch is the sha256 of each file as it exists in the image, plus the image
reference. A mismatch is refused here, which is the check that catches "this
patch was cut against a different build".

## Why it runs before the stock arm

It costs seconds. The stock arm costs twenty minutes. A patch that does not apply
should not be discovered after a baseline nobody will use has been measured.

## Where the files go

Node-local, under the work root — not into the attempt zone. The zone is
discarded with the attempt and the deployment outlives it, so a mount pointing
into the zone would break the first time the container restarted. The handoff
carries its own copy of the same bytes so the record still describes the change
after the node is gone.
