# seed_patch

Stand in for the kernel-optimization stage: produce a `kernel_patch` the rest of
this package can consume.

## Why there is a mock here at all

Stage 4 has not landed, and `analyze-demo`'s design — which specifies
forge-loop's *inputs* in detail — leaves its output undefined beyond "edits in a
git worktree and exports a patch" and a `--result-json` path with no schema. So
this package defines the `kernel_patch` contract (see `steps/patch.yaml`) and
mocks a producer for it. `analyze-demo` does the same thing one stage earlier,
with `seed_table` standing in for `profiling-demo`'s `kernel_scan`.

## Why the patch is generated rather than shipped

`analyze-demo`'s mock ships a recorded CSV, which is right for a file whose
content is the point. A patch's content is only meaningful against one exact
file, and a recorded diff stops applying the first time the image is rebuilt.
Cutting it here, against the image on the node, means the mock always applies and
the base hash it records is the real one.

## Why it changes no arithmetic

Because that makes the expected verdict knowable: correctness deltas exactly
zero, performance deltas inside the noise. It is the only way the validators
downstream can themselves be checked — a comparison that reports a regression on
this input has a fault in the judgement, not in the thing judged.

The cost is stated plainly in the handoff's `watchout`: a green report produced
from this patch is evidence that the pipeline works, not evidence about any real
optimisation.

## The two markers

The patch adds an import-time log line and a first-call log line guarded by a
module-level boolean. They exist so `check_patch_live` can distinguish "the bytes
are mounted" from "the code ran", which are different claims and only the second
one matters. The guard keeps the steady-state cost at one branch — this file is
also the performance baseline, and a marker that logged on every call would move
the number it is supposed to leave alone.

## Picking the target

Default `Glm5NextDecoderLayer.forward`, and it is a package variable because a
real forge patch names its own. Measured on this image, two better-looking
candidates are wrong: `swiglu_clamped` is reached only from the vision path, so a
text request never enters it, and it is `@torch.compile`d, where a logging call
forces a graph break.
