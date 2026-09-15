# `apply_patch`

Turn m4's optimised kernel into a set of read-only bind mounts, and prove every
file lands where it is supposed to.

A **program**, not an AI leaf. The judgement — *which* file in the engine the
optimised kernel replaces — was made two stages ago, when m3 wrote the workset's
integration point (M5.1.1); m4 filled it in; what is left here is extract, hash,
apply, compile, copy.

## Why it runs before the expensive things

`apply_patch → integrate_and_verify` is the only edge into the measurement, and
this side of it costs seconds. A patch that will not apply fails here rather than
after the stock arm has spent twenty minutes measuring a baseline nobody will
use.

## What it reads: m4's apply block (M5.1.2)

`seed_patch` is gone. The input is m4's real `kernel_optimization`, which carries

```
items/codes/<packup>/apply/manifest.json
items/codes/<packup>/apply/patches/*.patch     (when an entry uses `patch`)
```

with `schema_version`, `operator_id`, `logical_operator`, `image`, `apply_mode`,
`files[]`, and an optional `runtime_marker` and `expect`. Each `files[]` entry
names a `container_path`, the `base_sha256` of what it replaces, a `change` of
`modify` or `add`, and **exactly one of**:

- `patch` — a unified diff under `apply/patches/`;
- `replacement` — a path, relative to the packup, of the whole file that replaces
  it.

**`replacement` exists because that is what m4 actually produces**: an optimised
kernel file, not a diff. This body generates the diff from stock to replacement,
so `patch_overlay` has one shape and `check_overlay_applies` has one thing to
parse.

`container_path` is written `@SGLANG_ROOT@/srt/...` and not as an absolute path,
because `handoff/locality.py` refuses to seal content naming an absolute path
outside a small allow-list and every container root is outside it. The expansion
table is `assets/lib/container_roots.yaml` and stays in the package, which is
staged rather than published.

## The five things it does, in the order that is the argument

1. **Refuse what this stage cannot do.** `apply_mode: rebuild` fails immediately
   and says why. A kernel that must be compiled — HIP, CK, assembly — needs the
   image rebuilt, and failing loudly here is what stops such a patch being
   mounted, never executed, and reported as no regression.
2. **Pin against the image, not against a commit.** Every file is extracted from
   the image the manifest names and hashed against `base_sha256`. Both git
   repositories in the image are dirty relative to their own HEAD — the build
   replaces the sglang python tree wholesale with a PR overlay — so a commit id
   would pin nothing.
3. **Apply and compile.** A python file that does not compile takes the worker
   down during model import fifteen minutes later, where it reads as a
   model-loading failure and sends the reader to the wrong place.
4. **Copy to node-local disk.** The attempt zone is discarded and the deployment
   outlives it, so a mount pointing into the zone would break the first time the
   container restarted. The hashes are re-checked on the node afterwards.
5. **Carry the environment forward** (G5), inherited from m1 rather than
   re-derived — a stage that rebuilt the record could differ from m1's and
   nothing would notice.

## The one refusal worth naming

If a patch applies cleanly and changes nothing, this body **fails** rather than
producing an empty overlay. Two byte-identical arms make every check downstream
pass for the wrong reason: the flow compares the stock deployment against itself
and reports no regression. `check_overlay_applies` checks the same thing again
from the published record, because this body runs on the login node and the
validator grades what actually got sealed.

## Mock

`mock.sh stage5-integration patch_overlay`, plus a rendered `environment.yaml` —
the sealed handoff predates that record (MOCK-MAP adaptation A).
