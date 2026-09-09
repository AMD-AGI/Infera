# check_patch_shape

Completeness, strong. Is this a patch set anything could apply?

Every rule is decided by looking at a file that either says the thing or does
not — a field is present or absent, a hash is 64 hex characters or it is not, a
diff has `---`/`+++`/`@@` or it does not — which is what makes `strong` honest
here rather than aspirational.

The cheapest rule is the one worth having: **`apply_mode`**. A patch that has to
be compiled cannot be bind-mounted. Accepted quietly, it would produce a
deployment running stock code, two identical arms, and a green report saying the
change was safe. Failing here, naming the mode, costs a second.

The second is **container paths must be placeholders**. A manifest naming
`/sgl-workspace/...` outright cannot be published at all — `handoff/locality.py`
refuses absolute paths outside a small allow-list and scans every file — so a
patch that arrives in bare-path form is a patch nobody can seal, and saying so
here is better than a seal failure three steps later.

A patch with no `runtime_marker` is a note, not a failure. It can still be proven
mounted, only never proven to have run, and a real KernelForge patch is under no
obligation to know about this package.

The logic is in `assets/lib/patchkit.py`, shared with `apply_patch`, which runs
the same check before it touches the image. Two implementations of "is this
manifest usable" would be one of them being wrong.
