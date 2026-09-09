# `packup`

Assemble the whole flow into one reproduction kit — the single artefact that
leaves this graph.

A program. It reads eight handoffs spanning all five stages and writes one
directory in the `experiment-result-packup` skill's `deliverable_layout.md`
shape: `README.md`, `REPRODUCE.md`, `environment.md`, `notes.md`, and
`results/`, `logs/`, `scripts/`, `handoffs/`.

## Not the integration stage's kit — the flow's

`integration-demo`'s packup carried the two arms and the patch, which was right
for a package whose input was a hand-written mock. This one is downstream of all
four earlier stages, and a kit that carries only the integration numbers cannot
tell a reproducer where the kernel under test came from, which service m1 brought
up, or what m2 measured that made m3 pick this operator.

So `results/` carries one representative artefact per upstream stage —
m1's kit README, m2's bench summary and kernel table, m3's workset, m4's
verification and forge result — beside the report, the two arms' correctness
results and every replay round's summary. One artefact per stage rather than the
whole handoff: the handoffs are the record, and a packup is the path through
them.

A stage whose artefact is absent leaves a `<stage>.MISSING` file. "Absent" and
"never looked for" are the same silence otherwise.

## `content_type: code`, and it matters

Laying a packup into a `reproducible` kind renames `results/` to `items/result`
and leaves `REPRODUCE.md` with no item to be — which destroys exactly what
`check_packup_shape` exists to check. `code`'s `codes` item is unconstrained
inside, so the layout survives.

## REPRODUCE.md is executed, not read

`check_packup_shape` counts its **command** lines rather than its prose, because
it is the one file in a packup that somebody runs.

Two traps that have both bitten here and are worth knowing before editing it:

- **The seal refuses absolute paths, and it refuses them in prose too.** The rule
  is "two or more slash-separated segments whose preceding character is not in
  the exclusion set", and `>` is not in that set — so `<this package>/assets/x`
  is read as the absolute path `/assets/x`. Write `$PKG/assets/x`; `G` is in the
  exclusion set. Re-scan any hand-written document with
  `assets/lib/redact.py <dir> NOOP=/nonexistent`, which runs the seal's own rule.
- **A raw engine log is thousands of container-internal paths.** Logs arrive here
  already gzipped and stay that way; the seal skips a file it cannot decode as
  UTF-8.

## Mock

`mock_m5.sh packup` (MOCK-MAP adaptation F). There is no sealed `e2e_packup` —
the 2026-09-02 graph stopped at a correctly refused report before the packup step
was dispatched — so the source is the 47-file kit produced out of band by
`integration`'s own unmodified `packup.py` over that run's nine sealed handoffs.
**It is not sealed**, its provenance is `PRODUCED-BY-DEPLOY.md` in the directory
it comes from, and the mock writes that fact into the handoff's `watchout`.
