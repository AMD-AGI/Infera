# check_packup_shape

Completeness, `strong`. Adapted from `single_real_task`'s validator of the same
name, against the same layout reference.

## Presence is not enough, and the reason is measurable

The layout reference ships templates for `README.md` and `REPRODUCE.md`. A
producer told "your packup must contain these files" emits exactly those files
with the template bodies still in them, and a presence check passes.

This is not hypothetical. Hugging Face's live model-card validator returns HTTP
200 for a card whose entire prose is `[More Information Needed]` — a string its
own template emits 39 times and which appears in 636,321 repositories.

So each mandated document is measured in **content lines**: non-blank, not a
heading, not a code-fence marker. Headings are excluded precisely because a
skeleton is mostly headings.

## REPRODUCE.md is measured differently

It is the one file a reproducer actually executes, so its substance is counted in
**command lines** rather than in prose. A reproduction kit that describes the
procedure in paragraphs is a description, not a kit.

## results/ has a floor

Without evidence the packup is a narrative. Four non-empty files is a low bar and
it is there to catch the case where every upstream copy silently failed and left
an empty directory behind.

## What it does not check

That the commands work. `single_real_task` pairs its shape check with an
`external_dynamic` validator that hands a fresh Claude Code session the packup and
nothing else, and sees whether it reproduces. That validator is the natural next
addition here and is not yet written; the design records it as the one place in
this package where an agent earns its place.

---

## What changed on the way into `e2e-flow`

**It grades the whole flow's export, not one stage's.** `packup.py` now reads
eight handoffs spanning all five stages, and `results/` carries one representative
artefact per upstream stage beside the two arms' numbers — a kit that carries only
the integration numbers cannot explain where the kernel under test came from.

A stage whose artefact is absent leaves a `<stage>.MISSING` file rather than
nothing, because "absent" and "never looked for" are the same silence otherwise.

`require_files` is declared explicitly in the step yaml rather than inferred from
`min_content_lines`: a present-but-empty README satisfies a file list and tells a
reproducer nothing, and the two rules are separate so that each says which
failure it caught.
