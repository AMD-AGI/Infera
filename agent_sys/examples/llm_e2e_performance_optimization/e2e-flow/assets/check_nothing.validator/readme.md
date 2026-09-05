# check_nothing

Admits every declared handoff without opening it. A program validator; there
are no STEPS to follow because there is nothing to check.

**Not named in any hand-written `validators:` list.** It is injected into a
*generated copy* of this package by `assets/lib/make_debug_package.py`, which
`--package` selects per run, so the strict tree that other chains share is not
disturbed. Finding `check_nothing` in `steps/*.yaml` on `main` is a bug.

## Why it exists

`agent-sys` refuses a kind with no validator —

    handoff kind 'stock.measurement' names no validator.
    A kind with no validator cannot be admitted

— so "run with validation off" cannot be written as `validators: []`. This is
how it is written instead.

The purpose is to separate two failures that a strict run reports as one: a
chain that cannot *walk* (an edge that does not resolve, a body that does not
start, a file a downstream stage needs and never receives) from a chain that
walks and produces something a validator then argues with. The first has to be
fixed before the second can even be measured, and with validation on, the first
stage to disagree hides every stage behind it.

## What a green run against it establishes

**Reachability, and nothing else.** That every body ran, sealed what its kind
declares, and that the next stage could be handed it. Five stages chained have
never yet shown that.

**Not correctness.** Every verdict is `true` by construction — this is weaker
evidence than the trivially-passing validators this package has already been
caught by, because those at least read the file. The honest sentence for a
green run here is *"the chain walked all five stages with validation
disabled"*, never *"the validators passed"*.

`strength: weak`, because the schema says strength *"qualifies a PASS, never a
failure"*, and this PASS is unearned by design.

## Converging back

`make_debug_package.py --keep a,b,c` re-enables named validators while leaving
`check_nothing` on every kind, so a partially-restored tree still announces
itself. Re-enable the ones known to pass first and narrow onto the ones that
do not; a validator too expensive to settle gets written up and handed over
rather than argued with while the chain is still not walking.
