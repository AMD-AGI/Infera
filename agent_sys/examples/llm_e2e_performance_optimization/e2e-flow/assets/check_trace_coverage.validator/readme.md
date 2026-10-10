# `check_trace_coverage`

Completeness, `strong`, **program**. Six rules over a capture.

## The question is not "did eight files arrive"

A profiler window that opened on an idle scheduler produces eight perfectly
well-formed trace files holding nothing. On disk that is indistinguishable from
a good capture: same count, same names, plausible sizes. The difference is
inside, and the only way to see it is to decompress and parse.

## The rules

0. **This validator opens a trace itself** and counts what is in it. Mission
   M2.2.2 asks that the profile result be loaded by the corresponding analysis
   tool, and rules 1–3 below are otherwise read from a manifest the *producer*
   built — a manifest is a claim until something re-derives one of its rows.
1. **One readable trace per tensor-parallel rank.** Seven out of eight is not a
   partial success: an all-reduce that ran on eight ranks would be attributed
   against seven, and every share in the ranking downstream would then be wrong
   by an amount nobody can see.
2. **Every rank holds GPU kernels**, above a floor a warm-up artefact cannot
   reach. This is the idle-window rule.
3. **Every rank's span is plausible** for the window that was asked for. Much
   longer means the stop did not take effect; much shorter means it never really
   started. Both bounds are generous, because the profiler starts and stops
   asynchronously and the point is to catch a window that missed the load
   entirely.
4. **The files on disk match the manifest**, by name and by size. Without this
   the other rules are a description of something else.
5. **The measurement window carries no Python stacks, and the stack window
   carries them.** Both halves of one decision, and it is a rule because both
   fail silently. A measurement window taken with `with_stack` on is 13x the
   bytes it should be, measured, and is otherwise a perfectly good trace — the
   only symptom is a handoff nobody wants to move. A stack window taken with
   `with_stack` off holds no `python_function` events at all, and the launcher
   resolution downstream then reports "resolved 0" exactly as if the frames were
   genuinely unfindable.

## How rule 0 is done, and why not with torch

There is no torch and no perfetto on the login node where a validator body runs,
and a validator that needed a GPU node to grade a handoff would be a validator
that cannot run when the allocation ends. So the trace is read with
`assets/lib/trace_stream.py`, a streaming reader of the Chrome Trace Format —
the format torch writes and perfetto reads — which parses `traceEvents` one
event at a time rather than materialising 3.4 million objects.

**One rank by default (`verify_ranks: 1`), and the default is a cost decision
rather than a principle.** A full pass over one 65 MB rank is ~14 s measured on
this cluster; eight ranks plus a stack window is over two minutes, which would
make a validation phase's cheapest-first ordering meaningless. One rank is
enough for what the rule is for: a manifest that was fabricated, or built from a
different capture, does not survive one row being re-derived. The largest rank
is chosen, because a fabricated count is least likely to have been guessed close
there. `verify_ranks: -1` checks every rank; `0` trusts the manifest entirely
and says so in the args rather than silently.

## `cost: minutes`

`validator` spec §5.3 orders a phase cheapest-first and the tag is what does the
ordering. Rule 0 is seconds and the manifest read is instant, but the manifest
costs a parse of every event in every rank to *build*. Tagging it `seconds`
would put it ahead of checks that are genuinely cheaper and would misdescribe
what producing its input costs.

## Deferred

Extending rule 2 to check completeness against the sglang source and the model
structure is `../../../todo.md` **T1**; mission M2.8.2 says 先不做. Today this
cannot tell a trace that captured every layer from one that captured the first
two and stopped, because it has no model of what "every layer" means.

## Proven both ways

The real sealed capture PASSES, with rule 0 re-deriving 209,609 GPU kernels and
an 11.763 s span from the file and finding the manifest agrees. Six hand-built
failures are refused naming the fault, including a manifest whose kernel count
is a number about a different capture — which only rule 0 can see.
