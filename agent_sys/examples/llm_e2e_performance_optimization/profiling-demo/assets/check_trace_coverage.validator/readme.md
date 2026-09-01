# check_trace_coverage

Completeness, `strong`. Four rules over a capture.

## The question is not "did eight files arrive"

A profiler window that opened on an idle scheduler produces eight perfectly
well-formed trace files holding nothing. On disk that is indistinguishable from a
good capture: same count, same names, plausible sizes. The difference is inside,
and the only way to see it is to decompress and parse.

So the producer does that once, in `assets/load/manifest.py`, and records each
rank's GPU kernel count and time span. This validator reads those counts.

## The rules

1. **One readable trace per tensor-parallel rank.** Seven out of eight is not a
   partial success — an all-reduce that ran on eight ranks would be attributed
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
   the other three rules are a description of something else.

## Why `cost: minutes`

`validator` spec §5.3 orders a phase cheapest-first, and the tag is what does the
ordering. Reading the manifest is fast, but the manifest costs a parse of every
event in every rank to build — about a second per rank at this size. Tagging it
`seconds` would put it ahead of checks that are genuinely cheaper and would
misdescribe what producing its input costs.
