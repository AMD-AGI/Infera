# check_aiperf_report

Completeness, `strong`. Four rules over a replay report, one boolean per handoff.

## The failure it exists to catch

Not a crash. **AIPerf exits 0 when its schedule window turns out to contain
nothing**, and then every export is present, every file is well-formed, and the
whole thing describes zero requests. A shape check passes that report; a request
count does not. Rule 2 is the reason this validator is worth having.

## The rules

1. **Shape.** Every export AIPerf produces, plus the summary, the load
   configuration and the two command lines that identify the round, all non-empty.
2. **Requests happened**, and the metrics the next stage reads are present by
   name. Naming them here means an AIPerf release that renames a row fails as
   "the metric is missing" rather than as a `KeyError` three steps later.
3. **Error rate under the bar**, counted from the per-request records rather than
   from a summary field. AIPerf's error accounting has moved between releases,
   and a missing key would read as zero errors — the one wrong answer that looks
   like success. The bar is 5% rather than 0: a saturated deployment legitimately
   times out the occasional request, and a bar of zero would make this a flake
   detector.
4. **The replay describes a window**, and names the trace it replayed.

## It does not judge the numbers

The same rules run over both rounds. The profiled round is expected to be several
times slower, because its deployment has CUDA graphs off on purpose. A validator
that failed a report for being slow would be enforcing a policy nobody wrote, in
the one place where slowness is the intent.

What each round's numbers *mean* is in its own handoff's README and `watchout`,
which is where a judgement belongs when it cannot be mechanised.
