# `check_bench_result`

Completeness, `strong`, **program**. Six rules over a bench.

`kind: program` and not `ai`, per mission G4.1: every rule below is a count, a
schema validation or a comparison between two files. Nothing here is judged.

## The failure this exists to catch is not a crash

AIPerf exits 0 when its schedule window turns out to contain nothing. Every
export is then present, every file is well-formed, and the whole thing describes
zero requests. A shape check passes that; a request count does not.

## The rules

1. **Shape.** The five exports and the three environment records are present and
   non-empty. AIPerf writes four renderings of one run plus the per-request
   detail; all five are required, because the JSON is the record and the others
   are what a person opens when the record says something surprising.
2. **Schema.** `items/result/profile_export_aiperf.json` validates against
   `assets/schemas/bench_result.schema.json`, resolved **by name** through
   `assets/lib/schema.py` — the same file the producer validated against
   (mission G2). A name and not a path, so producer and validator cannot end up
   one directory apart and never notice.
3. **The load really ran.** The request count, read from the **export** rather
   than from the summary, clears `min_requests`. Also refused: a run marked
   `was_cancelled`, whose throughput describes however much of the schedule got
   sent; and a load taken with streaming off, which carries no TTFT and no
   inter-token latency — the two numbers stage 5 has to reproduce.
4. **The summary is a rendering, not a second opinion.** `summary.json` is
   derived from the CSV and the CSV from the same run as the JSON. When they
   disagree about how many requests there were, one of the four exports is from
   a different run, and there is no way to tell which from inside.
5. **Errors, counted per request**, from `profile_export.jsonl.gz` — not from a
   summary field. AIPerf's own error accounting has moved between releases, and
   a missing key would read as zero errors, which is the one wrong answer that
   looks like success.
6. **The replay's own configuration is recorded**, and it names a window and a
   trace.

## It does not judge the numbers, and that is deliberate

The same rules run over both modes. `profiling_mode_on` is expected to be
several times slower because its deployment has CUDA graphs off — measured on
the sealed pair, 15.65 ms mean inter-token latency against 124.98 ms, eight
times apart and both correct. A validator that failed a report for being slow
would be enforcing a policy nobody wrote, in the one place where slowness is the
intent. Comparing the two arms is stage 5's job and has its own validator.

## `min_requests` is 50 and not 100, and the gap cost a run

Measured on this trace: 346 requests in the default 120 s window, 166 in 60 s,
~80 in 30 s. A bar of 100 failed a perfectly working shortened replay,
invalidated both of that task's handoffs and stopped the graph. The bar must sit
below any legitimately shortened window, or it stops being "did anything get
sent" and becomes a second, undeclared constraint on window length.

## Proven both ways

Both sealed benches PASS. Seven hand-built failures are refused naming the
fault: an empty window, a summary from a different run, streaming off, a
cancelled run, a replay that does not say what it replayed, a lost metric, and a
missing record.
