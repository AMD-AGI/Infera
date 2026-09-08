# check_bench_report

Completeness, strong. Every replay round produced a complete AIPerf export,
actually sent requests, and stayed under the error bar.

**`min_requests` is the rule that earns it.** AIPerf exits 0 after synthesising
prompts even when the schedule window turns out empty, and every export is then
present, well formed, and describes no requests at all — a successful-looking
round that measured nothing.

**`max_error_rate` is not zero**, and should not be. A saturated deployment
legitimately times out the occasional request under a fixed schedule; a bar of
zero turns this into a flake detector rather than a completeness check.

File sizes are read at check time with `stat`, not trusted from a listing taken
when the file was written. Measured on `profiling-demo`: immediately after a
write, `ls -l` and `du -sb` disagreed by three orders of magnitude on the same
files, and a later `stat` agreed with neither.

The last rule is about the pair rather than the round: both arms must have run
the same sequence of steps. "Round 1 was cold for this trace" is only true of an
arm if the same things happened before it, so two arms that disagree on the
sequence are two different experiments and the comparison downstream is not one.

---

## What changed on the way into `e2e-flow`

**It grades each round against m2's schema.** `args.schema` names `bench_result`,
which is m2's file in `assets/schemas/` and not a second copy: m5's replay and
m2's bench are the same AIPerf export, so one document grades both, and a round
whose `profile_export_aiperf.json` fails here is one m2's own
`check_bench_result` would have refused. That is mission G2 and CONTRACT.md 4.1
in one line of `args`.

`profile_export_aiperf.json` therefore joins the required exports. It is the only
one of the five that is lossless and typed — the CSV blanks the percentile
columns AIPerf did not compute, the console text is the CSV with box-drawing
characters, and the JSONL is the per-request detail the summary came from.

Its input is the merged `{stock,patched}.measurement`; the replay rounds sit at
`items/result/r<N>/` exactly as before, beside the correctness results rather
than in a handoff of their own.
