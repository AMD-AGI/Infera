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
