# `check_profiling_evidence`

Completeness, `strong`, **program**. Five rules over the merge.

**Every rule here is a cross-part rule, and none of them can be checked anywhere
else in the flow.** Each of stage 2's two lines has its own validators, and they
grade that line's artefacts and cannot see the other's. The merge is the one
place the two lines meet, so it is the one place that can notice they were not
run against the same deployment.

## The rules

1. **Every part arrived**, as a non-empty directory under `items/result/`, and
   `items/env/parts.json` accounts for exactly those parts and no others. A
   manifest claiming a part the kind does not declare sends a consumer looking
   for a directory nobody agreed to.
2. **The parts describe one deployment**, in two halves:
   - *across the two lines*: same node, same image digest. **Not the same
     container** — the two lines are two bring-ups by design, because CUDA graph
     on and off cannot both be true of one running engine, and CONTRACT §5.2
     forbids either line reusing a container name it did not create. This rule
     was written the other way round first and correctly refused a correct
     merge, which is the right way round to find that out.
   - *within one line*: same container and same endpoint. `trace` and
     `kernel_table` came from the same bring-up as `bench_profiling_mode_on`.
     **This is the rule with teeth** — it catches a trace or a ranking folded in
     from a different profiled run, which is the one substitution that would
     leave every other number in this handoff looking right.
3. **The two benches replayed the same load** — same trace, same window, same
   concurrency ceiling, same served name. Without this the pair is two
   measurements of two different things, `profiling_mode_off` is not a control
   for `profiling_mode_on`, and the number stage 5's stock arm must reproduce
   (M5.1.3.1) describes a load nobody will run again.
4. **The ranking was derived from this trace.** The kernel table records the
   capture's own kernel total and the trace manifest records it independently;
   on the reference run both say 419,218. This flow has already been bitten
   once: a stage-3 run was fed a synthetic seed table and every validator
   downstream passed.
5. **The merged environment record agrees with its parts.** Every consumer reads
   `items/env/environment.yaml` and nothing else — that is the point of the
   merge — so a record that contradicts the evidence beneath it is worse than no
   record. Its `runtime` describes one of the two bring-ups and cannot describe
   both; what it may not name is a third container, which would be a record of a
   deployment this handoff carries no evidence of.

`check_environment` grades the merged record's *shape* against the schema. This
grades its *agreement* with what went into it. Neither substitutes for the other.

## Proven both ways

The real merge over the four sealed artefacts PASSES. Eight hand-built failures
are refused naming the fault: a trace folded in from a different profiled run,
two lines on different nodes, two benches replaying different loads, a ranking
over a capture the handoff does not carry, a missing part, an unaccounted part,
a merged record describing a deployment nothing here evidences, and no
provenance at all.
