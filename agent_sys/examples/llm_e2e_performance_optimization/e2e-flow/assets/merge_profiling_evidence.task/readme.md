# `merge_profiling_evidence`

Fold both lines into stage 2's single export (M2.9): *"bench_result、profiling
result、magpie standardized output result，三者整合进一个大的统一 handoff"*.

Stages 3 and 5 consume `profiling_evidence` rather than the four pieces, so this
is the boundary of stage 2 — and it is **the one place the two lines meet**,
which makes it the one place that can notice they were not run against the same
deployment. Each line's own validators grade that line's artefacts and cannot
see the other's.

## Inputs and outputs

| | |
|---|---|
| in | `profiling_mode_off.bench_result`, `profiling_mode_on.bench_result`, `profiling_mode_on.profile_result`, `profiling_mode_on.kernel_table` |
| out | `profiling_evidence` |
| graded by | `check_environment`, `check_profiling_evidence` |

**There is no mock path.** The four inputs are mocked upstream when the stage is
mocked, so this runs for real in every mode. That is deliberate: a hand-shaped
`profiling_evidence` would make the cross-part rules grade a stand-in, and those
rules are the only reason this handoff exists rather than four.

## STEPS

Executed in order by `entry.sh` → `merge.py`.

1. **Resolve the four inputs.** Each from its own `AGENT_SYS_INPUT_*` slot.
   *Accept:* all four set and naming directories. An unset slot means the
   closure does not declare that kind as an input, and the message says so —
   `handoffs:` must list every kind a task mentions, **inputs included**.

2. **Lift each part into `items/result/<part>/`, one rule for all four.** A
   part's content lands directly under its own directory with its environment
   beside it as `env/`:

   ```
   items/result/bench_profiling_mode_off/{summary.json,profile_export_*,env/}
   items/result/bench_profiling_mode_on/{…,env/}
   items/result/trace/{manifest.json,traces/,stacks/,stacks_manifest.json,env/}
   items/result/kernel_table/{text.json,table.csv,schema,env/}
   ```

   *Accept:* every named item present in its source. The `env/` of each part
   travels because the load configuration in it is what makes the two benches
   comparable and is not recoverable from the numbers.

3. **Record provenance in `items/env/parts.json`** — for each part: which
   handoff slot it came from, which of the two lines produced it, what was
   lifted, and the environment that part was taken in.
   *Accept:* four rows, each carrying a `line` and an `environment`. This is the
   piece that is not a copy, and it is what turns "these four describe one
   deployment" into a checkable claim.

4. **Inherit the environment record**, from the profiler-detached line's copy,
   via `env_render.py --inherit`.
   *Accept:* it prints the path it wrote. It validates before writing, so a
   merge that would produce a malformed record fails here instead of shipping
   something that reads like one. m1 is the sole producer of this document; a
   stage that re-derived it could differ from m1's with nothing to notice.

5. **Write `items/command`, `items/watchout` and `README.md`.**
   *Accept:* `items/command` is executable — `agent.gate` requires it of a
   `command` item — and names its inputs as shell variables rather than absolute
   paths, so the locality seal has nothing to reject.

6. **Validate the folded kernel table against the package's schema.**
   *Accept:* it validates. Cheap, and the merge is the last point at which the
   producer can still be blamed rather than m3 three tasks later.

## What `check_profiling_evidence` then checks, and why none of it fits anywhere else

- **Across the two lines: same node, same image digest.** Not the same
  *container* — the two lines are two bring-ups by design, because CUDA graph on
  and off cannot both be true of one running engine, and CONTRACT §5.2 forbids
  either line reusing a name it did not create. This rule was written the other
  way round first and correctly refused a correct merge.
- **Within one line: same container and same endpoint.** `trace` and
  `kernel_table` came from the same bring-up as `bench_profiling_mode_on`. This
  is the rule with teeth — it catches a trace or a ranking folded in from a
  different profiled run, which is the one substitution that would leave every
  other number here looking right.
- **The two benches replayed the same load** — same trace, same window, same
  concurrency ceiling. Without it the pair is two measurements of two different
  things and neither is a control for the other.
- **The ranking was derived from *this* trace**, by comparing the table's own
  record of its input against the trace manifest's independent count. On the
  reference run both say 419,218 GPU kernel events. This flow has already been
  bitten once by a ranking over a different capture: a stage-3 run was fed a
  34-row synthetic seed and every validator downstream passed.


## The content-root README your handoff must carry

**Only read this if you are an AI agent** — the program body writes these
already. You are one the moment somebody passes `--var m2_agent=e2e_profiler`,
which flips all three of this stage's leaves to `kind: ai` at once.

The file is `.../<handoff-id>/v1/content/README.md`, at the **content root** —
not a README inside a subdirectory, and not the one in a packup. The seal
requires specific section headings, **per content type**, and they are not the
same set for every output (`agent_sys/handoff/content.py:62-87`):

```
reproducible     ## Purpose · ## How to run · ## Result · ## Environment · ## Watch out
structured_text  ## Purpose · ## Schema
```

This leaf produces **one** output:

| output | content type | sections |
|---|---|---|
| `profiling_evidence` | `reproducible` | all five |

**A heading inside a blockquote, a list or a code fence is not a section** —
that is the seal's own sentence and the one you will see if you get it wrong.

**Do not take this brief's own `## Watch out` heading as the requirement being
met.** That heading belongs to this document; the seal reads the README you
write, not the brief you read.

**Why this is spelled out rather than left to judgement.** On 2026-09-05 two
stage-4 agents on two nodes both failed this seal within four minutes, and
neither failure resembled the other — one wrote no root README, one invented
its own headings. The refusal arrives **after** you have finished, and the
framework's retry cannot reach a finished agent
(`temp/bugs/2026-09-05-the-stall-detector-is-blind-to-a-task-that-holds-a-thread-and-does-nothing.md`),
so a wrong heading here is not a correctable mistake — it ends the run and the
GPU hold with it. One of those cost hold `112699`.
