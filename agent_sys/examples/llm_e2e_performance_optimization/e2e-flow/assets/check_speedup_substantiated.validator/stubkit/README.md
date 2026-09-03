# `stubkit` — drive `check_speedup_substantiated`'s re-measurement path, with no GPU and no torch

```sh
python3 assets/check_speedup_substantiated.validator/stubkit/run.py
```

Under two seconds. No GPU, no torch, no engine image, no node.

## What it proves

The validator's **expensive half** — everything past the apparatus check, which
until now had never executed. `_remeasure` copies the apparatus, runs the
workset's performance entrypoint twice, compares the seed against the workset's
recorded baseline, compares the candidate against the producer's claim, and
applies the noise floor. Seven of those steps are only reachable when a
runnable workset is on disk.

The workset here is a **stub**: `run_correctness.sh` and `run_performance.sh`
are scripts this kit controls, which emit reports in m3's own
`workset.schema.json#/$defs/performance_report` shape with numbers the case
chooses. That is the whole trick — a real workset measures whatever the
hardware does, and cannot be asked for a disagreement on demand.

## What it does NOT prove, and this matters more than what it does

- **No arithmetic on this host is the arithmetic on a GPU.** Every number here
  is one the stub was told to print. This kit cannot tell you that
  `weighted_mean_ms` is computed correctly by a real harness, that HIP graph
  replay timing is sound, or that a Triton kernel compiles.
- **It does not exercise m3's real entrypoints.** It mimics their *interface*
  — `--operator`, `--impl`, `--shape`, `--json`, and the report shape — so a
  change to that interface breaks this kit loudly, which is the point. It says
  nothing about whether their implementation is right.
- **It does not test the `_interpreter` probe, it satisfies it.** That function
  does not take an interpreter on trust — it runs `import torch` in each
  candidate and rejects one that cannot. Correct, and it is why the first run of
  this kit failed all seven cases identically on a torch-less login node,
  testing the probe rather than the path. So the kit puts a two-line `torch.py`
  on `PYTHONPATH`. The trap the probe guards (`bugs/002`, the PRODUCER row
  shadowing `AGENT_SYS_DEMO_PYTHON`) stays untested here and is exercised the
  first time this runs on a real node.

So: this kit is a harness test, not a measurement test. A green run means the
validator's control flow is right — that it aborts where it should abort, fails
where it should fail, and does not mistake a crashed measurement for a slow
one. It does not mean any speedup anywhere is real.

## The cases

| # | case | what it would look like without the check |
|---|---|---|
| 1 | the candidate is genuinely faster and the claim is honest | — (PASS) |
| 2 | the seed **disagrees** with the workset's recorded baseline | the premise silently did not hold; under the deleted rule this was a re-baseline, and the report stayed internally consistent while answering a different question |
| 3 | `run_performance.sh` **exits non-zero** on the candidate | a crashed measurement read as a zero-speedup measurement — the failure the leader named, and the worst one here, because it looks like an honest null result |
| 4 | the report is **missing `weighted_mean_ms`** for a case | a case silently dropped from the mean, so a candidate slower on one shape is averaged over two |
| 5 | the producer **over-claims** — 2.8× claimed, 1.1× re-measured | the number nobody downstream re-derives |
| 6 | the candidate clears nothing — 1.01× against a 1.057 floor | noise reported as a win |
| 7 | `noise_floor` absent from the claim | the consumer picks its own significance threshold (m3's objection, made executable) |

Cases 2 and 3 are the two the leader asked for specifically, and they are the
two a real workset cannot be asked to produce on demand.

## It found a real bug on its first proper run, which is case 4

`_medians` skips a shape whose figure is absent or non-numeric; `shared` was
then an intersection of the two sides; and the mean was taken over whatever came
back. So a candidate that **failed to measure on one shape was scored on the two
it managed** — which flatters exactly the kernel that is fast on the easy shapes
and broken on the hard one, and reports it as a clean pass.

The validator now requires every case the workset baselines to come back from
both sides, and says so when one does not. An entrypoint that exits 0 and
reports nothing for a case has failed to measure it; that is not a smaller
sample and the two must not fold together.

**And a second one, in this kit rather than in the validator.** Cases 4 and 7
originally passed on the missing-interpreter failure — they expected a refusal
and got one, for a reason that had nothing to do with what they claim to test.
Every case that expects a refusal now has to name the reason to look for, and a
case that does not is itself a failure.

## Why it lives here

Beside the validator it tests, inside a directory the ownership manifest
already assigns to m4 (CONTRACT §8a). It is `assets/`-resident, so it is copied
into every zone — a few kilobytes of shell and JSON — which means it can be run
**on the node**, where the `torch.py` shim is not needed and the real
`_interpreter` probe runs for real.

It takes no arguments and it does not point at a real workset. Substituting one
would mean giving up the thing that makes it useful: a real workset measures
whatever the hardware does, and cases 2, 3 and 4 need a measurement that
disagrees, crashes, or comes back incomplete **on demand**. Driving the
validator against a real workset is a different exercise, it needs a GPU, and it
belongs in the mock run rather than here.
