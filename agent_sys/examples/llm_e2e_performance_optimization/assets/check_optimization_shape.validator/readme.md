# check_optimization_shape

**Claim:** the handoff is one packup carrying the forge result, the optimized
kernel, the producer's own re-measurement, and an honest summary — and it does
not claim a speedup it did not make.

`completeness` / `strong` / `cost: seconds` / `logic_source: external_static`.

## Why it runs first

`cost: seconds` against `check_speedup_substantiated`'s `cost: minutes`, and a
validation phase is ordered cheap-first (`validator/phase.py:675-686`). A
handoff with no `forge_result.json` fails in a second rather than after a
re-measurement that had nothing to measure.

## The rules

**Layout**

- exactly one `<name>.packup_<YYYYMMDD>/` under `items/codes/`, with a real
  eight-digit date;
- `README.md`, `REPRODUCE.md`, `environment.md`, `notes.md` all present;
- `scripts/` and `results/` present and each holding at least one file.

**Substance**, counted in *content lines* — non-blank, not a heading, not a
fence marker:

| file | floor |
|---|---|
| `README.md` | 5 |
| `REPRODUCE.md` | 8 |
| `environment.md` | 8 |
| `notes.md` | 3 |

plus at least **5 command lines** inside fenced blocks in `REPRODUCE.md` (a
comment line inside a fence is documentation, not a command), a `## Result`
heading in `README.md`, and at least one digit in `environment.md` — a kit that
pins no version is the one whose absence breaks reproductions.

**No placeholders** — `TODO`, `TBD`, `FIXME`, `XXX`, `to be filled in` — in any
of the four.

### Why there is no `<…>` template-slot rule

The sibling package's shape check has one, so this body had one too. Run against
a real, complete, honest packup it fired twice: on `<workset>` inside a
`REPRODUCE.md` command, and on `<project_root>` in a sentence describing another
tool's default path. **Both are documentation metavariables and both are correct
writing.**

A regex cannot tell a slot the author forgot to fill from a metavariable the
author meant — they are the same characters. So the only question is which error
to prefer. This validator is `strong` and its PASS carries no qualification, so
a false failure is the worse one: it teaches authors to write vaguer
documentation in order to pass, which is the opposite of the point. The rule was
removed after measuring it, not omitted by oversight.

**Evidence.** Every path in `args.required_evidence` exists and is non-empty:

| file | why it is required |
|---|---|
| `results/forge_result.json` | forge's own verdict. Must parse and carry `baseline_ms`, `best_ms`, `mean_case_speedup`, `improved` |
| `results/optimization_report.md` | forge's human-readable report |
| `results/optimized_kernel.py` | the artefact itself. A report without the kernel is not a deliverable |
| `results/verification.json` | **the producer's own re-measurement.** Must parse and carry `mock`, `rounds`, `iters`, `mean_case_speedup`, `correctness_passed` |

## The one rule that is not about shape

**Mock consistency.** If `verification.json` says `"mock": true`, then:

- the packup `README.md` must contain the word `MOCK`, and
- `mean_case_speedup` must not be `> 1.0`.

And if it says `"mock": false`:

- `correctness_passed` must be exactly `true`;
- `rounds` must be at least 5, because the baseline protocol is five rounds and
  a comparison across differently-sized samples is not a comparison.

This is here rather than in the expensive validator because it costs nothing and
because **a mock that is not visibly a mock is the most misleading artefact this
package can produce.** It has every file, it parses, it reads as a success, and
nothing in it ran. The rule is cheap insurance against a wiring test being
mistaken for a result three weeks later.

## What it cannot catch

- **It runs nothing and verifies no number.** A `forge_result.json` claiming
  `mean_case_speedup: 99.0` passes here. Substantiating a number is
  `check_speedup_substantiated`'s job, and the split is deliberate.
- **It does not read the optimized kernel.** A syntactically valid file that
  computes the wrong thing passes.
- **It cannot tell a truthful `notes.md` from a fluent one.** Three content
  lines of honest gotchas and three of padding score identically.
