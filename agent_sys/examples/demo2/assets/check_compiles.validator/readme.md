# check_compiles — completeness, strong

Every `solution.cpp` in a `solutions_a` / `solutions_b` / `solutions_c` handoff
compiles with `g++ -O2 -std=c++17`, and reproduces every example the `problems`
artefact ships for the problem it is filed under, inside a 30-second per-case
wall-clock cap.

It is `strong` without qualification, and the reason is that nothing in it is
read rather than run. A source is compiled by the compiler the students were
told to target, and its output is produced by executing it against the inputs
the problem setter wrote. There is no rubric, no sampling and no model in the
loop: the verdict is what the machine did.

## What it does, in order

For each handoff id, over each directory under `items/codes/`:

1. `solution.cpp` must exist.
2. The problem id — the directory's name — must name a problem in the
   `problems` artefact, and that problem must ship at least one example.
3. `cpp.compile` must succeed.
4. Every example must run to exit 0 within the cap, and its stdout must equal
   the example's `output` after trailing whitespace is dropped per line and
   trailing blank lines are dropped.

The example's expected text is `examples[].output` in the `problems` artefact.
`cpp.check_cases` calls the same field `expected`, but that is the name of *its
own parameter* and not of anything in the artefact — the two are bridged at the
call site, not assumed to agree.

Any one of those failing fails **the handoff**, not just the problem — the
verdict is per handoff id, so the per-problem results are folded with `all`.
The per-problem detail goes to stdout, one line each, so a reviewer can see
which problem broke and why rather than only that something did.

An empty `items/codes/` is a **fail**. `all([])` is `True`, so folding an empty
set would make "submitted nothing" indistinguishable from "everything passed" —
the same shape as a `dict.get` folding `None` as falsy.

## How it reaches the problems

`store.declared_dir("problems")`, which resolves `AGENT_SYS_INPUT_PROBLEMS`.
A validation body runs with the *producing task's* resolved configuration, so
that variable names the `problems` artefact this student actually consumed —
not the newest `problems` anywhere in the store, which is what a scan would
answer and which would be wrong the moment there were two. It also needs no
store root, so it keeps working under confinement, where a scan does not.

## What it cannot tell you

- **Nothing about the problems nobody submitted.** It checks the directories
  that are present. A student who solves one of their four problems and files
  one directory passes this check on that one directory. Whether the share was
  covered is `check_scores`' question, downstream.
- **Nothing about efficiency.** A program that finishes the shipped examples
  inside the cap passes, however badly it would scale. That is deliberately
  `check_analysis`' territory on the claim side and `check_faster`'s on the
  measured side.
- **Nothing beyond the shipped examples.** The examples are the problem
  setter's, and they are few. `extra_tests` exists because of exactly this gap.

## The compiler is the one on this machine

`g++ 11.4` at the time of writing. A solution that depends on a newer libstdc++
fails here and would pass elsewhere, and that is a real limit of the check
rather than a property of the submission.
