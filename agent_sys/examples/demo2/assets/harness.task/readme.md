# harness — one executable, selected by test case

Three students submitted a solution to each problem. This step turns all of
them into **one** program: a single executable that takes a case id
`<student>/<problem_id>` and runs that student's solution to that problem,
reading the test input from stdin and writing the answer to stdout.

*"One unified executable that selects by test case"* is the requirement, and it
is why `check_one_binary` asserts the count is **exactly one** rather than at
least one. Thirty-six separate binaries would satisfy every other property that
validator tests.

## Why one binary rather than thirty-six

Because `score` then has one thing to invoke and one build to have gone wrong.
With one binary per submission, a scoring loop has to decide what a missing
binary means, and *"student b's solution to p7 did not compile"* becomes
indistinguishable from *"the build step skipped it"*. With one binary, either it
exists and answers every case id in its manifest, or the build failed loudly.

## How the sources are combined

`assets/lib/cpp.py`'s `compile(src, out)` builds **one translation unit** —
`g++ -O2 -std=c++17 -o out src` — so combining thirty-six solutions means
generating one `.cpp`, not linking thirty-six objects. Each submission is
transformed the same way:

1. every `#include` line is lifted to the top of the generated file and
   de-duplicated, because a system header cannot be included inside a
   namespace;
2. everything else the submission wrote is wrapped in
   `namespace sol_<student>_<problem_id> { ... }`, which makes each solution's
   globals, helpers and `main` distinct symbols with no renaming;
3. every macro the submission defined is `#undef`ed after its namespace closes,
   so `#define int long long` in one solution cannot reach the next one. This
   is the one hazard a single translation unit has that separate ones do not,
   and it is handled rather than hoped about.

A function named `main` inside a namespace is an ordinary function — the
restrictions on `main` apply to the one at global scope — so nothing has to be
renamed and no submission is edited beyond being moved.

The generated dispatcher is the only global `main`.

### The two hazards this does not remove

- A submission with an `#include` inside an `#ifdef` has that include hoisted
  out of its guard. Nothing in a self-contained competitive-programming
  solution needs one, and the alternative — a preprocessor-aware transform —
  is a second, worse copy of the compiler.
- Two submissions that both define the same macro *and* rely on it having
  different values would still collide inside one unit if either forgot to be
  a `#define`. The `#undef` pass covers everything `#define`d.

### When a submission will not build

The combined build is attempted first. If it fails, each submission is compiled
on its own — its namespace plus a trivial `main`, which is a whole program and
so is something `cpp.compile` can build — and the ones that fail are **dropped
by name**, recorded in the manifest's `excluded` list with the compiler's own
diagnostics, and the combined file is regenerated and built once more. A
solution that does not compile is a fact about that solution; it must not stop
the other thirty-five from being scored.

If the build fails even with nothing excluded, this step exits non-zero. A
harness that did not build is not a harness, and publishing an empty one would
turn a build failure into thirty-six wrong answers.

## The command line

```
harness --list                    every case id it answers to, one per line
harness <student>/<problem_id>    run that solution; test input on stdin
harness                           read the case id from the FIRST LINE of stdin,
                                  then hand the rest of stdin to the solution
```

**The third form exists because of `assets/lib/cpp.py`.** Its frozen
`run(binary, stdin_text, *, timeout)` executes `[str(binary)]` and passes no
argv at all, so an argv-only binary could not be driven through the helper that
`score` and `check_one_binary` are supposed to use. Accepting the case id on
either channel costs four lines in the generated dispatcher and keeps the
a convenience for a human at a prompt. The first line is consumed with `fgets` on
`stdin` rather than with `std::getline`, so that a solution reading with
`scanf` and a solution reading with `std::cin` both see the same remaining
input.

An unknown case id is exit code 2 and a message on stderr, which is
distinguishable from a solution that ran and got the answer wrong.

## What it reads

| variable | what is in it |
|---|---|
| `$AGENT_SYS_INPUT_PROBLEMS` | the problem set; its worked examples are copied into the manifest |
| `$AGENT_SYS_INPUT_SOLUTIONS_A` | student **a**'s submissions — `items/codes/<problem_id>/*.cpp` |
| `$AGENT_SYS_INPUT_SOLUTIONS_B` | student **b**'s, same shape |
| `$AGENT_SYS_INPUT_SOLUTIONS_C` | student **c**'s, same shape |
| `$AGENT_SYS_INPUT_REVIEW` | the reconciled review; its verdicts are copied into the manifest |

## What it writes

Into `$AGENT_SYS_OUTPUT_HARNESS`:

```
README.md                with `## Purpose`, `## Interface` and `## Boundary`
items/codes/harness.cpp  the one generated translation unit
items/codes/build.log    g++'s output, verbatim
items/codes/bin/harness  the one executable
items/manifest.json      every case id, plus what `score` needs to score them
```

`harness` is `content_type: code`, which **requires** an item named `codes` —
hence the directory, rather than files at the top of `items/`. `manifest.json`
is a key the content type never defined and is admitted by the kind's own
`items_schema`, the same way `examples/demo`'s `summary` admits `grounding`.

### Why the manifest carries the examples and the verdicts

`score`'s declared inputs are `harness` and `extra_tests` — it never sees
`problems` or `review`. But it needs the worked examples (they are part of what
a solution is scored on) and it needs each pair's review verdict (10% of the
weight). So this task, which *does* consume both, passes them through into its
own output.

That is the ruling `examples/demo/steps/describe.yaml` records in its `summary`
kind — *"the task author passes their own input through to their own output
when they define the task"* — applied a second time. The cost is the same and
is worth stating: the examples and the verdicts reach `score` **through the
harness builder**, so `check_scores` is exact about the arithmetic and inherits
this program's honesty about the copy. The copy is mechanical — no selection,
no summarising — which is what keeps it a copy.

## Layout

`entry.sh` is the command; `build.py` is the implementation; `assets/lib/cpp.py`
is the compiler wrapper it shares with three validators.
