# demo2 — the second task package

This directory is **data**. It holds YAML specs and the programs they name, and
nothing in `agent_sys` imports any of it — there is no `__init__.py` here and
there never will be. Like `../demo/`, it may use nothing a task package outside
this repository could not use: no privileged import, no private loader path, no
schema of its own.

```bash
pip install -e agent_sys
agent-sys run --package agent_sys/examples/demo2
```

Bring-up, or any run where you would rather not pay for a full problem set:

```bash
agent-sys run --package agent_sys/examples/demo2 --var n_problems=2
```

---

## What it is meant to prove that `../demo/` does not

`../demo/` is a chain of three and proves the format loads and a run completes.
Four things it cannot show are the whole reason this package exists:

| | |
|---|---|
| **Fan-out** | `solve_a`, `solve_b`, `solve_c` all consume `problems`, all become eligible in the same scheduler pass, and all run at once — `runner.start` spawns a thread per dispatch |
| **Fan-in** | `grade` joins three producers. It needs three *distinct* kinds, and why is below |
| **Depth-2 nesting** | `grade` is a subgraph inside a subgraph: root → `grade` → six leaves. `../demo/` is one level |
| **Ten validators** | `../demo/` has two. Four of these compile and run real C++, so *it does not build* and *it never terminates* are recorded verdicts rather than a hang |

## The graph

```
main                              non-leaf: readme, no entry.sh, NO agent
│                                 inputs [] · outputs [optimised]
│
├── directions                    ai: teacher
│   │                             picks CLRS topics from assets/catalog/
│   │                             out: directions           [structured_text]
│   └── check_directions               every topic is in the closed list
│
├── problems  ← directions        ai: setter
│   │                             out: problems             [structured_text]
│   ├── check_problems                 every slug is in leetcode_index.json
│   └── check_solvable                 the set is actually answerable
│
├── solve_a   ← problems  ┐       ai: student_a  out: solutions_a     [code]
├── solve_b   ← problems  ├ FAN   ai: student_b  out: solutions_b     [code]
├── solve_c   ← problems  ┘ OUT   ai: student_c  out: solutions_c     [code]
│      each of the three: check_compiles · check_analysis
│
├── grade     ← problems, solve_a, solve_b, solve_c        <- FAN-IN
│   │         non-leaf: readme, no entry.sh, NO agent
│   │         in [problems, solutions_a/_b/_c] · out [scores]
│   │
│   ├── review_x     froms []     ai: reviewer_x  out: review_x
│   ├── review_y     froms []     ai: reviewer_y  out: review_y
│   │        both: check_review_shape
│   ├── reconcile ← review_x, review_y
│   │                 program     out: review
│   │                             check_reviews_agree
│   ├── harness   ← reconcile     program  out: harness         [code]
│   │                             check_one_binary
│   ├── extra_tests ← harness     ai: examiner  out: extra_tests
│   │                             check_extra_tests
│   └── score     ← harness, extra_tests        is_end
│                      program    out: scores
│                                 check_scores
│
└── optimise  ← problems, solve_a, solve_b, solve_c, grade    is_end
                                  ai: optimiser
                                  out: optimised              [code]
                                  check_faster
```

`main`'s subgraph is 7 entries; `grade`'s is 6. Twelve handoff kinds, twelve
leaves, two non-leaves, and ten validators.

## What each stage does

| stage | body | what it produces |
|---|---|---|
| `directions` | ai | Which CLRS areas this round covers, drawn from `assets/catalog/clrs_topics.json`. `$DEMO2_N_DIRECTIONS` of them |
| `problems` | ai | A concrete problem set, every entry a real slug from `assets/catalog/leetcode_index.json`. `$DEMO2_N_PROBLEMS` of them |
| `solve_a/b/c` | ai | Three students' C++ answers, one directory per problem, written independently |
| `review_x/y` | ai | Two reviewers' opinions on all three answer sets, formed without seeing each other's |
| `reconcile` | program | One `review` merging the two, recording where they disagreed instead of hiding it |
| `harness` | program | A C++ test harness for the problem set, compiled — the kind ships sources *and* one binary |
| `extra_tests` | ai | `$DEMO2_N_EXTRA` further cases the students' own tests would miss |
| `score` | program | Every answer run against the harness and the extra tests. Emits `scores`, the only kind that leaves `grade` |
| `optimise` | ai | A faster set of answers, given the problems, all three attempts, and the scores |

## The one engine fact this package is shaped around

**One producer per kind, per subgraph.** `producer_of[kind]` and
`available[kind]` are single-slot (`task_graph/models.py:360,747,757`), so if
the three students all wrote a kind called `solutions`, `grade` would see only
the last one and the other two would vanish — no error, no warning, just two
missing answer sets.

That is why the kinds are `solutions_a`, `solutions_b` and `solutions_c`. It is
**not a naming style**; it is the only way to write a three-way join in this
engine today, and a reviewer who reads it as taste will write the collision back
in the first time they add a fourth student.

The second consequence is `review_x` and `review_y` both declaring `froms: []`.
Everything they consume is an *input of `grade`* rather than something produced
inside it, so no edge is derived and there is nothing earlier to point at. A
kind nobody inside a subgraph produces resolves to the parent's own input
(`models.py:349-361`), and this package is where that stops being a footnote.

## Layout

```
main.yaml       the outermost graph. MANDATORY, and its name is fixed
shared.yaml     what more than one step uses — the `runner` program agent
steps/          one file per step, holding everything that step introduces
assets/         MANDATORY. every body found by filename convention
  main.task/            readme.md            (non-leaf: no entry.sh)
  grade.task/           readme.md            (non-leaf: no entry.sh)
  <name>.task/          readme.md [+ entry.sh + *.py for a program]
  <name>.validator/     readme.md, entry.sh, check.py
  lib/store.py          reading a published handoff without importing `handoff`
  lib/cpp.py            compile, run and time one C++ source
  catalog/clrs_topics.json     the closed list check_directions matches
  catalog/leetcode_index.json  the closed list check_problems matches
```

**Nothing in this package binds a filename.** There is no `body:` key anywhere.
A folder named `${name}.${type}` under `assets/` scopes the lookup, and inside
it `readme.md` and `entry.sh` are found by their own names. Binding one by hand
is legal and warns at compile time, so a layout that needs bindings is a layout
that failed; this one needs none.

An AI task is a folder with `readme.md` and nothing beside it; a program task
adds `entry.sh`. **That one file is the whole difference**, and comparing two
folders here says more about what the distinction means than a design document
would.

### `assets/lib/store.py` is a verbatim copy

It is byte-for-byte `../demo/assets/lib/store.py` apart from two docstring
references that named demo-1 specifically. That is deliberate:
`tests/cli/test_isolation_shown.py::test_the_store_layout_this_package_reads_is_handoffs`
pins **demo-1's** copy against `handoff`'s real constants, and a copy here that
drifted would be a second, unpinned reader of a layout `handoff` owns. Fix the
original and copy it across.

### `assets/lib/cpp.py` calls `g++` directly, and why

The repository rule is to prefer a mature tool over writing it yourself, and the
mature tool here **is** `g++`. A single translation unit with no dependencies,
no link order and no incremental rebuild is the case CMake, Meson and `make` all
exist above; wiring one in would add a generator step, a build directory and a
second failure mode in exchange for nothing this package needs. Python has no de
facto "compile one C++ file" library — `cppyy` and `pybind11` solve embedding,
not batch compilation — so `subprocess.run` around the compiler is the thin
wrapper, and it is the whole of it.

What the module is really for is the timeout. A submitted program that loops for
ever must fail as *timed out* and not hang the run; `run()` returns
`TIMEOUT_RETURNCODE` (124, `timeout(1)`'s convention) rather than raising, and
getting that right once beats getting it nearly right in four validators.

Measured, not assumed — `scratch/demo2-2026-08/probe_cpp.py`, kept as evidence:
compile and run a hello-world, a correct case *and a deliberately wrong one*
(the instrument is not stuck on PASS), a `while(true)` that returns 124 after
30.03 s, and a source that does not build returning `(False, diagnostics)`.

## The scale knobs

Every count in this package reaches its body through its agent's `env` block, so
a bring-up run can shrink without editing a readme:

| variable | default | what it sizes |
|---|---|---|
| `n_directions` | 5 | topics `directions` picks |
| `n_problems` | 12 | problems in the set |
| `n_extra` | 10 | cases `extra_tests` adds |

A readme says *"write `$DEMO2_N_PROBLEMS` problems"*, never a literal count —
otherwise the knob and the instruction disagree and the instruction wins.

## What it needs

Credentials for the Claude backend, `g++` on `PATH`, and a working sandbox. No
GPU, no cluster, and **no network during the run**: the two catalogues under
`assets/catalog/` are shipped with the package precisely so that the closed
lists the validators match against do not depend on reaching leetcode.com.

`leetcode_index.json` is a hand-curated subset of 67 real problems across 18
topics, written from knowledge and never fetched. It is not a mirror and does
not claim to be; its own `_note` field says so.
