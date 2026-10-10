# `ok.algorithms_solve_grade.14` — solve, review, reconcile, score and grade

Fourteen tasks: three students solve the same problem set, two reviewers
disagree, a reconciler settles it, and a grader scores the result against real
compiled C++.

**This directory is data.** YAML specs and the programs they name; nothing in
`agent_sys` imports any of it, and there is no `__init__.py`. Like
`../ok.filetree_grounded_report.4/`, it may use nothing a task package outside
this repository could not use.

```bash
pip install -e agent_sys
agent-sys run --package agent_sys/examples/ok.algorithms_solve_grade.14

# bring-up, or any run where you would rather not pay for a full problem set
agent-sys run --package agent_sys/examples/ok.algorithms_solve_grade.14 \
              --var n_problems=2
```

## What it proves that `../ok.filetree_grounded_report.4/` does not

That package is a chain of three: it proves the format loads and a run
completes. Four things it cannot show are this package's whole reason to exist.

| | |
|---|---|
| **Fan-out** | `solve_a/b/c` all consume `problems` and become eligible in the same scheduling round |
| **Fan-in** | `grade` joins three producers, which needs three *distinct* kinds |
| **Depth-2 nesting** | `grade` is a subgraph inside a subgraph: root → `grade` → six leaves |
| **Ten validators** | four of them compile and run real C++ |

---

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

## The one engine fact this package is shaped around

**Every solution is compiled into a single binary together with a harness**, so
the solutions share one process and one stdin. A solution that reads stdin
before the harness hands it its own input starves every solution after it —
measured over seven programs: `fgets`, `getchar` and `getline` all do it, and all
three are fine with a synced one. Reading stdin at all before your turn is the
variable; which function you use is not.

That is why the harness exists, why the scoring is per-binary, and why
`check_one_binary` is a validator rather than a convention.

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

**Nothing binds a filename.** There is no `body:` key anywhere: a folder named
`${name}.${type}` under `assets/` scopes the lookup, and `readme.md` / `entry.sh`
are found by their own names. Binding one by hand is legal and warns at compile
time, so a layout that needs bindings is a layout that failed.

An AI task is a folder with `readme.md` and nothing beside it; a program task
adds `entry.sh`. **That one file is the whole difference.**

### Two files that look like duplication, and one that looks like a shortcut

**`assets/lib/store.py` is a verbatim copy** of the other packages'. A task
package may not import from the repository — it has to work as an
out-of-repository package would — so the duplication is the rule being obeyed.
[`../../docs/TODO.md`](../../docs/TODO.md) item 4 is what removes the need.

**`assets/lib/cpp.py` calls `g++` directly.** A build system would need
declaring, installing and confining, for one translation unit per solution.

## The scale knobs

Every count reaches its body through its agent's `env` block, so a bring-up run
shrinks without editing a readme. A readme says *"write `$DEMO2_N_PROBLEMS`
problems"*, never a literal — otherwise the knob and the instruction disagree
and the instruction wins.

| variable | default | what it sizes |
|---|---|---|
| `n_directions` | 5 | topics `directions` picks |
| `n_problems` | 12 | problems in the set |
| `n_extra` | 10 | cases `extra_tests` adds |

## What it needs

Credentials for the Claude backend, `g++` on `PATH`, and a working sandbox. No
GPU, no cluster, and **no network during the run**: the two catalogues under
`assets/catalog/` ship with the package precisely so the closed lists the
validators match against do not depend on reaching an external site.
`leetcode_index.json` is a hand-curated subset written from knowledge, not a
mirror, and its own `_note` field says so.
