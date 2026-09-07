# demo-runtime-disagree

Validates that a task whose body succeeds and whose output is well-formed can
still be sealed INVALID because the content fails a quality gate.

This is the pure-program version of `examples/demo2`'s
`review_x` / `review_y` → `reconcile` → `check_reviews_agree` path.

## Topology

```
main                        non-leaf: readme.md, no entry.sh
│
├── judge_x ──────────────► review_x  [structured_text]
│                             check_review_shape  PASS
├── judge_y ──────────────► review_y  [structured_text]
│                             check_review_shape  PASS
│         (two roots, froms: [])
│
├── reconcile ────────────► review    [structured_text]
│     froms: [judge_x, judge_y]; inputs: review_x, review_y
│     input validation:  check_review_shape        PASS on both
│     output validation: check_agree               FAIL on review
│
└── downstream              is_end, froms: [reconcile]
      inputs: review — never becomes VALID, so the body never runs
```

## Why `check_agree` fails

`assets/judge_x.task/judge.py` returns `accept` for `student_a`, `student_b`,
and `student_c`. `assets/judge_y.task/judge.py` returns `revise` for
`student_a` and `accept` for the other two. Both documents carry `student`,
`verdict`, and `comment` on every row, with verdicts drawn from
`[accept, revise]`, so `check_review_shape` passes on both.

`assets/reconcile.task/reconcile.py` joins the two documents by student, writes
matching verdicts into `agreed` and differing ones into `disagreed`, and exits
0. The merge itself is correct: `student_a` genuinely differs, so `disagreed`
has one entry and `totals.disagreed` is 1.

`assets/check_agree.validator/check.py` requires `disagreed` to be a list,
requires `totals.disagreed` to equal `len(disagreed)`, and then returns
`len(disagreed) == 0`. The list has one entry, so the verdict is FAIL and
`review` is sealed INVALID.

The distinction this package isolates: the body exited 0, the output satisfies
its schema, and the input validators passed. The rejection comes from the
semantic content of a structurally valid artefact, not from an execution error
or a malformed document.

## Expected terminal states

| Object | Terminal state | Verdict |
|---|---|---|
| `judge_x` | SUCCEEDED | `check_review_shape` PASS on `review_x` |
| `judge_y` | SUCCEEDED | `check_review_shape` PASS on `review_y` |
| `reconcile` | OUTPUT_VALIDATING | `check_agree` FAIL on `review` |
| `downstream` | WAITING_HANDOFF | input `review` is INVALID |
| `review_x` | VALID | — |
| `review_y` | VALID | — |
| `review` | INVALID | — |

The run ends quiescent with `downstream` in `WAITING_HANDOFF`. The two
expectations registered in `agent_sys/cli/expectations.py` under
`demo-runtime-disagree` are `agree_verdict_fails` and `downstream_waits`;
observing both is exit 0.

## Run

```bash
AGENT_SYS_NO_PERMISSIONS=1 agent-sys run --package agent_sys/examples/demo-runtime-disagree
```
