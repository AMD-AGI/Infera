# check_review_shape — completeness, strong

Both reviewers write the same document: one row per (student, problem) pair,
each row carrying two judgements, a verdict and a comment. This checks that the
document is that.

It is bound to **two kinds**, `review_x` and `review_y`, over one
implementation. That is `validator` criterion 12's shape, and it is also why
this validator exists at all: `handoff` spec §5.3 refuses a kind that names no
validator, so neither review kind would load without one.

## What it checks

Over every row, and the check is total:

| | |
|---|---|
| keys | every key `args.json`'s `required_row_keys` names is present |
| `student` | one of `args.json`'s `students` — `a`, `b`, `c` |
| `verdict` | one of `args.json`'s `verdicts` — `accept`, `revise` |
| `implements_claimed_algorithm`, `complexity_credible` | **JSON booleans**, not the strings `"true"` / `"false"` |
| `comment` | a non-empty string |
| pairs | no (student, problem) pair appears twice |

The boolean test is `isinstance(x, bool)` rather than truthiness, and the
direction matters: the string `"false"` is truthy, so a lenient test would read
a reviewer's rejection as an acceptance.

Then, when it can: **exactly one row per pair the students actually
submitted** — no missing pair, no invented one.

## The limit, stated rather than buried

The coverage check needs the submissions, and this validator's input is the
review. It reaches them through `store.declared_dir("solutions_a")` and its two
siblings — the **declared** route, `AGENT_SYS_INPUT_SOLUTIONS_A`, exported for
the producing task's own input slots.

That variable is present in the phase whose configuration is the *producer's*,
and absent in the one that takes the global row. `validator/phase.py:297-358` is
the account: `_configuration_sources` supplies `bound`, `producer` and
`global_`, and **no `consumer`** — so an input phase falls through to the global
row, which is `cli/main.py:601`'s four variables and no `AGENT_SYS_INPUT_*` at
all.

Concretely: in `review_x`'s and `review_y`'s **output** phases the coverage
check runs. In `reconcile`'s **input** phase it does not, and the body prints
`coverage NOT checked`. It does **not** quietly pass the coverage question — the
row-level checks are still total, and the transcript says which of the two runs
this was.

In practice the second case is usually never reached: `PhaseRunner` finds a
prior verdict for the same handoff version and reuses it
(`validator/phase.py:660-667`). That is a scheduling property and not something
this body relies on, which is why the fallback is written and says what it is.

`strong` is honest under that limit because `strength` qualifies a **pass**
(`validator` spec §5.4), and the pass this validator issues is *"every row is
well formed, and — where it could be determined — the rows cover exactly the
submitted pairs"*. The transcript distinguishes the two.

## What it does not check

Whether either reviewer is **right**. This is a shape check. Whether the two
reviewers agree is `check_reviews_agree`; whether their judgement is correct is
not checked by anything in this package, and no arrangement of two language
models could establish it.

## How it runs

`entry.sh` is the body. `validator.ScriptBodyRunner` runs it in a freshly
allocated zone with `args.json`, `inputs.json` and `materials.json` beside it,
and it writes `verdict.json` — one boolean per handoff id in `inputs.json`.
