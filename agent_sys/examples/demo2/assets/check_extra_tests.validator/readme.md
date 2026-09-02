# check_extra_tests — completeness, strong

The examiner was asked for `$DEMO2_N_EXTRA` **new** test cases per problem. This
checks that they are that many, that every field is filled in, and — the part
the task exists for — that none of them is a worked example the problems
artefact already ships.

## What it checks

| | |
|---|---|
| fields | every case has a non-empty `problem_id`, `input`, `expected` and `why`, and all four are strings |
| internal duplicates | no two cases for one problem share an input |
| count | at least `$DEMO2_N_EXTRA` cases per problem |
| shipped duplicates | no case's input matches a worked example of that problem |
| ids | no case is keyed on a problem id the problems artefact does not have |

Comparison of inputs is stripped whole and then stripped per line — the same
rule `assets/lib/cpp.py`'s `check_cases` uses on outputs. Re-indenting a shipped
example does not make it new, and a byte comparison would say it did.

## Where the count comes from

**`$DEMO2_N_EXTRA` first.** That is what the `examiner` agent's `env` block
resolved for this run, so `--var n_extra=2` shrinks the requirement in step with
the instruction the examiner was actually given. A literal `10` here would fail
every bring-up run for a reason that has nothing to do with the examiner.

The variable is present in the phase whose configuration is the producing
task's, and absent in one that takes the global row — `validator/phase.py:297-358`
supplies `bound`, `producer` and `global_` and no `consumer`, so an *input*
phase falls through to `cli/main.py:601`'s four variables. Concretely: it is
there in `extra_tests`' output phase, and not in `score`'s input phase.

`args.default_n_extra` is the fallback, and it is used **only** when the
environment is the examiner's and the variable is genuinely absent — detected by
whether `store.declared_dir("problems")` resolves, which it does on exactly that
row. Using it on the global row would fail a `--var n_extra=2` run against a
requirement of 10 that nobody asked for.

When neither is available, the absolute count is not enforced and the body
instead requires **the same number of cases for every problem**. That is a real
completeness property — an examiner who ran out of ideas on problem 9 fails it —
it needs nothing from the environment, and it does not manufacture a
requirement. The transcript says which rule applied.

In practice the second row is usually never reached: `PhaseRunner` finds a prior
verdict for the same handoff version and reuses it
(`validator/phase.py:660-667`). That is a scheduling property this body does not
rely on, which is why the fallback is written and says what it is.

## Reaching the shipped examples

Through `store.declared_dir("problems")` — `AGENT_SYS_INPUT_PROBLEMS`, exported
because the producing task, `extra_tests`, consumes `problems` itself. Same row,
same limit: when it does not resolve, the duplication check is reported as **not
checked** rather than passed. Absent evidence and satisfied evidence have the
same shape, and folding the first into the second is how a check quietly stops
checking.

## What it does not check

**Whether an `expected` output is correct.** The readme tells the examiner to
derive it by hand from the statement, and nothing here can verify that they did
— running a student's solution to find out would be circular, since those
solutions are what the cases exist to test. A wrong expectation fails a correct
submission in `score`, and it looks exactly like a bug in the submission. That
is the one hole in this stage and it is stated rather than papered over.

**Whether the cases are interesting.** `why` must be non-empty; nothing reads
it.

`strong` covers what the pass statement claims: the field and duplicate checks
are total over the document, and the two environment-dependent checks name
themselves in the transcript when they could not run.

## How it runs

`entry.sh` is the body. `validator.ScriptBodyRunner` runs it in a freshly
allocated zone with `args.json`, `inputs.json` and `materials.json` beside it,
and it writes `verdict.json` — one boolean per handoff id in `inputs.json`.
