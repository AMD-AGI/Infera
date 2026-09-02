# check_reproduces — usability, weak

Hand the `runbook` handoff's packup to a **fresh Claude Code session**, tell it
to follow `REPRODUCE.md` for real, and pass iff it reports that it reproduced
the run **and** every artefact its report names exists and is non-empty.

The mission's words are *"ai: 拿到handoff成功运行"* — take the handoff and
successfully run it. That is a `usability` question: not *is the artefact
complete* (`check_packup_shape` asks that) but *can somebody who was not there
actually use it*.

## It is not a different kind of validator

`validator.schema.json` has **no `kind:` field** and `additionalProperties:
false`, so there is no such thing as an "AI validator" in the spec vocabulary.
This body is a script like every other validator body, run by
`validator.ScriptBodyRunner`, and what makes it the AI one is that it spends its
time inside a `claude` subprocess. The two are told apart by their tags —
`external_dynamic`/`gpu_hours` here against `external_static`/`seconds` on the
shape check — which is the pair `validator` spec §5.3 orders a phase by, cheap
first. So the shape check runs and can fail before this one is ever started.

## Why it is `weak`, and what the corroboration is actually worth

**The verdict originates with the reproducer, not with this body.** The
reproducer reads the kit, runs it, and judges the result against the kit's own
`Expected output` section. This body then does exactly one thing on top of that
report: it checks that every artefact the report names **exists inside the kit
copy and is non-empty**.

That is a real corroboration and it is much less than having verified the run:

- It catches the report that claims success and produced nothing — the failure
  mode a language model asked "did it work" is most likely to have.
- It does **not** catch a report that produced a plausible file with wrong
  contents. Nothing here reads an artefact's bytes, because what those bytes
  should say is the *kit's* claim and differs per experiment.

`strength` qualifies a PASS (`validator` spec §5.4). A PASS here means: *a fresh
session followed this kit end to end, said it reproduced, and left behind the
files it said it produced.* It does not mean this validator watched the model
serve a token. Calling that `strong` would be the dishonesty
`examples/demo/steps/describe.yaml` is about — *a sophisticated check that is
silently approximate is not a `strong` validator*.

A `False`, by contrast, is a hard finding: `REPRODUCE.md` could not be followed
to a result on this machine, and the transcript beside the failure says how far
it got.

## The criterion lives in the kit, not in this prompt

The prompt is deliberately silent about what success looks like. It points the
reproducer at `REPRODUCE.md`'s `Expected output` section — or `README.md`'s
`Result` section when there is none — and says *those sections are the
criterion; nothing else is*.

Restating the experiment's success condition in this prompt would mean the kit
was checked against **this validator's** idea of the experiment rather than its
own, and a kit whose `Expected output` was wrong or missing would still pass.
The property being measured is that the kit is self-contained, so the kit has to
supply the bar.

It is also what makes this validator testable without a GPU. A canned kit whose
`REPRODUCE.md` says *run `echo`, then check that a file appeared* is reproduced
in seconds, and a canned kit whose commands do not work fails — both under the
same body, with no special case in it. That pair, a kit that reproduces and a
kit that cannot, is how this body is exercised without hardware: an instrument
only ever pointed at the good case proves nothing.

## The contract with the reproducer

It works in a **copy** of the kit — `shutil.copytree` into
`<validation zone>/reproduce/<handoff id>/` — because the staged content is what
the phase is validating and this body has no business writing into it, and
because the reproducer must be able to create files beside the kit it follows.

It must leave `./reproduction.json` behind:

```json
{"reproduced": true,
 "evidence": "one paragraph: what was run and what was observed",
 "artifacts": ["paths, relative to the kit, of files produced that back it"]}
```

Named `reproduction.json` and not `verdict.json` on purpose: `verdict.json` is
the validation zone's own file and belongs to `PhaseRunner`'s seam.

Each fault is separate and reported separately:

| fault | verdict |
|---|---|
| `claude` is neither on `PATH` nor at `$HOME/.local/bin/claude` | FAIL |
| it did not finish inside `args.timeout_seconds` | FAIL |
| it exited non-zero | FAIL |
| no `reproduction.json`, or it is not readable JSON | FAIL |
| `reproduced` is not a boolean, or `evidence` is empty | FAIL |
| `reproduced: false` | FAIL — and that is the honest verdict, not an error |
| `reproduced: true` with no artefacts | FAIL |
| an artefact is missing, empty, or resolves outside the kit copy | FAIL |

The artefact path is resolved and required to stay inside the kit copy. An
artefact somewhere else on the machine is not evidence this validation can stand
behind — it may predate the run.

## Two things it does deliberately

**It runs the reproducer with `--dangerously-skip-permissions`.** The
reproducer's whole job is to execute the kit's commands; under `-p` an
unapproved tool call is denied rather than prompted, so a narrower flag would
fail every kit for a reason that has nothing to do with the kit. This stage runs
with `agent_sys`'s own permission enforcement off by default and this is the
same decision one layer down. It is recorded here rather than buried because it
is the sharpest edge in this package.

**The whole transcript goes to a file, never to stderr.**
`validator/phase.py:167-172` folds a body's stderr tail into an exception
message, so anything written there travels into the event stream. The transcript
is `<validation zone>/reproduce/<handoff id>/claude.log`, and a failing verdict
is meant to be read there.

## Secrets

`ANTHROPIC_API_KEY` and `ANTHROPIC_BASE_URL` reach this body **by name**, from
the operator's own `~/.claude/settings.json` `env` block, through
`env_mgr.harness.harness_env` and the two rows of `validator` spec §8.2 that
carry it. Nothing in this package writes a value: not in YAML, not in `args.json`, not in a `--var`.

`claude` consumes them itself. This module never reads either variable, and the
prompt tells the reproducer in as many words never to print an environment
variable's value or copy one into a file.

## Cost

`gpu_hours`, which is the honest order of magnitude for the real handoff: the
reproducer brings a 27 B model up, and on ROCm the cold start alone runs to
minutes of JIT compilation before the server is ready. `args.timeout_seconds` defaults to
5400 and is a `--var` knob — `--var reproduce_timeout_seconds=120` is what a
bring-up run passes.

## Layout

`entry.sh` is the command, `check.py` is the implementation, and
`assets/lib/zone.py` is the four body-facing zone files shared with
`check_packup_shape`.
