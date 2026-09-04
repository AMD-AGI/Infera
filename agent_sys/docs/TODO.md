# agent_sys — TODO

Near-term work: decisions to make and pieces to build inside the alpha. Long-term
subsystems live in [`ROADMAP.md`](ROADMAP.md).

Each item names where it came from and what would close it.

**Checked against the tree on 2026-08-28, after the implementation stage.** Three
items closed by being built; one changed shape. Marked inline rather than deleted,
because an item that was closed by construction is worth a reader knowing about.

**And nine seams opened that are not here.** Implementation surfaces questions a
design stage cannot, and they live in `interfaces.md` §5 rather than being copied
— that file is normative and this one is not. §5.8 (who materialises a Pointer's
value), §5.12 (two version allocators with no join), §5.13 (a script body has no
agent and `Verdict.agent_id` is required) and §5.14 (what publishes a handoff, and
from where) are the four that block something real.

---

## Decisions still open

| # | Item | What closes it |
|---|---|---|
| 1 | **What the demo's task actually does** | Package content, not a system decision (demo spec §1.1): the demo package can change its task without a spec changing. Still wanted — something small, verifiable, and not contrived |
| 2 | **What tasks the e2e test picks** | Settled that it is a separate artefact modelled on the demo's shape with its own tasks (demo spec §5). Which tasks is a system-level implementation-stage choice |
| 3 | ~~**Cross-handoff input validation input naming**~~ **— closed by construction** | Lookup is by handoff **uuid**, and `handoff/pointer.py` is RFC 6901 over `python-jsonpath`, the one library of six that separates malformed / missing / null. Two inputs of one kind are two uuids, so the ambiguity cannot arise. `tests/handoff/test_pointer.py` |
| 4 | ~~**Digest canonicalisation**~~ **— closed** | `handoff/digest.py`: `ALGORITHM = "agent_sys.handoff.tree.v1"`, sha256 git-shaped over the subtree, and `canonical()` is RFC 8785 JCS via `rfc8785` — wrapped by an encoder that **rejects `-0.0`**, which the library emits as `0` (measured). Written down rather than decided by accident |

| 4a | **A package's layout must separate a task's `bin` from the validators' `bin`** — *user-owned, and it is the precondition for staging* | **F19 reversed to staging** (`interfaces.md` §4.16), so a task gets a copy of what it needs rather than a grant on the package root. That only closes `env_mgr` criterion 13 if **a task's executable set can be named without dragging `validators/` along** — which is a package-layout guarantee, not something `env_mgr` can enforce. The user owns it. Until it holds, staging moves the leak rather than closing it |

## Unowned, reported more than once, recorded so they do not go stale silently

Each of these has been raised by a package that does not own it and has stayed
unclaimed. **Not blocked on anyone — nobody has them.**

| # | Item | Why it is not already fixed |
|---|---|---|
| 4b | **A typo'd `kind` in `Task.kinds` is caught by nothing at runtime** | `_participates` turns it into a no-op, and §4.16's narrowing removed the last place it would have raised. Probably `closure` check 6, at load time. **Reported twice by `env_mgr`, still unowned** |
| 4c | **P0 — a validator cannot reach the artefact its target was produced *from*, so five task packages scan the store instead** — *and the route this row previously proposed does not exist* | Was scoped to `examples/demo/logic/store.py`; the file is now `examples/demo/assets/lib/store.py` and **five copies** of it (`demo`, `demo2`, and three under `examples/llm_e2e_performance_optimization/`). See below |
| 4d | **`test_a_gate_failure_does_not_deadlock_the_next_dispatch` fails 2 runs in 4** | Green alone and green in its own file. **No cause offered** — and the day's rule applies: a red suite in a shared worktree is not evidence about anyone's change. With `agent-mod-2`. **2026-08-29, end of day: 4 full-suite runs, 4 green** (`1905 passed, 3 skipped, 4 xfailed`, ~64 s each). **Not "fixed" — the worktree was quiet, so the trigger may simply have been absent**, which is the converse of the rule above and the same instrument problem. Running it alone proves nothing and was already known not to; recorded so the next person starts from four data points rather than repeating the isolated run |
| 4e | **A hole in the store has no reaper** | §4.14 makes holes permanent and never renumbered by design. Whether they should ever be collected is undecided, not deferred |
| 4f | **`check_grounded` has never been observed catching anything** — *ruled parked 2026-08-29, deliberately not worked* | Criterion 10 aims to show a validator catching an ungrounded number; three end-to-end runs showed a good model **declining to fabricate one** instead, so the validator's **failing** direction — what its `strong` claim is about — has never executed. **The user's ruling: not a framework question and not a principle question, this is `check_grounded`'s own business semantics, and it is not worth the time.** The shape they suggested if anyone ever picks it up: **split it in two** — one validator over the other fields, and a second that judges only whether the agent's answer about the missing duration is *reasonable*, passing if it is. **Two measurements bear on any such build:** `check_grounded` matches `\d+`, *"digits, not a parser"*, so `256` reads as grounded via `sha256_prefix` — the grounding set is **wider than what the facts assert**, and a fabricated number landing inside any digit run in the copied facts passes anyway. And `logic/check_grounded/readme.md` named the `UNEXPECTED_SUCCESS`/exit-3 outcome in advance, so **exit 3 is the artefact working, not a fault to repair** |

### 4c in full — why the store scan exists, and why declaring the input would not remove it

**This row said the declared route was `materials.json`, and that reaching a
non-target artefact was a matter of *declaring* it — `inputs: ['summary',
'facts']`. Read against the code, that fix does not work.** A validator's
`inputs` is a **filter over the task's slots on this phase's side, not a
request**. `validator/phase.py:731`:

```python
return list(task.inputs if kind is PhaseKind.INPUT else task.outputs)
```

and `phase.py:657` selects from exactly that: `mine = [t for t in targets if
self._kind_of(t, registry) in spec.inputs]`. `env_mgr/prepare.py:691-695` stages
the same set. So a kind the task does not hold **on that side** is not a target,
is not staged, and cannot be declared into existence.

**The concrete case.** `check_problems` must verify that the problem set cites a
direction that exists — i.e. that the artefact is faithful to what its producer
consumed. `directions` is on the producing task's **input** side; the validator
runs on that task's **output** phase (and again on the students' input phases,
where the candidate set is `[problems]` too). The two sides never meet in any
phase, so there is no phase in which `directions` is reachable. Its declaration
is `inputs: [problems]` (`examples/demo2/steps/problems.yaml:48`) while its body
reads `directions` — so the schema's own promise for that field, *"DECLARED
rather than discovered, so a reviewer can answer 'what does this actually read'
without running it"*, is already false here. Same shape in
`demo/check_grounded`.

**What the packages do instead.** `lib/store.py` reads `handoff`'s on-disk
layout through `AGENT_SYS_DEMO_STORE` (`cli/main.py:825`) and scans for *the
newest artefact of that kind anywhere in the store*. Its own docstring calls
that crude and wrong in a graph with more than one producer; it happens to be
right in these packages because there is exactly one. The ~30
`staged_content(hid) or content_dir(hid)` sites are a different thing and not
this problem — each is commented as the fallback for a run with **no `env_mgr`
wired**, i.e. a validator run standalone.

**Why it is P0 and why it is quiet.** The scan is only alive because two things
are switched off: `prepare_validation` *"does not confine anything"*
(`env_mgr/prepare.py:686`, and `EnvManager.prepare_validation` at `:751` records
that who confines a validation body *"is a third question that this ruling did
not settle"*), and `AGENT_SYS_NO_PERMISSIONS` defaults to on
(`prepare.py:80-95`). Either one landing kills the route — `env_mgr`'s `p11`
measured **EACCES on the store root from a confined body**, and `store_root()`
is `os.environ[...]` rather than `.get`, so the body dies *before*
`write_verdict` and `PhaseRunner` gets **no `verdict.json` at all** rather than
a `False`. So confining validations — ROADMAP §6.1's P0 — silently converts a
grounding check into a missing file. **Not measured:** whether a confined
*validation* body fails the same way an agent body does; that needs a policy
applied to one validation zone and a run.

**The fix that removes the knowledge rather than moving it.** Give the output
phase read access to the producer's inputs — stage `task.inputs` read-only
alongside `task.outputs` in `prepare_validation`, and let `inputs:` select from
the union. Then the declaration becomes true, `declared_dir` is the only route a
body needs, and `versions` / `content_dir` / `kind_of` / `latest_of_kind` /
`handoff_dir` delete from all five copies. **A design question for `validator`
and `env_mgr` jointly, not a patch** — it widens what an output validation may
see, which is a criterion-13 (anti-gaming) question and must be argued there
before it is built.

Raised again 2026-09-04 while labelling runtime directories (PR #156), which is
what made the five duplicated readers visible in one diff.

## To build in the alpha

| # | Item | Note |
|---|---|---|
| 5 | **A whole-system CLI** | Receives a global task, a config YAML, and some CLI options, and runs the whole thing. Currently only the demo has an entry point |
| 6 | **`env_mgr` submodule that sets up the Claude Code SDK** | Installs and configures the SDK from the API key and endpoint supplied in config, so a fresh machine can run an agent without hand-setup |
| 7 | **A skill / rule / hook set per handoff content type** — *the delivery mechanism now exists*: `env_mgr/material.py` deploys `rules`, `hooks` and `skills` into the zone (`MATERIAL_KEYS`). What is missing is the four sets themselves | Every content type — reproducible, code, structured text, text — needs its own agent skill set to produce it correctly. Four sets, not one generic one |
| 8 | **The `--validation-strict-level` CLI switch** | Controls whether a validation phase may be skipped: by config, or because something else already validated the handoff |
| 9 | **The mandatory-knowledge CLI option** | Knowledge parts are strongly suggested with a warning by default; this flag makes them mandatory |
| 10 | **Agent-harness format transform helper** | Converts rules, hooks, and skills between harness formats. Claude Code's format is the canonical stored form. An independent module — `agent_harness_backend_transform_helper` or similar |
| 11 | **Remote↔local operations as agent tool calls** | Not a natural-language description of a procedure. An agent should call a tool, so it can actually use it reliably |
| 12 | ~~**Task-agent env reuse**~~ **— the alpha half is built** | The general mechanism stays deferred (roadmap §6). The validation-phase answer is `validator/environment.py::choose_configuration`, spec §8.2's chain: the bound env, else the consumer's for input validation and the producer's for output validation, else a predefined global one |
