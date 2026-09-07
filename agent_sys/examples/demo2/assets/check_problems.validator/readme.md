# check_problems — completeness, strong

Every problem carries every declared field, none of them empty; the set is the
requested length; every `leetcode_slug` is in the closed index; every
`direction_id` names a direction that exists in the artefact this set was built
from; no `id` repeats.

## Why `strong` is honest here

Every rule is a presence test, a set membership or a count. Nothing is
estimated, nothing is sampled, and the check reads every problem and every one
of the eight required fields:

`id`, `direction_id`, `leetcode_slug`, `title`, `statement`, `input_format`,
`output_format`, `constraints` — each a non-empty string — plus `examples` as a
non-empty list.

**`examples` is checked for presence only.** Whether those examples are *worked*
— two of them, exact bytes on both sides — is a different question in a
different dimension, and `check_solvable` answers it as `weak` rather than
having this one quietly overclaim.

## What it will not tell you

Whether a problem is well posed, whether its `constraints` are consistent with
its `statement`, or whether the slug it cites is a sensible match for the
direction it came from. The prompt asks the setter to prefer a slug whose tags
include the direction's tag; nothing here enforces that, because "is this a good
example of that area" is a judgement and this is a completeness check.

## Reaching the directions artefact — two routes, one of them crude

The referential check needs a second handoff, and an output phase stages only
what it validates. So:

1. `store.declared_dir("directions")` reads `AGENT_SYS_INPUT_DIRECTIONS` — the
   artefact the producing task actually consumed. **Exact, and available in
   `problems`' output phase only**: `validator.choose_configuration` uses the
   producer's configuration on the output phase and there is no consumer row at
   all (`validator/environment.py:116-142`).
2. `store.latest_of_kind("directions")` — *the newest `directions` anywhere in
   the store*. Its own docstring calls this crude, and in a graph with several
   producers of one kind it would be wrong. Here it is not: `main` has exactly
   one producer of `directions`, and the single-slot `producer_of[kind]`
   (`task_graph/models.py:360`) means a consumer could not see a second one
   even if there were. It reaches the store through `AGENT_SYS_DEMO_STORE`,
   which the global configuration row does carry, so it is the route that fires
   in the three students' input phases.

If neither resolves, **every handoff is False**. A check that could not run has
not found nothing. Same shape, same reason, as
`examples/demo/assets/check_grounded.validator/check.py`'s fallback.

## The index shape it reads

```json
{"problems": [
  {"slug": "merge-k-sorted-lists",
   "title": "Merge k Sorted Lists",
   "topic_id": "divide_and_conquer",
   "difficulty": "hard"}
]}
```

Only `slug` is read. `topic_id` is guidance the prompt gives the setter — prefer
a slug whose topic is the direction you are writing it under — and it is
deliberately **not** enforced here: with 18 topics and 67 rows, a direction can
run out of index entries before it runs out of problems, and turning a
preference into a rule would fail a set that is otherwise correct.

Missing, unparseable, or no non-empty `problems` list: every handoff False, with
the reason printed. An empty permitted set would fail every problem too and
would read exactly like a setter who invented all twelve.

## How it runs

`entry.sh` is the body; `validator.ScriptBodyRunner` runs it in a fresh zone
holding `args.json`, `inputs.json` and `materials.json`, and it writes
`verdict.json` — one boolean per handoff id.

It runs **four times per graph**: `problems`' output phase, then the input phase
of each of `solve_a`, `solve_b` and `solve_c`. That is why `expected_count`
comes through `args.json` and not through the `setter` agent's `env`: the three
input phases never see that agent's environment.

Every fault is printed, not just the first. A run here costs several model
calls, and one fix per run is not a debug loop.
