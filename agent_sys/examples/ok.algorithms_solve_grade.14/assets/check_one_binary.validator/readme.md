# check_one_binary — completeness, strong

The requirement was *one unified executable that selects by test case*. This
checks that literally: **exactly one** executable in the staged content, and it
really does select.

## What it checks

| | |
|---|---|
| the count | exactly one executable regular file anywhere under the content. Not "at least one" |
| the manifest | `items/manifest.json` parses, declares a non-empty `cases` list, and its `binary` path is the executable that was found |
| `--list` | the binary runs and prints **exactly** the case ids the manifest declares — no missing id, no extra one |
| a **negative control** | a case id no manifest could contain exits `2`, the harness's code for *unknown case id* |
| a spot-check | the first declared case id is accepted, runs, and does not time out |

The count is asserted rather than assumed because it is the whole requirement.
Thirty-six binaries in a directory would list correctly, dispatch correctly and
run correctly, and would be the wrong artefact.

## The negative control, and why it is here

A dispatcher that ignored its argument and always ran the same solution would
pass every positive test above: it would list correctly (the list is a static
array), and the spot-check would run something and exit 0.

So an id that cannot be in any manifest is sent first, and it must be rejected
with exit code `2`. Only then is a real id sent. Without that pair, the
spot-check is an instrument pointed at the safe case, and a pass would carry no
information about whether the selection works.

The spot-check itself asserts that the case *ran* and does **not** compare the
output. Whether a solution produces the right answer is `score`'s question, and
answering it here would make two validators owners of one fact.

## Two mechanical notes

**The selector goes on stdin, not on argv.** `assets/lib/cpp.py`'s frozen
`run(binary, stdin_text, *, timeout)` executes `[str(binary)]` and passes no
argv, so nothing driven through that helper can use an argv-only interface.
`assets/harness.task/build.py` generates a binary that takes the case id from
`argv[1]` **or** from the first line of stdin for exactly this reason, and this
body uses the second channel.

**The binary is copied before it is run.** Whether the executable bit survives
`env_mgr`'s staging is not something a validator body can assume, and it has no
business `chmod`ing its own input. So it is copied to a temporary directory,
made executable there, and the copy is thrown away afterwards.

## What it does not check

That every listed case id runs — one is sampled, and thirty-six process
launches in a validator is a cost with no matching gain, since `score` runs
every one of them a moment later against real input and records the result.
That the answers are right; that is `score` and `check_scores`.

`strong` is honest for what is claimed: the count, the manifest agreement and
the discrimination are all total, and the sampling is confined to the one
property — *"and this id also runs"* — that the pass statement names.

## How it runs

`entry.sh` is the body. `validator.ScriptBodyRunner` runs it in a freshly
allocated zone with `args.json`, `inputs.json` and `materials.json` beside it,
and it writes `verdict.json` — one boolean per handoff id in `inputs.json`.
