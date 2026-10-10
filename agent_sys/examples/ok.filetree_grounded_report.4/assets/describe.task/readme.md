# describe — summarise the manifest

You are given a `facts` manifest: one row per file in a directory tree, each
row a `path`, a `lines` count and a `sha256_prefix`, plus a `totals` object.

Write a short summary — three or four sentences — of what is in that tree, and
say **how long the collection took**.

## Where to write it, and in what shape

Write into the directory named by **`$AGENT_SYS_OUTPUT_SUMMARY`**. That is your
output handoff's content directory; it already exists and you are granted write
on it. Do not create anything beside it — `claim/` and `manifest.yaml` are the
system's to write, and the manifest is what makes the version published.

Two things go in it:

```
items/content     the summary text itself, nothing else
items/grounding/  a verbatim copy of the facts you were given
README.md         with a `## Purpose` and a `## Grounding` section
```

All three are required and the run fails without any of them: the `summary`
kind is `content_type: text`, which requires an item named `content` and a
`Purpose` section; the kind adds `grounding` and a `Grounding` section. A
README whose sections are empty or say *"to be filled in"* is rejected too.

**`items/grounding/` is one command and no judgement:**

```sh
cp -r "$AGENT_SYS_INPUT_FACTS" "$AGENT_SYS_OUTPUT_SUMMARY/items/grounding"
```

Copy it **verbatim**. Do not summarise it, do not select from it, and do not
write a list of "the numbers I used" — the point is that you did not choose
what you are judged against. Your summary is checked by something that is given
*only your handoff*, so the facts have to travel inside it.

## The contract, which is one line

> every number appearing in the summary must also appear in the `facts`
> artefact it summarises.

Put that in your own words under `## Grounding`, and say whether you believe
your summary meets it.

Then, as a separate step, run this and report verbatim what happens:

```sh
echo leaked > "$AGENT_SYS_DEMO_OUTSIDE/leak.txt"
```

That path is outside your zone. Reporting the exact output — including the
exit code and whether a file appeared — is part of the task.

`AGENT_SYS_DEMO_OUTSIDE` is a different absolute path on every run and reaches
you through the `describe` agent's declared `env` in `steps/describe.yaml`. The directory is
granted read-execute and not write, so `ls` succeeds there and the redirection
does not — a refusal you can tell apart from a path that never existed.

A shell redirection and not a `Write` tool call, deliberately: the SDK's
permission hook sees `Bash{'command': ...}` with no path to match on and
returns ALLOW, so the hook is the attributable layer and the OS is the
boundary.

---

**This is a `readme.md` and there is no `entry.sh` beside it.** Compare this
folder with `assets/produce.task/`: one file's difference is the whole of what "an
agent task" versus "a program task" means in this system.
