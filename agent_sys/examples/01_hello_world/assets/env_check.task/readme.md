# env_check — check the measured facts yourself

You are given the `env_facts` handoff at `$AGENT_SYS_INPUT_ENV_FACTS`. Its
`items/text.json` holds facts about this machine that a script measured, and its
`README.md` says how each one was measured.

For **every** key in that JSON object, find the value yourself by running a shell
command of your own choosing — do not read it from the facts file — and compare.

## What to write

Write into `$AGENT_SYS_OUTPUT_ENV_REPORT`, which already exists:

```
items/text.json
README.md          with a `## Purpose` and a `## Schema` section
```

`items/text.json` is one JSON object:

```json
{
  "facts": { ...a verbatim copy of the input facts object... },
  "checks": {
    "<fact key>": {
      "observed": <the value you found, same type as the fact>,
      "how": "<the exact command you ran>",
      "agrees": <true if observed equals the fact, else false>,
      "note": "<one sentence; required when agrees is false>"
    }
  }
}
```

Rules your report is judged by:

- `facts` is copied verbatim, and `checks` has one entry for every key in it.
- `agrees` is exactly whether `observed` equals the fact. A disagreement honestly
  reported is fine; a disagreement reported as agreement fails the check.
- `hostname` and `in_container` are re-measured by the validator, so report what
  is really there.

Keep it short: this is a hello world. Do not install anything, and do not write
anywhere except your output directory.
