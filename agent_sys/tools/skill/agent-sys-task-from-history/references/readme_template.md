# The task readme — the prompt an agent gets

For an AI task this file is the whole system prompt (the 100-char `goal` is
the user prompt). It carries the expert knowledge the history paid for. The
repo's strongest readmes (`e2e-flow/assets/deploy_and_prove.task/readme.md`,
`optimize_kernel.task/readme.md`) all follow this shape: **the AI sequences and
reads; it does not invent the method.**

```markdown
# <task name> — <one line: what comes out>

## What you are given

- `$AGENT_SYS_INPUT_<KIND>` — <what it holds, which file to read first>
- variables: `$MODEL_PATH` (…), `$IMAGE` (…)   ← from the agent's env: block
- the machine: <what is true about it; what to check before using it (shared GPUs!)>

## Success, stated so it can fail

<the concrete bar, copied from the user's words in the history: which number,
which comparison, which file proves it. If the user never stated one, say what
the history's result achieved and mark the bar "proposed".>

## STEPS

Each step: the command (copy-pasteable, from what worked in the history), then
**Accept when:** <the observable that says it worked>.

0. Preflight — <GPU free? image present? weights readable?> Accept when: …
1. <…> Accept when: …
2. <…>

## Traps (each one cost time in the history)

- **<symptom as the agent will see it>** — <cause> — <fix>.
- …

## What you must write

Into `$AGENT_SYS_OUTPUT_<KIND>`:

    README.md        ## <the content type's sections, exactly>
    items/<…>        <what each item holds>

<For `reproducible`: `items/command` is an executable script that reproduces
`items/result` from scratch; `items/env/` records the environment (no secret values).>

## What is checked

<one line per validator: its brief and the bar, so the agent knows the target
without reading the validator's code.>

## Do not

- <the rules from the history: no deleting outside <dir>, which GPUs are not
  yours, no secret values in the handoff, …>
```

Keep it as long as the knowledge needs and no longer; cut anything an agent
would do correctly without being told.
