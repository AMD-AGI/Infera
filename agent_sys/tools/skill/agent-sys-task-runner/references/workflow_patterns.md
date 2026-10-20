# Workflow patterns for running a package

## The generated script (`astk workflow`)

```js
const done = {}
for (const leaf of PLAN) {                       // listing order is topological
  done[leaf.name] = Promise.all(leaf.deps.map(d => done[d])).then(async deps => {
    if (deps.some(r => !r || r.status !== 'succeeded')) return {leaf: leaf.name, status: 'skipped', ...}
    return agent(leaf.prompt, {label: leaf.name, phase: leaf.phase, schema: RESULT})
  })
}
```

Each program leaf costs one low-effort agent (it only runs prepare,
run-program and finish) — acceptable for a readable graph; run program leaves
inline instead if that matters. A leaf counts as done only when its agent
reports `finish_exit: 0`.

A promise per leaf, not `parallel()` per stage: a leaf starts the moment its own
inputs are published, which is the agent_sys scheduler's readiness rule, and
the slowest branch does not hold the others. `phase` groups leaves by their
parent non-leaf (m1_deploy, m2_profiling, …) for the progress view.

## Refining a leaf into stages

Do it when a leaf's readme has a long STEPS list whose steps have their own
acceptance (e2e-flow `optimize_kernel`: 10_read_inputs … 70_selfcheck;
`deploy_and_prove`: STEPS 0–8). Keep `prepare` first and `finish` last, in the
same zone; the stages run **sequentially** (they share the zone):

```js
const prep = await agent(`astk prepare --run R --leaf L; return the zone path`, {...})
for (const s of STAGES) {                      // e.g. [{name:'premise', steps:'0-1'}, ...]
  const r = await agent(`In zone ${prep}, after '. env.sh': do STEPS ${s.steps} of brief.md, ` +
                        `stop at the first acceptance that fails; report it.`, {schema: STAGE})
  if (!r.ok) return {leaf: 'L', status: 'failed', summary: r.reason}
}
return agent(`astk finish --run R --leaf L; fix the handoff and retry (<= 3)`, {schema: RESULT})
```

Useful inside a stage: an adversarial check before `finish` — a second agent
asked to *refute* the claim the handoff makes (does the number come from the
command it names? does the README describe what is in items/?). It costs one
agent and catches what a schema cannot.

## Debugging and fast iteration

- Re-run one leaf: `astk prepare` again opens a new attempt (new zone, new
  output version); published upstream handoffs are reused.
- Skip ahead: `astk seed` a downstream input from an earlier run's published
  handoff instead of re-running its producer. That proves the leaves you ran,
  not the one you skipped — say so in the report.
- Mock stages that a package supports (`--var mock_stages=…` in e2e-flow) work
  unchanged: the program leaf runs its mock path.

## Divergences from an agent_sys run (always list them in the report)

| agent_sys | here |
|---|---|
| AI leaf = `ClaudeSdkBackend` session, system prompt = readme + outputs brief, `CLAUDE_CONFIG_DIR` in the zone | a Claude Code (sub)agent reading `brief.md` (the same text); the operator's CLAUDE.md and settings apply |
| workspace = `git clone --shared` of the main repo | an empty `workspace/` |
| no retry on a failed validation; the producer never sees why | `finish` shows the reasons and allows ≤ 3 attempts; the count is in `state.json` |
| agent-bodied validators: UNREACHED (no executor registered) | UNREACHED as well; Claude may add a labelled judgement of its own |
| input validation re-runs input kinds' validators, but the default strict level reuses the verdict recorded at publication | inputs were validated when published (`finish`/`seed`); not re-run |
| confinement off by default | the same (no confinement) |
| run store records, events, monitor | `state.json`, `validation.yaml`, `manifest.yaml` only |

Handoff layout, seal test, gate test, validator contract and verdict rows are
the same, so a published handoff here is a published handoff there.
