# Workflow patterns for running a package

## The generated script (`astk workflow`)

A promise per leaf, not `parallel()` per stage: a leaf starts the moment its
own upstream leaves are done — the agent_sys scheduler's readiness rule — so
the slowest branch does not hold the others. A leaf counts as done only when
its agent reports `finish_exit: 0` **and** the run state agrees: before a
dependent starts, a low-effort gate agent reads `astk status --json`, and the
dependent is skipped unless every upstream leaf is `succeeded` (`astk prepare`
refuses the same case on its own). `phase` groups leaves by their parent
non-leaf for the progress view. Each program leaf costs one low-effort agent.

## Refining a leaf into stages

Do it when a leaf's readme has a long STEPS list whose steps have their own
acceptance (e2e-flow `optimize_kernel`, `deploy_and_prove`). Keep `prepare`
first and `finish` last, in the same zone; the stages run **sequentially**:

```js
const prep = await agent(`astk prepare --run R --leaf L; return the zone path`, {...})
for (const s of STAGES) {                      // e.g. [{name:'premise', steps:'0-1'}, ...]
  const r = await agent(`In zone ${prep}, after '. env.sh': do STEPS ${s.steps} of brief.md, ` +
                        `stop at the first acceptance that fails; report it.`, {schema: STAGE})
  if (!r.ok) return {leaf: 'L', status: 'failed', summary: r.reason}
}
return agent(`astk finish --run R --leaf L; fix the handoff and retry (<= 3)`, {schema: RESULT})
```

Worth adding before `finish`: an agent asked to *refute* the handoff's claim
(does the number come from the command it names?) — it catches what a schema cannot.

## Debugging and fast iteration

- Re-run one leaf: `astk prepare` again opens a new attempt; upstream handoffs are reused.
- Skip ahead: `astk seed` a downstream input from an earlier run instead of re-running its
  producer (also after it failed: once every kind it produces is seeded, it shows `seeded`).
  Say in the report which leaves were skipped.
- Mock stages a package supports (`--var mock_stages=…` in e2e-flow) work unchanged.

## Divergences from an agent_sys run (always list them in the report)

| agent_sys | here |
|---|---|
| AI leaf = `ClaudeSdkBackend` session, system prompt = readme + outputs brief, `CLAUDE_CONFIG_DIR` in the zone | a Claude Code (sub)agent reading `brief.md` (the same text); the operator's CLAUDE.md and settings apply |
| workspace = `git clone --shared` of the main repo | an empty `workspace/` |
| no retry on a failed validation; the producer never sees why | `finish` shows the reasons and allows ≤ 3 attempts; the count is in `state.json` |
| agent-bodied validators: UNREACHED (no executor registered) | UNREACHED as well; Claude may add a labelled judgement of its own |
| input validation re-runs input kinds' validators, but the default strict level reuses the verdict recorded at publication | inputs were validated when published (`finish`/`seed`); the leaf's own phase validators run at `prepare` as the input phase, their verdicts appended to the input's `validation.yaml` |
| a failed output phase seals the attempt's model versions invalid | a failed `finish` publishes no output of the attempt |
| confinement off by default | the same (no confinement) |
| run store records, events, monitor | `state.json`, `validation.yaml`, `manifest.yaml` only |

Handoff layout, seal test, gate test, validator contract and verdict rows are
the same, so a published handoff here is a published handoff there.
