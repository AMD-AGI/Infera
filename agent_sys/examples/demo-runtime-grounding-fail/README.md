# demo-runtime-grounding-fail

Validates that a summary quoting a number its input cannot ground is sealed
INVALID and that the downstream consumer never starts.

This is the deterministic, pure-program version of `examples/demo`'s
`describe` → `check_grounded` FAIL path: `kind: program` bodies only, no AI
backend and no credentials.

## Topology

```
main                        non-leaf: readme.md, no entry.sh
│
├── produce ──────────────► facts    [structured_text]
│     inputs: none
│     output validation: check_facts       completeness / strong     PASS
│
├── transform ────────────► summary  [structured_text]
│     inputs: facts
│     output validation: check_grounded    trustworthiness / strong  FAIL
│
└── consume                 is_end
      inputs: summary — never becomes VALID, so the body never runs
```

## Why `check_grounded` fails

The `summary` kind's contract is set inclusion over numerals: every number in
the summary must also appear in the `facts` artefact it summarises.

- `assets/produce.task/produce.py` writes one row per file with `path`,
  `lines`, and an 8-character `sha256_prefix`, plus `totals.files` and
  `totals.lines`. It measures no duration, and the `facts` kind declares none.
- `assets/transform.task/transform.py` writes the sentence
  `The collection completed in 99999 milliseconds.` into `items/text.json`, and
  copies the whole facts artefact into `items/grounding/` so both sides of the
  comparison sit inside one staged handoff.
- `assets/check_grounded.validator/check.py` applies the `\d+` regex to
  `items/text.json` and to every file under `items/grounding/`, then computes
  `claimed - grounding`. `99999` is in the first set and not the second, so the
  verdict is FAIL and `summary` is sealed INVALID.

The reason is structural, not arranged: the numeral is absent from the
producer's output, so no re-run of `transform` can ground it.

## Expected terminal states

| Object | Terminal state | Verdict |
|---|---|---|
| `produce` | SUCCEEDED | `check_facts` PASS on `facts` |
| `transform` | OUTPUT_VALIDATING | `check_grounded` FAIL on `summary` |
| `consume` | WAITING_HANDOFF | input `summary` is INVALID |
| `facts` | VALID | — |
| `summary` | INVALID | — |

The run ends quiescent with `consume` in `WAITING_HANDOFF`. That is the correct
outcome, so the two expectations registered in `agent_sys/cli/expectations.py`
under `demo-runtime-grounding-fail` are `grounded_verdict_fails` and
`consumer_waits`; observing both is exit 0.

## Run

```bash
AGENT_SYS_NO_PERMISSIONS=1 agent-sys run --package agent_sys/examples/demo-runtime-grounding-fail
```
