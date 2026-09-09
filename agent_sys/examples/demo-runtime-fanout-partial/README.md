# demo-runtime-fanout-partial

Validates that a partial failure in a fan-out blocks the fan-in consumer, while
the sibling branches that passed stay VALID.

One validator judges three kinds, the same shape as `examples/demo2`'s
`check_compiles` over `solutions_a` / `solutions_b` / `solutions_c`. All bodies
are `kind: program`.

## Topology

```
main                        non-leaf: readme.md, no entry.sh
│
├── emit_a ───────────────► thing_a  [structured_text]   check_thing  PASS
├── emit_b ───────────────► thing_b  [structured_text]   check_thing  FAIL
├── emit_c ───────────────► thing_c  [structured_text]   check_thing  PASS
│         (three roots, froms: [])
│
└── merge                   is_end, froms: [emit_a, emit_b, emit_c]
      inputs: thing_a, thing_b, thing_c — needs all three VALID
```

## Why `check_thing` fails

`check_thing` is declared once with `inputs: [thing_a, thing_b, thing_c]` and
`args.required_row_keys: [name, value, tag]`. Its body,
`assets/check_thing.validator/check.py`, reads `items/text.json`, requires
`rows` to be a non-empty list in which every row carries all three required
keys, and requires `totals.count` to equal `len(rows)`.

- `emit_a` and `emit_c` write rows with `name`, `value`, and `tag`, so both
  pass.
- `assets/emit_b.task/emit.py` writes three rows carrying only `name` and
  `value`. The `tag` key is absent, `any(key not in row for key in required)`
  is true, and the verdict is FAIL, so `thing_b` is sealed INVALID.

`merge` declares all three kinds as inputs. Two of them are VALID and the third
never becomes VALID, so the fan-in gate never opens and `merge` never starts.
The failure is confined to one branch: it does not invalidate `thing_a` or
`thing_c`.

## Expected terminal states

| Object | Terminal state | Verdict |
|---|---|---|
| `emit_a` | SUCCEEDED | `check_thing` PASS on `thing_a` |
| `emit_b` | OUTPUT_VALIDATING | `check_thing` FAIL on `thing_b` |
| `emit_c` | SUCCEEDED | `check_thing` PASS on `thing_c` |
| `merge` | WAITING_HANDOFF | input `thing_b` is INVALID |
| `thing_a` | VALID | — |
| `thing_b` | INVALID | — |
| `thing_c` | VALID | — |

The run ends quiescent with `merge` in `WAITING_HANDOFF`. The two expectations
registered in `agent_sys/cli/expectations.py` under
`demo-runtime-fanout-partial` are `thing_b_verdict_fails` and `merge_waits`;
observing both is exit 0.

## Run

```bash
AGENT_SYS_NO_PERMISSIONS=1 agent-sys run --package agent_sys/examples/demo-runtime-fanout-partial
```
