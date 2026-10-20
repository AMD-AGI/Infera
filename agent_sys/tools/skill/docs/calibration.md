# Validator calibration — strict on what matters, silent on the rest

A validator that refuses reasonable results trains people to bypass it; one
that passes anything is theatre. The bar is set by what **every correct result
satisfies**, and a validator says, in its refusal, how to fix the handoff.

## Check

1. **Presence** — the items and files the next consumer needs exist and are not empty.
2. **Structure** — JSON/YAML parses; the keys a consumer reads are there.
3. **Executability** — `command`/`script` items are `+x` and parse (`sh -n`).
4. **Evidence** — a result names the command that produced it and the
   environment it ran in; a reproduction claim carries the numbers it reproduced.
5. **Stated bounds** — only for a number the task is *about*, with the tolerance
   written next to it and the reason (`numeric: [{file, key, min, max, why}]`).
6. **Hygiene** — no unfilled template text, no literal secret values.

## Do not check

- Wording, tone, headings beyond what agent_sys requires, or length beyond a
  stub-detecting floor. A floor goes **well under** the real result
  (about a third of it) and is labelled a floor.
- An exact number. Shared machines drift; e2e-flow measured that a tolerance as
  tight as within-run spread fails honest evidence across sittings.
- Anything the producer cannot control (a busy node, a flaky network) as a hard
  failure — put it in a `note:`.

## Repo precedents worth copying

- `check_bench_report`: "a `max_error_rate` of zero turns this into a flake detector".
- `check_deploy_serves`: floors "deliberately low… a floor that encoded a
  performance expectation would fail a healthy deployment on a busy node".
- `check_workset_runs`: different tolerances for within-sitting (0.10) and
  cross-sitting (0.25) — and "do not widen either to make a run pass".
- `check_packup_shape`: presence alone is gamed by template bodies, so it
  counts content lines and command lines.
- `validator/docs/spec.md`: a validator is `strong` when it can state in advance
  the number or comparison that decides it; otherwise `weak`. A crude check that
  is honestly described is strong; a sophisticated one that is silently
  approximate is not.

## Strong vs weak — examples

| check | label | why |
|---|---|---|
| `items/command` exists, is +x, parses under `sh -n` | strong | the decision is stated in advance and exact |
| error rate ≤ 0.05 in `result/summary.json` | strong | a stated number decides it |
| ≥ 50% of cited `path:line`s resolve to a file with that many lines | strong, **if its brief says exactly that** | the number is stated in advance; it does not prove a line supports the claim, so the brief must not claim more ("citations resolve", not "citations are correct") |
| an LLM judges whether the report answers the question | weak | no number decides it in advance |

A `weak` pass is low-confidence evidence: open the artefact before trusting it.

## Prove the calibration (the skills do this every time)

| control | input | must |
|---|---|---|
| positive | the real result from the history / the work being packed | **PASS** |
| negative | `astk template` skeleton (placeholders) | FAIL |
| negative | an empty content dir | FAIL (seal refuses) |

If the positive control fails, decide which is wrong — the result or the bar —
and write the decision into the plan's `calibration:` field (scaffold renders it
into the validator's readme; a hand-maintained package writes it there directly). Never loosen a bar to pass
a result you would not hand to a colleague; never pad a result to pass a bar.

## Size and secrets (the skills' rules; agent_sys has neither)

- A file over 4 MB is a question for the user: include, reference by absolute
  path + sha256, or drop. Logs are a question too (keep, trim, gzip).
- Secrets: record the variable's **name** and **where it comes from**, never the
  value. `astk audit` and the generic validator both refuse a literal value.
