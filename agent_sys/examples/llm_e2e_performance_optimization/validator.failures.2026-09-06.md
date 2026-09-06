
## check_deploy_kit — crashed, no verdict (NOT a refusal)

| | |
|---|---|
| when | 2026-09-06T07:47:22Z (crash at 07:29:59Z) |
| run | `/data/yihou/agent_sys_runroot/runs/20260906T064218-15c264/` |
| module | m1, `deploy_and_prove`, real `kind: ai` agent, real TP4 bring-up |
| verdict | **none** — `exited 1 and wrote no verdict.json; nothing was decided` |
| cause | `ImportError: Draft202012Validator` — system jsonschema 3.2.0, see `bug.record.2026-09-06.md` |

**This says nothing about the artefact.** The kit was sealed by a real agent after
two successful TP4 bring-ups; the validator never read it. Filing it here so the
next reader does not count it as evidence against the kit — a crash that arrives
before the check is not a judgement, and `deploy_kit: invalid` in the run summary
is the framework's word for "undecided", not for "bad".

**check_environment and check_deploy_serves: NOT RUN.** `check_deploy_kit` died
first and nothing after it executed. So this run establishes zero of m1's three
verdicts. The zone file counts are therefore not yet meaningful and are not
reported.
