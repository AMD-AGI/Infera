
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

---

# Run `20260906T154908-d9c7af` (run 4) — the deepest board of the day

**Recorded 2026-09-06T16:52:55Z by m35.** Zero-file-zone check run first (core principle 2):
`6 zone(s), none empty`.

## Passed, and three of them for the first time on this cluster

| validator | kind | note it produced (evidence it read something) |
|---|---|---|
| `check_deploy_kit` | deploy_kit | `qwen3-32b-mix.packup_20260906` |
| `check_deploy_serves` | deploy_kit | own bring-up, router healthy on 8140 |
| `check_bench_result` | bench_result ×2 | 1262 request records, 0 errored; decode graph ceiling 32 >= concurrency 21.66 |
| `check_kernel_table` | kernel_table | 138 kernels, top 25 cover 84.8%, shares sum 99.98 — **passed on the artefact, not on the `min_launchers=0` waiver** |
| **`check_trace_coverage`** | profile_result | **the one that refused on `fdb0bd` and `298750`.** stack window 2 rank(s), 5613901 python_function events; 4 rank(s), 825176 GPU kernel events |
| **`check_profiling_evidence`** | profiling_evidence | **FIRST EXECUTION ANYWHERE.** "the ranking and the trace agree on 825176 GPU kernel events" |
| `check_worklist_shape` | kernel_worklist | (no findings) |
| `check_command_parses` | several | `bash -n` clean |

**`check_trace_coverage` passing is what `trace_end_ms=120000` bought**, and it
is the first time Path A has been tested at all — three earlier runs died before
reaching it.

## The one refusal — `check_identity_resolved` on `operator_identity`

```
5/5 resolved (ratio 1.00, floor 0.0 — a floor of zero grades nothing;
              set --var min_resolve_ratio to grade it)
PROBLEM: duplicate logical_operator(s): ['layernorm_aiter_add_rmsnorm_quant'].
         It becomes a directory name in the workset, so two of them collide silently
```

**The refusal is correct and the defect is real.** Five operators, two of which
minted the same name:

```
gemm_aiter_bf16gemm_bf16_tn
attention_ck_tile_kentry
layernorm_aiter_add_rmsnorm_quant     <- twice
layernorm_aiter_add_rmsnorm_quant     <- twice
elementwise_sgl_hip10activation18act_and_mul
```

### It is a CONTRACT §5.3 shape: the producer cannot satisfy the validator

`identify.py:236` builds the name by **dropping tile/tuning tokens and keeping
four** — deliberately lossy, so a name survives a rebuild:

```python
kept = [t for t in tokens if not noise.match(t)][:4]
```

**Nothing then enforces uniqueness, and `check_identity_resolved` requires it.**
So the collapse is correct, the requirement is correct, and no launch variable
can reconcile them. **The fix belongs at the producer**: uniqueness is a genuine
requirement and a silent directory collision is exactly what should be refused.

Patch drafted and **unapplied**:
`/data/yihou/e2e_verify_20260906/m35/identify_unique_names.patch`.
Known-answer tested on the refused artefact itself: dups `NONE` afterwards,
**3 of 5 names byte-identical**, `kernel_identity` kept in sync. Suffix is the
`kernel_id` rather than a counter, so an unrelated kernel entering the worklist
does not renumber everything.

### Correction to my own first proposal, recorded because it would have wasted a run

I first reported the fix as *"one sentence in the identify instruction."*
**That would have done nothing** — the name is minted in code, and no instruction
reaches it. **I proposed an instruction fix for a code defect**, which is the
mirror of the run-3 preflight case where the fix genuinely was the instruction.
The question to ask first is *where is this value actually produced*, and I
answered it second.
