---
name: infera-experiment
description: Run one hypothesis as a one-variable A/B on Infera — prove the lever is live, measure both arms alike with AgentX, judge the AgentX numbers with scripts/ab_judge.py (noise never becomes an accept), and record a verdict the acceptance check can recompute
---

# infera-experiment

## Protocol

1. **Write the change down first**: which lever, from what to what, the mechanism you expect and its size, and
   which mechanism metrics should move (queue triple, step time, busiest-rank running count, hit rates, transfer
   queue ...). One variable per experiment.
2. **Dry run**: bring the candidate up and diff its live configuration against the baseline's
   (`infera-deploy`, `effective-config.json`). Only the intended difference may appear.
3. **Prove the lever is live** before measuring: the flag on the live command line or in `/get_server_info`, the
   patched file's hash inside the container plus a runtime marker for a code change. A green static check that
   the instrument has never been seen to fail proves nothing: when in doubt, show it red on a known-bad control.
4. **Measure both arms alike** (`infera-bench`): AgentX on both, same load, slice, seed and client version; both
   fresh or both warmed; for a router-only change, A/B on one deployment and repeat in reverse order. Gate the
   candidate (`infera-quality-gate`). Convert every point with `ledger agentx` (never write a result by hand) and leave
   its run directory where it is: the acceptance check converts it again. Record the base arm as a repeat of the
   baseline series (`--series <ledger.baseline_series> --role baseline --label r1-spf-base`, `-base2` for the next)
   and the variant as the experiment's own series (`--series r1-spf`). Screening points (AgentX fast, a microbench,
   dstep) help you choose and explain; they are no input of the verdict and stay out of the top of `results/`
   (`results/screen/` or `log/`).
5. **Judge** with the script on the AgentX numbers alone (rules in its docstring: the project's noise band and accept
   bar, `ledger.noise_pct` and `accept_pct` in project.yaml, 2% and 3% by default; X not worse than its noise; the
   medians agreeing with the means). `--write` puts the verdict, with its inputs, into every candidate file. Both arms
   are at one concurrency, the base arm in the baseline series; give it every valid result at that concurrency in
   `results/` (one A/B per concurrency per task): leaving one out fails the check, whatever series or role it says.

   ```bash
   "$AGENT_SYS_PYTHON" "$AGENT_SYS_SKILLS/infera-experiment/scripts/ab_judge.py" \
     --baseline-files "$AGENT_SYS_WORKSPACE"/results/r1-spf-base-c64.json "$AGENT_SYS_WORKSPACE"/results/r1-spf-base2-c64.json \
     --candidate-files "$AGENT_SYS_WORKSPACE"/results/r1-spf-c64.json "$AGENT_SYS_WORKSPACE"/results/r1-spf-2-c64.json \
     --mechanism "prefill queue -80%, step time unchanged" --write
   ```

   The metric, X, their directions and the noise band come from project.yaml; `--accept-pct` can only raise the bar,
   and project.yaml's rules are the user's to change. `--mechanism` is a remark kept with the verdict: the
   mechanism explains a result, it does not decide it. `rerun` means: run another pair before deciding; with X worse,
   compare at equal X (the ledger's working point Δ). A change whose expected gain is below the noise is judged noise:
   `--note 待集成` leaves it to the round's close. Numbers typed in (`--baseline 12010 --candidate 12650`) are a trial:
   printed, never written.
6. **Void** a spoiled measurement instead of leaving it out; there are two kinds:
   `--void failed-requests --candidate-files <file> --write` when AgentX counts more than 10% failed requests (the
   script computes it and refuses otherwise), and `--void other --reason "<what spoiled it>" --evidence <file>` for
   anything else (foreign load on the GPUs, HSA out-of-resources, preemption: the evidence is the log that shows it).
   `--baseline-files` voids a base arm. A void result stays in the ledger, marked, and is out of the A/B; judge again
   without it.
7. **Check** before you report: `"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger verify "$AGENT_SYS_WORKSPACE/results"`
   (the experiment's acceptance check). It converts the raw output of every input again and judges again; a verdict
   that does not match the data, or that cannot be recomputed (no AgentX source, a raw file gone), fails with what to
   fix. Never edit a result or a verdict by hand: judge again.

## Report

Start with the decision and the AgentX number (`+15.4% tok/s/GPU at C64, accept`), then the mechanism evidence
that explains it (`[M]` / `[I]`), the exact change (diff, flag or env), the run directories, and what the next experiment should
be. A rejected or noise result is a finding too: say what it rules out.
