---
name: infera-experiment
description: Run one hypothesis as a one-variable A/B on Infera — prove the lever is live, measure both arms alike with formal AgentX runs one after another on one node, judge the AgentX numbers with scripts/ab_judge.py (the base arm must reproduce the baseline series; noise never becomes an accept; an accept needs the candidate's GSM8K), and record a verdict the acceptance check can recompute
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
4. **Measure both arms alike** (`infera-bench`): formal AgentX on both (`DURATION` at least `ledger.min_duration_s`,
   3600, not fast mode), same load, slice, seed and client version; both fresh or both warmed; one after another on
   the same node, never two at once on it, and with no more than `ledger.max_arm_gap_min` (60) minutes between one
   measurement and the next (redeploying and warming up take 20-30 min); for a router-only change, A/B on one
   deployment and repeat in reverse order. The base arm has to reproduce the baseline series: its mean within
   `ledger.reproduce_pct` (5%) of the series' point at that concurrency as the task that sealed the version measured
   it; if it does not, the verdict is rerun (a quieter node or hour may reproduce it). Convert every point with
   `ledger agentx` (never write a result by hand) and leave its run directory where it is, AIPerf's export
   (`aiperf_artifacts/`) included: the window, the node and the duration are read from it, and the acceptance check
   converts it again. Record the base arm as a repeat of the baseline series (`--series <ledger.baseline_series>
   --role baseline --label r1-spf-base`, `-base2` for the next) and the variant as the experiment's own series
   (`--series r1-spf`). Screening points (AgentX fast or shorter runs, a microbench, dstep) help you choose and
   explain; a fast or short AgentX point converted into `results/` is marked screening in the ledger, needs no
   decision and is never an input of the verdict. Each input is a run of its own: one raw output converted under two
   labels is not two measurements (ab_judge refuses it and `ledger verify` fails). Run GSM8K on the candidate stack (`infera-quality-gate`: at least 200 questions, lm_eval's output under
   this task's workspace, `ledger gsm8k --series r1-spf`) before you judge: an accept needs it, and every GSM8K in
   `results/` counts.
5. **Judge** with the script on the AgentX numbers alone (rules in its docstring: the project's noise band and accept
   bar, `ledger.noise_pct` and `accept_pct` in project.yaml, 2% and 3% by default; X not worse than its noise; the
   medians agreeing with the means; and the gates: formal inputs, the base arm reproducing, the timing, the noise
   floor raised to the version's retest noise, GSM8K). `--write` puts the verdict, with its inputs and gates, into
   every candidate file. Both arms are at one concurrency, the base arm in the baseline series; give it every valid
   formal result at that concurrency in `results/` (one A/B per concurrency per task): leaving one out fails the
   check, whatever series or role it says.

   ```bash
   "$AGENT_SYS_PYTHON" "$AGENT_SYS_SKILLS/infera-experiment/scripts/ab_judge.py" \
     --baseline-files "$AGENT_SYS_WORKSPACE"/results/r1-spf-base-c64.json "$AGENT_SYS_WORKSPACE"/results/r1-spf-base2-c64.json \
     --candidate-files "$AGENT_SYS_WORKSPACE"/results/r1-spf-c64.json "$AGENT_SYS_WORKSPACE"/results/r1-spf-2-c64.json \
     --mechanism "prefill queue -80%, step time unchanged" --write
   ```

   The metric, X, their directions and the noise band come from project.yaml; `--accept-pct` can only raise the bar,
   `--reproduce-pct` and `--max-arm-gap-min` only tighten theirs, and project.yaml's rules are the user's to change.
   `--mechanism` is a remark kept with the verdict: the mechanism explains a result, it does not decide it. `rerun`
   means: run another pair before deciding (the notes say why: the base arm off its series, arms overlapping or far
   apart, a gain between the floor and the bar); with X worse, compare at equal X (the ledger's working point Δ). A
   change whose expected gain is below the noise is judged noise: `--note 待集成` leaves it to the round's close.
   When the noise floor itself is above the project's accept bar (the version's retest noise is; a bar raised with
   `--accept-pct` does not change this), the measurement cannot tell a
   change of the bar's size: a result inside it is noise with `resolvable: false`, noted `· 待集成` (the ledger shows
   判不了 · 待集成); another pair would not lower that floor, so do not rerun it. A significantly worse GSM8K turns
   an accept into a reject; without the candidate's GSM8K the script writes no accept. Numbers typed in
   (`--baseline 12010 --candidate 12650`) are a trial: printed, never written.
6. **Void** a spoiled measurement instead of leaving it out; there are two kinds:
   `--void failed-requests --candidate-files <file> --write` when AgentX counts more than 10% failed requests (the
   script computes it and refuses otherwise), and `--void other --reason "<what spoiled it>" --evidence <file>` for
   anything else (foreign load on the GPUs, HSA out-of-resources, preemption: the evidence is the log that shows it).
   `--baseline-files` voids a base arm. A void result stays in the ledger, marked, and is out of the A/B; judge again
   without it.
7. **Check** before you report: `"$AGENT_SYS_PYTHON" -m agent_sys_v2.ledger verify "$AGENT_SYS_WORKSPACE/results"`
   (the experiment's acceptance check). It converts the raw output of every input again (AIPerf's export and
   runtime.env included), applies the gates and judges again, and counts an accept's GSM8K again from lm_eval's
   samples; a verdict that does not match the data, or that cannot be recomputed (no AgentX source, a raw file gone,
   no GSM8K for an accept), fails with what to fix. A gate that cannot be checked (an old result without its window,
   no reference point at that concurrency) is written unchecked in the notes, and does not fail. Never edit a result
   or a verdict by hand: judge again.

## Report

Start with the decision and the AgentX number (`+15.4% tok/s/GPU at C64, accept`), then the mechanism evidence
that explains it (`[M]` / `[I]`), the exact change (diff, flag or env), the run directories, and what the next experiment should
be. A rejected or noise result is a finding too: say what it rules out.
