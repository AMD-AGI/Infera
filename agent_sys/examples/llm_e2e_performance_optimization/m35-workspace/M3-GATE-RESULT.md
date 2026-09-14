# m3 gate result — the decision between the two m4 routes

> ## READ THIS FIRST — is this a RESULT or a TEMPLATE?
>
> This file is generated. **Check the `workset` line under Provenance:**
>
> - a path under a **run directory** (`.../runs/<id>/handoffs/...`) → this is a
>   **real result** from that run;
> - a path under **`known/`** → this is a **known-answer FIXTURE**, not a result.
>   It says what the tool *would* report. **No real workset was read.**
>
> **As of 2026-09-06 the gate had never fired on a real workset.** Six chain runs
> on this cluster; none reached m3. Every copy of this file produced that day is a
> fixture rendering. **Do not quote it as a finding about any operator.**

**Generated 2026-09-06T13:10:28Z** by `make_gate_result.sh`, repo HEAD `c022dcfd`.
Author: m35. **This is the artefact; any message about it is a summary of this.**

## Provenance

```
run dir     : /data/yihou/agent_sys_runroot/runs/20260906T130845-298750
workset     : known/module
orchestrator: 3308290
  how       : supplied explicitly
```

**The launch line, read from the process rather than from any document**
(a record says what was intended; the cmdline says what this run got):

```
adhoc_cases=3
aiperf_trace=/data/yihou/e2e_verify_20260906/m2/materials/conversation_trace.v2.jsonl
bench_rounds=3
container=yihou_e2e_chain
context_length=40960
dsa_args=none
eval_thinking=none
expect_ranks=4
forge_mock=1
gsm8k_data=/data/yihou/e2e_verify_20260906/m2/materials/gsm8k_test.jsonl
image=infera/engine-sglang:qwen3-local-20260906
instruction=Bring
jobid=29184
kernel_table_min_launchers=0
magpie_root=/data/yihou/Magpie
measure_gpu=4
mock_stages=none
model_name=Qwen/Qwen3-32B
model_path=/apps/data/models/Qwen3-32B
node=smci355-ccs-aus-n04-25
node_ip=10.235.192.131
parser_args=none
port_etcd=8103
port_router=8101
port_worker=8102
scratch_root=/data/yihou/e2e_flow/kfo
stack_window_s=0
tp=4
trace_end_ms=60000
transport=local
validate_work_root=/data/yihou/e2e_flow/validate
work_root=/data/yihou/e2e_flow
```

## The extractor, verbatim

```
workset: known/module/items/codes/workset.yaml
operators: 1

=== usable_op
  substitution   'module_symbol'
  public_symbol  'sampling_from_probs_torch'
  target_files   ['srt/layers/sampler.py']
  repo_root_var  '@SGLANG_ROOT@'
  apply_mode     'overlay_files'
  build_step     None
  entry_function 'Sampler.forward'
  inputs         ['logits', 'out']
  baseline       159 chars, def run(: True, public defs: ['Sampler', 'logger', 'run', 'sampling_from_probs_torch']

  ready to run:
    python3 /data/yihou/e2e_verify_20260906/m35/mk_reverse_payload.py \
      --image infera/engine-sglang:qwen3-local-20260906 \
      --container-path <SGLANG_ROOT>/srt/layers/sampler.py \
      --operator usable_op \
      --function Sampler.forward \
      --delegate-to sampling_from_probs_torch \
      --out /data/yihou/e2e_verify_20260906/m35/payload.usable_op
    # SGLANG_ROOT on this image = /sgl-workspace/sglang/python/sglang
    # then check that sampling_from_probs_torch's parameters accept ['logits', 'out'] —
    #   a mismatch surfaces as an EMPTY measurement, not an error

no stop conditions found.
exit status: 0   (0 = usable, 1 = STOP, 2 = bad path)
```

## The decision this forces

**Exit 0 — no stop condition.** The operator is `module_symbol`, it declares a
`public_symbol`, `target_files` and an `entry_function`, and no `build_step`.

**Read the `baseline` line above, because it decides the route:**

- **module-shaped** (defines the public symbol *and* `run`) → **m2's single line
  with `forge_mock=1` survives.** `30_run_forge.sh` seeds from that baseline and
  the overlay keeps the engine module's surface, so `apply.py:828` has nothing to
  refuse. **My two-form split is unnecessary and should be retired.**
- **harness-shaped** (defines `run` but not the public symbol) → **`forge_mock=1`
  produces an overlay that drops the module's whole public surface and
  `apply.py:828` refuses AFTER a bring-up.** The route is the reverse payload
  (`mk_reverse_payload.py --delegate-to <public_symbol>`), which is the only
  payload shape that has ever passed `apply` anywhere.

## What this does NOT establish

- **It is a read of a document, not a run of `apply`.** It predicts what
  `apply.py:828` will do; it does not exercise it.
- **The delegation signature stays open.** The delegation is `run(*args, **kwargs)`
  and restates nothing, but the public symbol's parameters must accept the
  Definition's `inputs` keys. **A mismatch surfaces as an EMPTY MEASUREMENT from
  `check_speedup_substantiated`, not as a signature error** — pre-registered as
  "the delegation did not match the Definition's inputs", not a producer defect.
- **A weak or absent `edit_target.entry_function` is explanation 1, and we now
  know exactly which resolution level went missing.** `stack_window_s=0` means no
  launcher blocks, and a launcher block is what `identify` resolution **level 1**
  (`trace_python_stack`, `identify.py:9`) reads. So level 1 is unavailable to this
  run and resolution falls to **level 2, `kernel_finder`, which is what
  `magpie_root` feeds**. On the first cluster level 1 always succeeded
  (`m3_analysis.yaml:122-126`: all five operators at `resolve_ratio: 1.0`), **so the
  Magpie fallback was never exercised there — and Magpie has never completed a real
  scan on this cluster.** `min_resolve_ratio` defaults to `0.0`, so
  `check_identity_resolved` **will not refuse for resolving nothing**: an
  `operator_identity` is produced either way.
  **Measured, and over the whole validator set rather than a sample:** none of
  `check_worklist_shape`, `check_identity_resolved`, `check_workset_shape`,
  `check_workset_runs` or `check_environment` mentions
  `launcher`/`stacks_manifest`/`stack_window` — 0 hits across 15 files. **So the
  wall does NOT move to m3 validation; it moves m3 onto an untested route.**
  (Question raised by the checkpoint writer; mechanism by the leader; file set
  widened and re-measured here.)
- **m4 cannot complete in a short hold.** `check_workset_runs` is `cost: gpu_hours`
  and `check_speedup_substantiated` has a **60-minute hard ceiling** (two entrypoint
  runs at `timeout_seconds: 1800`). See RUNG5-CHECKLIST P10.
