---
name: infera-profile
description: Profile a measured deployment cheap layer first (time account, resource use, then kernels or CPU of the bottleneck layer only) and hand the plan a ranked candidates.yaml; standard by default, ultracode only for onboarding an engine or a deep dive you approve
---

# infera-profile

The profile task of a round: from the baseline's runs to a ranked, machine-readable list of candidates the plan
drafts experiments from. In the ATOM GLM-5.2 project almost every conclusion that later paid off came from the cheap
layers (a read-only pass over existing data, a 5 s decode trace, py-spy); full-window instrumentation cost the most
and was not reused. So: cheapest first, and only the bottleneck layer gets the expensive tools.

## Modes

| Mode | When | How | Rough cost |
|---|---|---|---|
| `standard` (default) | the engine has a playbook (`playbooks/<engine>.md`) | one agent through P0–P6 below, one verifying subagent for P6 | 0.5–1 day, 2–3 node·h, $50–150 |
| `onboarding` | first project on an engine × topology × major version, or its playbook no longer holds | ultracode (`agent: {ultracode: true}`), lean: 6–8 research/verify topics, one on-node validation; writes the playbook, the adapter (metric and log names) and the kernel taxonomy | ~1 day, $300–500 |
| `deep` | the time account cannot explain what is seen; cheap levers used up (two rounds < 2% together, gap still > 15%); a causal closed loop is needed for weeks of work | ultracode with pdtrace-style instrumentation | 1–2 days, $500–1000 |

The two ultracode modes need the user's approval first (task frontmatter `agent: {ultracode: true}` and a budget).
A standard run ends with "deep recommended: yes/no, because …" in its report.

## Playbooks (one per engine)

`playbooks/<engine>.md` (atom, sglang, vllm) holds what is engine-specific: the profiler control plane and its
traps, metric and log names (the adapter), the trace layout, the kernel taxonomy, known pitfalls, and the bench
commands (decode step bench, kernel breakdown). Read the one of the project's engine before P0; without one, the
task is an onboarding and its first output is the playbook. See `playbooks/README.md` for the sections.

## Standard flow

| Step | What | Output | Pass when |
|---|---|---|---|
| P0 inputs | the baseline's runs (AIPerf `profile_export.jsonl`, agentx JSON, engine logs with timestamps, effective config, image id), noise (≥ 2 runs of one config), the system samples | `inputs.md` | everything is there; else back to the baseline |
| P1 time account | per request: λ, R, TTFT/ITL percentiles, ISL/OSL, lanes C = λ(R/φ+Z), in-flight split into TTFT and decode; client-side stalls (chunk gaps ≥ 300 ms) over time | `time_account.json` | lane closure error < 5% |
| P2 resources | P/D GPU busy, decode step and batch, queues (prefill waiting, transfer, prealloc), KV use and prefix hits, P→D bytes vs NIC, CPU per process, power and clocks | `resources.csv`, bottleneck table | a verdict per resource: bottleneck / latency tax / capacity; nothing > 90% busy → the CPU / scheduling gap is hypothesis one |
| P3 kernels (bottleneck layer only) | decode: fixed-batch step bench at 3–4 B around the working point, one short trace each; prefill: offline small load, P only | `kernels/*.csv`, step composition | Σkernel/step ≥ 0.95 GPU-bound, < 0.85 a launch/CPU gap; categories sum to 90–101%, other < 10% |
| P4 CPU path (when P1/P2 point there) | py-spy 60 s on API server, scheduler, tokenizer, router; GC; /metrics scrape rate | `cpu/` | a root-cause stack for stalls ≥ 100 ms |
| P5 candidates | each: layer, evidence, ΔY/ΔX range (local derivative, at fixed C), confidence, cost, fast validation level (L0–L5) and metric, falsification | `candidates.yaml`, ranked table | ≥ 3 items, fields complete, ranked by expected gain / cost |
| P6 verify | recompute the top 10 numbers independently (a subagent); mark every claim `[M]` / `[I]` | `verify.md` | refuted numbers fixed in the report |

Guard rails: scoring windows carry zero-overhead collection only (traces go to unscored diagnostic runs or the
bench); create output directories on the host first; poll until a trace is on disk (stop returns early; a missing
directory still answers 200); never change scripts on NFS while a run uses them.

## Outputs

`report.md` (conclusion first: the bottleneck, latency taxes, capacity risk, the top 3 candidates with ranges, deep
or not), `profile.json` (`infera.profile/v1`: runs, window, noise, time account, resources, step model, kernel
breakdown, verdicts, deep_recommended), `verify.md`, and `candidates.yaml`:

```yaml
schema: infera.candidates/v1
baseline: {series: v0, conc: 64, y: 33176.9, x: 55.74, noise_pct: 0.2}
items:
  - id: c1
    title: API process gen-2 GC pauses
    layer: cpu_path            # decode_step | prefill | kv_transfer | capacity | cpu_path | config | routing
    bottleneck_class: latency_tax   # bottleneck | latency_tax | capacity
    evidence: "[M] gen-2 is 42.5% of the API wall time"
    expected: {delta_pct: [1.2, 1.8], dx_pct: [0.4, 1.3], confidence: C}
    cost: S                    # S | M | L
    fast_validation: {level: L3, metric: "gen-2 pause slope, client stalls ≥ 300 ms"}
    falsify: "gen-2 pause share drops < 50%"
```

The plan copies a candidate's `expected.delta_pct`, `fast_validation.level` and `falsify` into the `expect` of the
experiment it drafts (skill infera-round).
