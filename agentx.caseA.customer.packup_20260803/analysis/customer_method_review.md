# Review of the customer's method and deployment recipe

The task brief asks: *"客户的部署方法如果不一样,请分析哪些地方值得学习,交由用户决策"*.
This file separates the **bench method** (which we ran, and which is good) from
the **deployment recipe** shipped alongside it (which we did not run, and which
is a different stack solving different constraints).

Source: ROCm/MAD PR #173, local branch `pr173`, dirs `scripts/AgentX_CaseA/` and
`scripts/GLM-5.2-disagg/`.

---

## Part 1 — the bench method

### What it is

A **deterministic, open-loop, frozen trace** plus a generic aiperf replay driver:

- `gen_caseA_conformance.py` — seeded synthesizer (seed 42 → byte-identical corpus)
- `verify_caseA.py` — 13-axis conformance checker
- `caseA_conformance_corpus.tar.gz` — the frozen 200-session / 1,778-request trace
- `replay_caseA.sh` — concurrency sweep against any OpenAI-compatible `URL`

We re-ran `verify_caseA.py` against the frozen corpus on two machines:
**13/13 PASS both times.** The artifact is what it claims to be.

### Worth adopting: three things

**1. Freeze the demand.** This is the substantive methodological advance over our
own bench and the reason to take it seriously. par8's own analysis concedes its
headline TTFT is confounded — two variables moved together and "this data cannot
apportion it". An open-loop frozen trace removes that confound *structurally*:
offered demand is a file. Comparing two deployments becomes a one-variable
experiment.

**2. Clamp the context jointly at trace-construction time.** The customer's
trace has `max(in + out) = 258,303 < 262,144` by construction, so **zero**
requests can be rejected for context overflow. Our workload samples input and
output independently with no joint clamp, which is the entire source of par8's
0.52 % error rate. This is a small fix to our own sampler and it costs nothing.

**3. Tag every record with `turn_index`.** aiperf does; our driver does not. It
is what let this kit price the prefix cache directly — first-turn TTFT 8,981 ms
vs cached-turn 3,568 ms **at matched input size**. par8 could not produce that
table from its own data.

### Defects found — both worth reporting upstream

**Defect 1 — the README misstates the cache metric it reports.**

The Case-A README says:

> Reported cache-hit is the endpoint's realized server-side prefix hit

The code does not do this. `aiperf/metrics/theoretical_prefix_cache.py:22-30`
computes `Theoretical Prefix Cache Hit` from the **loader's own walk of the
trace's `hash_ids`** — an infinite-cache upper bound read off the input file. It
never queries the server. The metric is honestly *named* "Theoretical"; the
README's prose contradicts the name.

Consequence for anyone using this bench: the reported cache figure is a property
of the corpus (always ~88 %) and is **invariant to the deployment under test**.
A server with prefix caching entirely disabled would report the same 88 %. The
real server-side number must be computed from
`usage_prompt_cache_read_tokens` in the raw records, which is what this kit did.

**Defect 2 — `replay_caseA.sh` writes results into the container namespace.**

`run_aiperf()` mounts only `$HERE`, `/models`, and `/shared_nfs`, but
`--output-artifact-dir` is `$OUT/c$C/art` where `OUT` defaults to `$HERE/results`.
That default happens to work. **Any `OUT` outside `$HERE` silently writes inside
the container and is destroyed on exit** — the sweep then reports `FAILED` for
every point because the summary parser cannot find
`profile_export_aiperf.csv`, even though the run itself succeeded.

We hit exactly this. The run was fine (231 and 323 requests, 0 errors); only the
script's CSV row was lost. We recovered the artifacts with a `docker cp` polling
loop (`../scripts/rescue_artifacts.sh`) rather than modify the customer's script.
A one-line fix upstream would be to mount `$OUT` as well.

### Two constraints to know before running it

- **`min_benchmark_duration_seconds = 900`** is enforced by the scenario
  (`inferencex_agentx_mvp.py:33`). The script's own default `DUR=300` would be
  rejected by the validator. Their README documents `DUR` but not this floor.
- **conc = 1 is unsupported**, and additionally **any session with < 2 turns is
  silently skipped** (`trajectory_source.py:423`) because the warmup/profile
  split needs at least one turn on each side. Their README documents the first
  but not the second — we hit the second while building a 1-session smoke corpus
  and spent a stalled window on it (`sent=0` for 690 s, no error raised).

## Part 2 — the deployment recipe (`scripts/GLM-5.2-disagg/`)

**We did not run this**, per the operator's decision to bench our existing
deployment. Read from source only.

### It is a different stack solving different constraints

| | customer recipe | ours (under test) |
|---|---|---|
| engine | **ATOM native** (`atom.entrypoints.openai_server`) | sglang (`infera/engine-sglang:merged-e`) |
| router | **atomesh**, `--policy random` | infera Rust router, kv-aware policy |
| KV transport | mooncake **TCP** (`MC_FORCE_TCP=1`) | mooncake **RoCE/RDMA** over ionic |
| prefix caching | **`--no-enable_prefix_caching`** — off | **on**, 88.9 % measured |
| topology | TP8 prefill + TP8 decode | TP8 prefill (DPA off) + TP8/DPA8 decode |
| spec decode | not configured | MTP, 4 draft tokens |
| KV dtype | `fp8` | default |

### Their two "non-obvious settings" do not transfer to us

1. **`online_quant_config` excludes `*expert*`** — an ATOM MXFP4 re-quantization
   deadlock. We do not run ATOM.
2. **`MC_FORCE_TCP=1`** — their nodes lack `amdgpu_peermem`, so GPU-direct RDMA
   registration is unavailable and mooncake must fall back to TCP. **Our nodes do
   not have this problem**; par8 completed a 4,005 s run over RoCE. Adopting
   `MC_FORCE_TCP` here would be a downgrade.

### The comparison that would be interesting, and is not made anywhere

Their recipe runs with **prefix caching off** and a **random** routing policy.
Ours runs with prefix caching on. On a Case-A workload whose defining property is
**88 % prefix reuse**, that is not a small difference — and this kit measured what
it is worth on our stack: **2.5× on TTFT** at matched input size.

**Whether their ATOM stack would show the same, more, or less is not
established** — different engine, different KV dtype, different transport. The
discriminating run is the customer's own trace against their own recipe with
`--no-enable_prefix_caching` toggled. That is one env change in their
`serve_atom.sh` and it is theirs to run, not ours.

## Recommendation to the operator

| item | recommendation |
|---|---|
| the frozen-trace method | **adopt as a second bench**, not a replacement — see `vs_infera_bench.md` |
| joint in+out context clamp | **adopt into our sampler** — removes our 0.52 % error class at no cost |
| `turn_index` in our metrics stream | **adopt** — cheap, and it unlocks the cache-value table |
| their README cache-metric claim | **report upstream** — documentation defect, could mislead a reader into thinking the number reflects their server |
| their `OUT`-outside-`$HERE` bug | **report upstream** — one-line mount fix |
| `MC_FORCE_TCP` / ATOM quant workarounds | **do not adopt** — they solve constraints we do not have |
| their `--no-enable_prefix_caching` default | **flag to them** — on a Case-A workload this is leaving 2.5× TTFT on the table, at least on our stack |
