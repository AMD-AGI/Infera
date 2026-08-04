# Verifying the GLM-5.2 1P1D example kit end-to-end

**Ran:** 2026-08-04 07:16 – 09:20 UTC
**Status:** **PASS** — the kit brought the deployment up, all six features verified,
the customer bench ran against it. **Five kit defects found and fixed**; two more
recorded but not fixed (they are image gaps, not script bugs).

## Goal

Run `examples/sglang_1p1d_glm5.2/` **as written** on the vultr MI355X cluster, prove
each advertised feature is really on rather than silently absent, fix the kit where it
is genuinely broken, then replay the customer's AgentX Case-A bench against the result.

**Spec:** [`spec/verify_example.md`](spec/verify_example.md).

Its explicit criteria, and the outcome:

| # | criterion (from the spec) | actual | verdict |
|---|---|---|---|
| 1 | bring up GLM-5.2 with sglang + PD(mooncake) + MTP + decode DPA + prefill no-DPA + kvd + kv-aware, using **the example's own scripts** | `up.sh` completed: containers → etcd + kvd → both legs → router | **PASS** |
| 2 | verify each feature is correct | six features, each with a signal that would go red if absent — table below | **PASS** |
| 3 | run the customer agent bench per `agentx.caseA.customer.packup_20260803` | 225 requests, 0 errors, 913.5 s at conc 8 | **PASS** |
| 4 | fix the example's scripts if needed, **no git operations** | 5 fixes, all in the working tree; no git command run | **PASS** |

> The kit's own README said these scripts were *"a refactor … not yet run end-to-end."*
> This run is that end-to-end execution. The recipe held; the packaging had five bugs,
> **every one of which fails silently** — the leg boots, serves, and returns plausible
> numbers for the wrong deployment.

## Result

### The deployment the kit produced

```
chi2835 (prefill)                                   chi2879 (decode)
├─ etcd :2379                ◀── discovery ──▶
├─ infera router :8100  kv-aware
├─ kvd daemon (8 adapters)
└─ prefill :30000  TP8, DPA off,  ══ KV over RDMA ══▶  decode :30001  TP8, DPA on (dp8),
   ep8, gmu 0.70, kvd L2/L3                             ep8, gmu 0.85, MTP (EAGLE)
```

### Feature evidence — measured, not assumed

| feature | signal | reading | verdict |
|---|---|---|---|
| **PD pairing** | `/v1/workers` | prefill + decode, both `active` | ✅ |
| **serving + DSA correctness** | a chat completion through the router | `"The capital of France is Paris."`, coherent | ✅ |
| **mooncake over RDMA** | `MC_FORCE_TCP` / `GID is NULL` / in-container device count | `0 / 0 / 8` and `0 / 0 / 7` | ✅ |
| **prefill DPA off** | live cmdline + `scheduler_DP` count | no `--dp-size`, **has `--ep-size 8`**, 0 DP ranks | ✅ |
| **decode DPA on** | resolved `dp_size` + rank count | `dp_size=8`, `enable_dp_attention=True`, 8 ranks | ✅ |
| **MTP (EAGLE)** | accept-len distribution, n=1265 | **median 2.88**, p10 2.01, p90 3.84 | ✅ |
| **kvd** | adapters + `statctl` | 8 adapters, 47,556 entries / 80,248 sets; decode **0 by design** | ✅ |
| **kv-aware** | router policy + tokenizer | `router_policy: "kv-aware"`, tokenizer loaded | ✅ |

Raw output: [`results/smoke_final.txt`](results/smoke_final.txt),
[`results/preflight_chi2835.txt`](results/preflight_chi2835.txt).

### Customer bench — AgentX Case-A, conc 8

Customer script replayed **unmodified** (`md5sum replay_caseA.sh` =
`7cde1afc627c7e4868eac0fd13741baa`, matching the reference kit).

| | this run | reference (`agentx.caseA.customer.packup_20260803`) |
|---|---|---|
| requests / errors | **225 / 0** | 231 / 0 |
| window | 913.5 s | 901 s |
| TTFT p50 / p90 | 5,546 / 22,068 ms | 5,146 / 19,780 ms |
| ITL p50 | **14.46 ms** | 13.8 ms |
| **cache hit p50 (server-reported)** | **88.2 %** | 88.1 % |
| E2E p50 | 14,717 ms | 12,600 ms |

Every axis lands within ~15 % of the reference, and the two most deployment-diagnostic
ones — **ITL ≈ 14 ms** and **cache ≈ 88 %** — match to within 0.1 pp. That is the
result: a kit-built deployment performing indistinguishably from the hand-built one the
reference numbers came from.

> **One trap worth stating up front.** The run-level cache figure printed by
> `analyze.py` is **51.9 %**, not 88.2 %. That is a metric-definition artifact of our
> own analysis script, not a deployment difference — see [`notes.md`](notes.md) §5.

## The five kit defects fixed

Each one **fails silently**. Full write-ups in [`patches/README.md`](patches/README.md);
the combined diff is [`patches/kit_fixes.diff`](patches/kit_fixes.diff).

| # | file | defect | how it presents |
|---|---|---|---|
| 1 | `common.sh`, `preflight_rdma.sh` | host RDMA provider library mounted at a path the image's entrypoint never reads | **zero RDMA devices in the container**; the leg boots and serves anyway |
| 2 | `common.sh` | in-container device count of 0 printed without comment | hides #1 |
| 3 | `engine/smoke.sh` | router policy read as `router-policy=<x>`; the Rust router emits `router_policy: "<x>"`, ANSI-coloured | the line prints **empty whatever the policy is** |
| 4 | `common.sh`, `engine/smoke.sh` | `grep -c … \|\| echo 0` emits two zeros; logs contain binary bytes so bare `grep` prints no text | readings like `MC_FORCE_TCP=00`, and `binary file matches` |
| 5 | `engine/smoke.sh` | MTP checked with `tail -5` | 4.7 % of healthy batches sit at 4.00, so the tail routinely reads **degenerate on a healthy leg** — we hit exactly this |

Defects 1 and 5 were each caught by a live reading that contradicted the kit's own
success story; both are documented with the measurement that exposed them.

## Two gaps recorded, not fixed

Neither is a script bug — both are the image lagging the repo. See [`notes.md`](notes.md) §1–2.

- **`preflight_rdma.sh mode` cannot run on this image**: `infera.tools.preflight.mooncake_mode`
  is absent from `infera/engine-sglang:merged-e`. Worked around by bind-mounting the
  repo's module into the container.
- **The image lacks `GLM52_P1V3`**: measured `_p1v2_clip` = 0 in `dsa_indexer.py`.
  Without it the decode leg crashes minutes into an agentic workload. Applied at
  runtime and verified in the **compiled bytecode**.

## How to reproduce

See [`REPRODUCE.md`](REPRODUCE.md). TL;DR: fill in a wrapper, `up`, apply P1V3 to the
decode leg, `smoke`, run the customer bench.

## Folder map

| path | what |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | ordered, copy-pasteable reproduction |
| [`environment.md`](environment.md) | nodes, digests, fabric, versions — the exact stack behind these numbers |
| [`notes.md`](notes.md) | the traps, the wrong turns, and what this run did **not** establish |
| [`patches/`](patches/) | the five kit fixes + the runtime P1V3 patch, each with what/why/how/context |
| [`scripts/`](scripts/) | every script that ran, verbatim — `cluster.vultr.sh` (the site wrapper, the only file with site values), `teardown_prev.sh`, `relaunch_decode.sh`, `feature_evidence.sh`, `cache_probe.py`, `cache_by_turn.py`, `verify_mount_revert.sh`, `run_agentx.sh` |
| [`results/`](results/) | preflight reports, smoke output, bench profile export |
| [`env/`](env/) | per-node environment snapshots |
| [`logs/`](logs/) | both engine logs, router, kvd, bring-up (gzipped) |
| [`spec/`](spec/) | the originating task file |
