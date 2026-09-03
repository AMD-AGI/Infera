# GLM-5.3-Flash-MXFP4 on infera + SGLang — MIX bring-up, MI355X / gfx950

**Ran:** 2026-09-01 → 2026-09-02
**Author:** yihou (AMD)
**Status:** **PASS** — GLM-5.3-Flash-MXFP4 serves through the full infera MIX
stack (infera engine wrapper + etcd + kv-aware router) on 4×MI355X, TP4, decode
CUDA graphs on. Reproduced independently on **two different nodes**.

## Goal

`mission.md` (copied in as `spec.mission.md`) step 4: *"跑通 glm5.3 flash mxfp4
版本 pd-mix"*, and it is the mission's stated first priority
(*"优先完成 glm5.3 flash mxfp4 版本"*). Deliver GLM-5.3-Flash-MXFP4 serving
through infera + SGLang in MIX (aggregated, non-PD) mode on this 8×MI355X host,
and take a fixed-length throughput reading against the GLM-5.2 baseline.

**Success criterion** (mission step 4 + the alignment bar in step 3.1):
the deployment serves through infera + sglang, and fixlen performance is
*"大致与 glm5.2 对齐"*.

## Result

| # | Criterion | Outcome |
|---|---|---|
| 1 | GLM-5.3-Flash-MXFP4 loads at all | **PASS** — after `--disable-shared-experts-fusion`; without it the loader dies in `_load_w2` |
| 2 | Serves through infera MIX (wrapper + etcd + kv-aware router) | **PASS** — 1 worker `disagg_mode: mixed`, router `kv-aware`, coherent answers (`results/router_state_n0133.txt`, `results/ladder.md`) |
| 3 | AITER fast path live | **PASS** — 8 mHC lines (2 per rank × 4) |
| 4 | Both memory pools present (paged KV + KDA) | **PASS** — 4818 decode lines carrying `full token usage` **and** `mamba usage` in `logs/n0133/worker_glm53_mix.log.gz` |
| 5 | Decode CUDA graphs on | **PASS** — decode lines report `cuda graph: True` |
| 6 | fixlen roughly in line with GLM-5.2 | **PASS**, but read the three caveats in `results/fixlen_vs_glm52.md` before quoting any ratio |

fixlen p50 (isl 7400 / osl 320), TP4, graphs on, node `smci355-ccs-aus-n01-33`:

| conc | 1 | 8 | 16 | 24 |
|---|---:|---:|---:|---:|
| output tok/s | 111.02 | 561.04 | 962.55 | 1391.15 |
| TTFT p50 (ms) | 255 | 1065 | 744 | 619 |

fixlen p90 (isl 15500 / osl 3300), same node and configuration — **no GLM-5.2
baseline exists for this shape**, so it is reported and not compared:

| conc | 1 | 8 | 16 | 24 |
|---|---:|---:|---:|---:|
| output tok/s | 123.06 | 837.26 | 1509.46 | 2144.90 |
| TTFT p50 (ms) | 323 | 1237 | 688 | 1219 |

**Scope.** This packup covers **`flash-mxfp4` only**. The repo now also ships an
example kit for all four GLM-5.3 checkpoints and four CI-disabled e2e rows;
`big-fp8` and `big-mxfp4` were validated by the big-model track, `flash-fp8`
carries a recipe that is **not validated**, and **PD is not covered at all**.
The full table is in `repo-changes/README.md`.

## The centrepiece: why it did not load, and the one-flag fix

sglang PR **#36607** commit `bd1cc98b` *"[AMD] Enable GLM shared-expert fusion on
gfx95"* opened the gfx950 branch of
`Glm5NextForConditionalGeneration.shared_experts_fusion_disable_reason`
(`glm5_next.py:1414`) **without carrying the
`quant_blocks_shared_experts_fusion(quant_config)` guard** that its two sibling
families already have (`deepseek_v2.py:3069`, `deepseek_v4.py:3289`).
`QuarkConfig.can_fuse_shared_expert()` (`quark.py:1001`) computes the correct
answer — `False` for this checkpoint — and is never consulted.

The gate then says "fuse", the loader renames the **BF16** shared expert into
routed slot 288 (`glm5_next.py:1735`) of an **MXFP4-packed** `FusedMoE`, and the
copy fails. First-hand tensor-level evidence, from an instrumented run
(`logs/n0433/rung0_wtrace.log.gz`):

```
[WTRACE] layer_id=10 prefix=None wname='model.layers.10.mlp.experts.w2_weight' \
         shard=w2 eid=288 param=(289, 4096, 256) loaded=(4096, 2048) qm=QuarkFusedMoEMethod
RuntimeError: The size of tensor a (256) must match the size of tensor b (512)
              at non-singleton dimension 1
```

288 routed experts means ids 0..287, so `eid=288` **is** the shared expert, and
`param` dim0 = 289 is the routed buffer grown by one to hold it.
`moe_intermediate_size` 2048 → TP4-shards to 512 → MXFP4 packs two values per
byte → 256, against a BF16 `down_proj` that shards to 512. 2:1, exactly.

**The log line that discriminates** is `Shared experts fusion optimization
enabled.` — present in the failing run, absent in every passing one.

**Fix: `--disable-shared-experts-fusion`.** An engine-side patch of equivalent
effect is in `patches/` (not required to reproduce; see that file's header).

## Folder map

- `REPRODUCE.md` — ordered, copy-pasteable reproduction from zero
- `environment.md` — the two hosts, image digests, pinned SHAs, RDMA fabric
- `results/root_cause.md` — **the root cause in full**: evidence trail, the code
  defect, upstream status, the same-class issues elsewhere, and what is still
  unknown
- `notes.md` — method, the two corrections worth recording, environment traps,
  load-bearing flags, and how to read the GLM-5.2 comparison
- `scripts/` — the scripts that actually ran, verbatim
- `patches/` — the upstream-style engine fix + its anchor-verify fixtures
- `results/` — ladder evidence, fixlen CSVs, router state, the GLM-5.2 comparison
- `research/` — the vendor model card, the aiter image-delta probe, `p1_test.py`
- `repo-changes/` — the infera repo changes this run needed, and the commits they landed in
- `logs/` — gzipped worker/build/bench logs, split by node
- `PLAN.md` — the plan, including the rung ladder that isolated the cause
- `spec.mission.md` — the originating spec
