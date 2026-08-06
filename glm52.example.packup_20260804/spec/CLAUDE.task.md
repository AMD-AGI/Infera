# Task — ship a deliverable GLM-5.2 PD example under `examples/`

Turn the GLM-5.2 deployment experience that currently lives in experiment packups
into **one deliverable example** in this repo: `examples/sglang_1p1d_glm5.2/`.
SGLang + mooncake **1P1D + DP-attention + MTP**, one-/few-key scripts, plus a
`results/` folder carrying the conc=8 numbers from both benchmarks.

Spec: [`mission.glm5.2.example.md`](mission.glm5.2.example.md) (repo root).
Plan: `/home/yihou/.claude/plans/misty-sleeping-platypus.md`.

## Working directories

| what | where |
|---|---|
| deliverable | `examples/sglang_1p1d_glm5.2/` |
| local workspace (all temp work) | `work.glm52.example/` |

Nothing outside those two gets written. `work.glm52.example/` is scratch and is
not part of the deliverable.

## Hard constraints

1. **No local information in the deliverable.** No node names, no scheduler job
   ids, no `/mnt/vast` or `/shared_nfs` paths, no jump-host IPs, no branch names,
   no adhoc image tags (`infera/engine-sglang:merged-*`). Where a trap is worth
   warning about, keep the **mechanism** and drop the scene — as a script comment
   or a README "Notes & gotchas" entry.
2. **No process information.** The example is a recipe, not a lab notebook.
   Wrong turns, retries and debugging history stay out.
3. **Image is a placeholder**: `inferaimage/infera-sglang:0.2.0`, flagged in the
   README as pending the real release tag. Never document how an experiment image
   was built.
4. **No bench client launcher.** Service self-check + sglang's own
   `bench_serving` only. The customer's AgentX kit is referenced by URL from
   `results/` and the README, never vendored or wrapped.
5. **Only these packups may be consulted** (operator instruction):
   - `../infera.merge.liying.kv.mtp/agentx.caseA.customer.packup_20260803`
   - `../infera.merge.liying.kv.mtp/fixlen.glm52.fullfeature.packup_20260801`
   - `../infera.merge.liying.kv.mtp/par8.glm52.dpaoff.packup_20260803`
   - `../infera.merge.liying.kv.mtp/agenticbench.mtp.caseA.packup_20260801`
   - `../infera.merge.liying.kv.mtp/par8.armA.dpaon.roundrobin.spur.packup_20260804`
   - `../infera.merge.liying.kv.mtp/par8.armB.dpaoff.kvaware.spur.packup_20260804`

   Plus the customer's own kit at `/home/yihou/dev/git/MAD` (branch `pr173`) and
   the live spur artifacts under `/shared_nfs/yihou_agentx_caseA/`.
6. English in all work products; Chinese only when reporting to the operator.

## Decisions already made

| question | decision |
|---|---|
| directory shape | mirror `examples/deepseek_v4` — `common.sh` + `engine/` subdir, not flat numbered scripts |
| transport | **two wrappers, no autodetection**: peer-mem multi-NIC vs dma-buf single-ODP-NIC |
| customer bench | referenced only; no runnable aiperf script ships |
| `results/` scope | both clusters × both benches, plus a cross-cluster analysis of why one looks slower |

## Three traps the leg script must not reintroduce

Each is first-hand from the packups above, and each silently invalidates a
comparison rather than failing loudly.

1. **`--ep-size` and `--enable-dp-attention` are different parallelism axes**
   (expert vs attention). Gating both on one `if` means `DPA=0` also collapses
   the MoE from ep8 to the TP default, and no latency delta is attributable.
   Emit `--ep-size $TP` unconditionally.
2. **`--chunked-prefill-size` is a GLOBAL budget** that SGLang divides by
   `dp_size` **only when DP-attention is on**. A DPA-off branch that hardcodes
   the per-rank number changes the global chunk 8×.
3. **Prefill activation OOM is fixed by LOWERING `--mem-fraction-static`** — the
   opposite of the decode-side retract fix. Diagnose by phase: decode retract →
   raise; prefill `HSA_STATUS_ERROR_OUT_OF_RESOURCES` at low token usage → lower.

## Verification

No cluster allocation for this task. Verification is static plus locally
runnable, and anything needing real hardware is registered as **not validated**
in the README rather than implied:

- `bash -n` (and `shellcheck` where available) on every script.
- Every infera flag used must be grepped against this repo's
  `infera/engine/sglang/args.py` / `infera/server/args.py` on `main`.
- `python -m infera.server --help` and `python -m infera.engine.sglang --help`
  run locally to confirm spelling.
- Every number in `results/` traces to a packup file or a raw artifact; the spur
  customer-bench ladders are recomputed from `profile_export.jsonl`, not copied
  from a summary line.
- Grep the finished deliverable for leaked local information before declaring done.
