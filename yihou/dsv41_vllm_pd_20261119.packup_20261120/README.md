# DSv4.1-Flash on MI355X with vLLM — PD-disaggregation study, pack-up 2026-11-20

**Time:** 2026-11-19 08:00Z – 2026-11-20 05:30Z. **Owner:** yihou. **Branch:** `dev/yihou/dsv41` (infera).
**Status:** Phase 1 (research) done; Phase 2 (reproduce + PD feasibility) in progress. PD 1P1D bring-up (E7) started on
crusoe v3-186 at 05:22Z on 11-20 — NOT included here (no result yet).

## Goal (from `process/task_mission_dsv41.md`)
DeepSeek-V4.1-Flash, MI355X, vLLM engine, prefill/decode disaggregation, InferenceX AgentX throughput.
Target (user, 11-19): +50% over the public MI355X best **427,451 tok/s/GPU** (vLLM colocated DP2+UMBP c384,
InferenceX CI 2026-11-18, PR #5036) → **~641k tok/s/GPU**.

## Results so far (official AgentX contract, 3600 s, seed 42, synthetic DSpark AL 3.51; single runs)
All on openai node smci355-ccs-aus-n10-29, image `vllm/vllm-openai-rocm:nightly-60bd6dd2…@sha256:67a3a0fb…`,
**3-4 servers running concurrently on the one node** (host interference possible, not measured).

| run | config | tok/s/GPU | reference | note |
|---|---|---:|---|---|
| e2-r2-c384-11191138 | R2 = public-bar recipe (DP2 + UMBP host tier) c384 | **407,049** | CI 427,451 (-4.8%) | p90 interactivity 14.8 (CI 15.8) |
| e1-r1-c128-11191033 | R1 DP2 c128, no UMBP | 255,797 | CI 271,329 (-5.7%) | |
| e1-r1rep-c128-11191210 | R1 replicate | 266,906 (raw*) | CI -1.6% | replicate spread 4.3% |
| e3-r1-c192-11191211 | R1 at c192 | 336,424 (raw*) | — | full 3628 s window |
| e3-r1-c256-11191212 | R1 at c256 | 346,601 (raw*) | — | full 3630 s window |
| e4c-r1ds-c128-11191040 | R1 + `--kv-cache-dtype fp8_ds_mla` | 243,585 | -4.8% vs R1 | KV pool -13% |
| e4b-r1dsnosp-c128-11191039 | R1 + fp8_ds_mla + sparse-MLA off (= kernel path any PD connector forces) | 221,199 | -13.5% vs R1 | ITL p50 14.7→20.7 ms |

\* raw = recomputed from `profile_export.jsonl` by `scripts/raw_score_from_export.py` (method reproduces the
harness score of e1-r1-c128 exactly); the harness post-processing did not run because the node was preempted
after profiling ended, so the contract check was not executed for these three.

Key findings (first-hand unless marked): any non-UMBP KV connector forces `fp8_ds_mla` KV + the Triton sparse-MLA
path on ROCm (fp8_e4m3 is rejected at load), costing ~13.5% colocated at c128 → PD must win that back first.
Phase-1 research: `research/SYNTHESIS.md` (state of the art, model facts, plan), `research/dynamo_vllm.md`
(NVIDIA Dynamo+vLLM lever table).

## Navigate
- `REPRODUCE.md` — step-by-step to re-run every point above (openai or crusoe).
- `environment.md` — hardware, image digest, package versions, external paths, secrets needed.
- `notes.md` — gotchas found (each cost a failed run), wrong turns, open questions.
- `scripts/kit/` — the launcher/bench kit (serve.sh, run_point.sh, serve_pd.sh, pd_smoke.*, watchdog.sh, …).
- `configs/` — every config used (+ dry-run renders). `results/` — per-run summaries, ledger.csv, node preflight.
- `research/` — all Phase-1/2 research reports. `process/` — mission, project CLAUDE.md, team-poll log.
