# GLM-5.2 v0.5.21 PD decode "Memory access fault" with `index_share_for_mtp_iteration=true` — root cause, fix, validation (2026-10-20 → 2026-10-21 UTC)

Reproduction kit of the idxshare debug (2026-10-20 01:55Z first fault round → 2026-10-21 02:0xZ kit built).
**Status: fix validated on one node (idxf1a, m2m-097); replicates idxf1b (m2m-268) and idxf1c (m2m-054) are IN PROGRESS**
(started 2026-10-21 01:57:14Z, no result when this kit was built).

Setup: image `infera-sglang:v0521-yihou-1010-mc4340` (sglang `9f5d8712` + image-local diff), same-node 1P1D P4D4 on spur MI355X
(prefill GPU0-3, decode GPU4-7), EAGLE MTP 5/1/6 on decode, triton DSA, TP4/DP4 + DP attention, OffloaderV2 G4K1 step 1 on prefill,
AgentX C56 (A1 fresh + A2 scored, 3600 s each). Correctness is waived by construction (`DECODE_SIMULATE_ACC_LEN=3.61`).

## Goal / success criterion

User rule 2026-10-20 01:5xZ: all experiments run with PR #42553 + index_share ON + decode custom AR ON. With index_share ON the decode
worker died; the user parked perf tuning (06:10Z) to debug it. Pass bar for a fix (env file of idxf1a): **full A1+A2 at C56 with no decode MAF**,
index_share TRUE, the fix as the only change vs the faulting g4n1 config.
Result: **idxf1a passed** (A1+A2, ~170 min of bench, 0 MAF; A2 score 21,814 vs idx-false g4n3 A2 22,045 on the same node, n = 1 each).
Replicates on two more nodes are running.

## Symptom

Decode: `Memory access fault by GPU node-N (...) on address 0x....00000. Reason: Unknown.` → `Fatal Python error: Aborted`, 23-100 min into load,
only with index_share=true (`results/maf/maf_lines.txt`). Kernel log (3 faults, CAP_SYSLOG container): UTCL2 client TCP (shader), PERMISSION fault,
2 MiB-aligned pages (`results/maf/dmesg.from_prep_notes.txt`, transcribed, not raw).

## Root cause chain (file:line in the image tree = sglang `9f5d8712514c2d9dd29c76c6080774db3fd50aa2` + image-local diff; lines re-checked for this kit)

1. `EAGLEWorkerV2.seed_dsa_topk_from_draft_extend` = index_share and seed width (`python/sglang/srt/speculative/eagle_worker_v2.py:294`).
   A PD decode batch without a DSA seed sets `EagleDraftInput.cuda_graph_compatible = False` (`eagle_worker_v2.py:1033`, `:1228`). That bit is
   min-reduced over DP (`managers/scheduler_components/dp_attn.py:54-70`, "forces every DP rank eager"). With index_share off it is never False.
2. An IDLE DP rank then drafts eagerly: `draft()` skips `draft_attn_backend.init_forward_metadata` for idle (`eagle_worker_v2.py:651-659`);
   `_draft_forward_idle` (`:869`, context at `:892`) runs every step under `ForwardContext(attn_backend=draft_attn_backend.attn_backends[i])`.
3. `ModelRunner._forward_raw` keeps that outer context (`model_executor/model_runner.py:1891`). `EagerRunner._execute_idle`
   (`model_executor/runner/eager_runner.py:452-463`) re-initializes `model_runner.attn_backend`, which for the draft runner is the draft-extend
   backend (`eagle_worker_v2.py:389`) — not `attn_backends[i]`.
4. The DSA indexer reads `get_attn_backend()` (`model_executor/forward_context.py:66`; indexer `layers/attention/dsa/dsa_indexer.py:1981`) =
   `attn_backends[i]` with the `forward_metadata` of an earlier real batch. `_should_skip_logits_computation` (`dsa_indexer.py:825`) sees IDLE with
   seq_lens_cpu max 1 ≤ 2048 → `_forward_cuda_k_only` (`:1745`, top-k call `:1832`) ranks `index_topk`(2048)-wide dummy logits with the stale
   lengths (19,238 … 104,740) → `sgl::dsa_topk::coop_topk_kernel<2048,12,4096,1024>` (`python/sglang/kernels/aot/include/hip/dsa_topk_coop.cuh`)
   scans past the row: MAF when the next page is unmapped, otherwise a silent over-read.

First-hand evidence for each link: ROCm debug agent names the kernel + pc (`results/debug_agent/`), call-site guard catches the BAD call with its
Python stack (`results/bad_dumps/dbge5a.first_bad_full.txt`), batch dump shows IDLE bs 1-2 with stale attention metadata
(`results/bad_dumps/dbge5*.skguard_bad.txt`), code read (lines above). Upstream: the same idle path is present at `origin/main` 16eb87c06f
(code read 10-20) and the `eager_runner._execute_idle` body is unchanged at 93088901 (read for this kit).

## Fix

`EagerRunner._execute_idle`, when `batch_size > 0`: after `model_runner.attn_backend.init_forward_metadata(forward_batch)`, also initialize the
active forward-context backend when it is a different object (`has_forward_context()` / `get_attn_backend()`). Additive, 2 imports + 4 lines.
Mounted file `scripts/mounts-src/eager_runner.yihou-idlefix.py`; diffs `patches/0001-*` (vs image) and `patches/0002-*` (ported to upstream main).

## Evidence table (C56, same-node P4D4; MAF / BAD counted by grep over the decode log, `results/maf/maf_lines.txt`)

| round | node | config (vs g4n1) | index_share | fix | outcome |
|---|---|---|---|---|---|
| g4n2 | 250 | + chunk 65536 | true | no | **MAF** 02:50:03Z, ~30 min into A1 |
| g4n2b | 250 | = g4n2 | true | no | **MAF** 04:08:23Z, ~36 min into A1 |
| g4n1 | 097 | new-rule baseline | true | no | A1 done (21,671); **MAF** 04:29:02Z ~43 min into A2 |
| g4n1b | 250 | = g4n1 | true | no | stopped by operator (superseded), 0 MAF while running |
| g4n4 | 250 | decode custom AR off | true | no | **MAF** 05:33:13Z, ~30 min into A1 |
| g4n3 | 097 | index_share false | false | no | A1+A2 clean; A1 20,331, **A2 22,045** |
| g4n3b | 250 | = g4n3 | false | no | A1+A2 clean; A1 19,933, A2 21,044 |
| dbge1a | 097 | + debug agent + seedguard count | true | no | **MAF** 08:53:31Z (A1 +23 min); agent: coop_topk_kernel pc +700; seedguard bad_entries 0 |
| dbge1b | 250 | = dbge1a | true | no | **MAF** 10:42:30Z (~70 min); same kernel, same pc |
| dbge2a/2b/2c | 097 | + indexer top-k guard v1/v2/v3 | true | no | stopped: guards saw no serving calls (blind spots, `notes.md`) |
| dbge2d / 2e | 097 / 250 | + indexer guard v4 (k-only counters) | true | no | stopped: counters frozen in serving |
| dbge3a / 3b | 097 / 250 | + sgl_kernel call-site guard (SKGUARD v1) | true | no | stopped: only warmup/capture sites recorded |
| dbge4a | 097 | + indexer v5 + SKGUARD v2 (sync eager check) | true | no | 1 BAD k-only call 12:17:06Z (len 19,238 in a 2048-wide row) → **MAF** 1 s later, same pc |
| dbge4b | 250 | = dbge4a | true | no | 4 BAD 12:13:51-52Z with no MAF; operator stop issued (bench rc=143 12:21:11Z); 2 more BAD + **MAF** 12:21:12Z |
| dbge5a | 097 | SKGUARD v3 mask (clamp + batch dump) | true | no | 12 BAD clamped, 0 MAF; batch dumps (IDLE, stale metadata) |
| dbge5b | 250 | = dbge5a | true | no | 24 BAD clamped, 0 MAF, ran to job MaxWall 13:42Z |
| dbge6a | 097 | dbge5a guards in count mode + **fix** | true | yes | full A1, 0 BAD, 0 MAF; A1 21,189; DP0 pd_seedless 1360, draft_eager 3653 / 83297 iters (ledger); stopped 10 min into A2 |
| idxf1a | 097 | g4n1 + **fix only** | true | yes | **A1+A2 clean**, 0 MAF; A1 20,530, **A2 21,814** |
| idxf1b | 268 | = idxf1a (node only) | true | yes | **IN PROGRESS** (started 2026-10-21 01:57:14Z) |
| idxf1c | 054 | = idxf1a (node only) | true | yes | **IN PROGRESS** (started 2026-10-21 01:57:14Z) |

Scores: `results/scores.tools_hits.txt` (score = `request_metrics.throughput.per_gpu.total_tput_tps`). Single runs; idx true+fix vs idx false
differ by 1.0 % at A2 on 097 — not a performance claim.

## Navigate

`REPRODUCE.md` · `environment.md` · `notes.md` (wrong turns, guard blind spots, open points) · `patches/` (README with what/why/how) ·
`scripts/harness/` (overlays, base chain, mount tables) · `scripts/mounts-src/` (fix, guards, base overlays) · `scripts/session/` (env per round,
launch/stop helpers) · `scripts/diagnosis/` · `results/` (maf, bad_dumps, debug_agent, seedguard, bench JSON, rounds status/chain, dryruns, launch) ·
`analysis/` (REPORT, ROOTCAUSE, prep notes, ledger snapshot 2026-10-21 02:0xZ).
Copied part rebuilt by `scripts/tools/packup_copy_idxshare.yihou.sh` (copy-only). Server logs (> 4 MB, or not requested) are listed in
`results/not_copied.tsv`, not copied.
