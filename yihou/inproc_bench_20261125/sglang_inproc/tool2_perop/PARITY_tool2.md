# PARITY — tool 2 (per-op micro-bench) for DeepSeek-V4.1-Flash vs the GLM-5.2 port

Reference: AKLR @ 33689332 `aklr/benchmarks/models/glm52/{model_bench,profile_decode,profile_prefill}.py`,
shared `sglang_bench.py`, `kernel_breakdown.py`, `main.py`, and the deepseekv4 port (copies in
`../../research/aklr_src/`); mechanism table and watch-outs from KB handoff `kbk-tool-perop-microbench`.
Target: sglang 6be0d75ae, image `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111`, DeepSeek-V4.1-Flash, MI355X.

Status: **impl** = implemented as in glm52; **adapt** = same function, different mechanism (how/why);
**n/a** = not applicable (why, first-hand). Evidence = GPU point(s) under
`/data/yihou/inproc_bench_20261125/work_t2/iterations/` on smci355-ccs-aus-n03-33 (GPUs 4-7) exercising it;
"CPU" = `tests/test_tool2_cpu.py`.

## Feature rows

| # | glm52 feature (flag / section / mechanism step / watch-out) | status | how / why | evidence |
|---|---|---|---|---|
| 1 | `profile_decode.py` entry | impl | `dsv41/profile_decode.py` | t2_d2, t2_d3 |
| 2 | `profile_prefill.py` entry | impl | `dsv41/profile_prefill.py` | t2_p1b, t2_p2b |
| 3 | `model_bench.py` per model + shared `sglang_bench.py` | impl | `dsv41/model_bench.py`, `sglang_bench.py` | all |
| 4 | load: `one_batch.load_model`, one process per rank, dead rank -> non-zero exit (mechanism "load") | impl | deepseekv4's sentinel `wait()` loop (aborts on the first dead rank) | t2_d1 (rank exited 1 -> exit 1) |
| 5 | `publish_once` / initializers before load (API drift) | adapt | 6be0d75ae: `publish(..., ranks=SpawnRanks(...))` always; `one_batch._init_process_global_configs` (adds bf16 GEMM config) | all |
| 6 | `build_server_args`: force `--disable-cuda-graph` unless `--full-step`; `auto` KV -> `fp8_e4m3`; `resolve_once()`; CONC % dp check | impl | same; plus decode-graph max_bs declared = local batch under `--full-step` (one_batch mechanism; kit lesson: graph not covering the batch silently ran eager) | t2_fs1 (graph replay count) |
| 7 | `--ISL --reps --repeat --out` | impl | + `--result-dir` (report next to provenance) | t2_d3 (`--repeat 2`) |
| 8 | `--CONC` = total, split over DP, must divide | impl | deepseekv4 form: several values per load, `[UNSUPPORTED]` / `[BLOCKED_CAPACITY]` lines, never reduced | t2_d3 (16 and 64 in one load) |
| 9 | `--full-step N` (N+2 synced steps, fastest) | impl | non-spec: fresh native batch per sample outside the clock, `mr.forward` under sglang graphs; counts decode-graph `execute` calls | t2_fs1 |
| 10 | `--trace` (top 35 kernels printed, chrome trace rank 0) | impl | `trace_step`; spec: one DSPARK step | t2_tr1 |
| 11 | `--prefix-cache-hit` (prefill) | adapt | cached prefix handed to the new request through the tree cache's `prefix_device_indices` (6be0d75ae reads a new request's prefix there; glm52 set `req.prefix_indices`) | t2_p2b |
| 12 | record inputs: forward pre-hook, first call wins (mechanism "record") | impl + adapt | pre-hooks on every TREE module; **adapt**: on gfx950 the DSv4.1 loop calls `forward_layer_fused_boundary(model, i, HcState)` and never `layer(...)`, so the `layer` row wraps that call site (also `forward_hc_pre_from_prev` for the non-fused path); `attn.sources` wraps the backend's `forward_low_ratio_sources`; `hc_head` wraps `mhc_combine` | t2_d2 (all 8 `kind\|layer` rows timed) |
| 13 | ambient state re-scoped on replay (`attn_inputs`, MLP flags; watch-out "missing MLP flag restore") | adapt | 6be0d75ae has no `fuse_mlp_allreduce`; every `ForwardFlags` field is snapshotted at the hook and re-scoped, and the active `ForwardContext` (carries `moe_num_token_non_padded`) is re-entered; DSv4.1 has no `ATTN_TP_CONTEXT.set_attn_inputs` user | t2_d2 |
| 14 | inputs recorded before in-place mutation (deepseekv4) | impl + adapt | tensor snapshot also descends into msgspec Structs / NamedTuples (`HcState`, `HcPending`), since `HcState.release()` empties the state after the combine | t2_d2 |
| 15 | `bench()`: GRAPH_REPS=20 calls per graph inside `graph_capture()`, 3 eager warm calls, `--reps` replays, min over `--repeat`, pool freed per op | impl | unchanged | all |
| 16 | prefill inner = `max(1, min(20, 16384 // extend))` | impl | unchanged (CPU test) | t2_p1b, CPU |
| 17 | cold-weight rotation ring (256 MB MALL, 512 MB ring, <=256 slots, `agreed()`) | impl | unchanged; not applied to call-site rows (no single weight) | t2_d2 (Config row), CPU |
| 18 | parent rows cycle through real layers of the kind (deepseekv4) | impl + adapt | `layer` call site re-targeted at each layer index of the kind | t2_d2 |
| 19 | seeded balanced routing (seed 1234), fused shared column kept, draft too | impl | `_balanced_forward` on all 40 target layers and the DSPARK stages; refuses lazy routing outputs without `topk_ids` instead of silently skipping; `--routing native` added as a diagnostic | t2_d2 ("installed on 40 target"), t2_mtp1, CPU |
| 20 | fake KV: random bytes < 0x7C, index scales near 1, no prefill forward (watch-out "zero KV") | adapt | DSv4.1 pools: SWA, `kv_pools[1/2]`, `index_pools[1/2]` (FP4 payload + ue8m0 scale buffers) found by walking sub-pools; scale buffers 0x7E..0x80; filled bytes listed in Memory; `--kv-fill prefill` alternative | t2_d2 (Memory: buffers filled) |
| 21 | MTP KV: draft pool randomized too | adapt | DSPARK draft KV is injected from target hidden states at prefill; the MTP path reaches steady state through a real prefill bootstrap (as the server) instead of random KV + seeded draft input | t2_mtp1 |
| 22 | layer kinds from the built model + roll-up (`Layers`, `add_residuals`, `leaf_totals`, LEAVES partition) | adapt | DSv4.1 kinds: `moe` + `+hash` + `+c{0,1,2}` + `+kvsrc` + `+indexer` + `+engram` + `+late` -> 8 kinds on Flash; `engram` row outside the layer residual | t2_d2 (8 kinds = 40 layers), CPU |
| 23 | `assert_lockstep` + `agreed()` on skips / ring | impl | unchanged | all multi-rank points |
| 24 | decode reference: whole model in one graph (inner=1) | impl | `mr.model.forward(input_ids, positions, fb)` | t2_d2 ratio 0.987 |
| 25 | prefill reference: eager full forward; TTFT = sum of chunks | impl | unchanged | t2_p1b |
| 26 | rank-max under EP / min otherwise (deepseekv4 `world_times`) | impl | decode and prefill | t2_d2 ("max over ranks (EP)") |
| 27 | Summary (step, roll-up, ratio, throughput) | impl | unchanged | t2_d2 |
| 28 | Config (model, layers, parallel, moe, batch, context, kv dtype, spec, weights, measurement) | impl | + KV setup, routing, late-tail rows | t2_d2 |
| 29 | Versions (bench, sglang, aiter, torch, rocm, triton, flydsl, tilelang) | impl | + `aklr`; bench = git rev or, on a node copy without git, sha256 of the tool's files | t2_d2 |
| 30 | Memory, per rank (KV per buffer group, weights, HBM, capacity notes, EP note) | adapt | per DSv4.1 sub-pool; weights counted on device only (t2_d2 summed the 188.8 GB host-resident Engram table into "weights"; fixed from t2_d3) | t2_d3 |
| 31 | Hotspots (leaf rows, shares to 100%, `::` residual contents) | impl | DSv4.1 contents labels | t2_d2 |
| 32 | Per layer kind (tree per kind) | impl | unchanged | t2_d2 |
| 33 | Roll-up (`kind layer` x count + globals) | impl | + `kind engram` rows | t2_d2 |
| 34 | Coverage (hooked, never called, where it lands) | impl | deepseekv4 table form | t2_d2 |
| 35 | Not replayable (ran, failed alone, where its ms sits) | impl | catches RuntimeError and the assertion/type/value errors a standalone replay raises | t2_d2 (empty), CPU |
| 36 | dummy-weight banner + filename tag | impl | unchanged | CPU (string) |
| 37 | auto report name (model, phase, isl, conc, layout, kv dtype, dummy, date) | impl | `mtp`/`decode`/`prefill` | t2_d2 |
| 38 | MTP: build sglang's spec worker in the scheduler's order (glm52 `_build_spec_worker`) | adapt | `DSparkWorkerV2` via `SpeculativeAlgorithm.create_worker`, target `TpModelWorker`, pools, backends, graphs, tree cache — order adapted from tool 3 (`tool3_spec_step/bench/profile_decode.py`) | t2_mtp1 |
| 39 | MTP: one step = `forward_batch_generation`, per-forward graph timing (`bench_forwards`) summed as the reference | impl | target + draft `model.forward` wrapped | t2_mtp1 |
| 40 | MTP: draft model one row outside the per-layer roll-up; draft per-op rows printed | impl | `draft model` row = draft forwards of the step; per-op draft table (`draft|s<i>.<op>`) in the report | t2_mtp1 |
| 41 | MTP: throughput with `SGLANG_SIMULATE_ACC_LEN`, "not reported" without; ITL row | impl | same env knob (DSPARK reads it in its verify executor) | t2_mtp1 |
| 42 | MTP: implied second run with `--full-step 3` (+ phase split by CUDA events) | impl | phases: propose / verify compact / verify non-compact / accept+finalize / forward:draft / forward:target | t2_mtp1 (2nd run) |
| 43 | MTP: seed DSA top-k carried between steps (glm52 `_bench_dsa_seed`) | n/a | GLM-5.2-specific draft input field; DSPARK carries its state in `next_draft_input`, which the bench installs after each step exactly as the scheduler (`batch.spec_info = result.next_draft_input`) | code read: `spec_dspark._forward` |
| 44 | prefill refuses `--speculative-algorithm` | impl | unchanged | CPU (code path) |
| 45 | prefill: chunk plan, context with factor 2, page rows freed per chunk | impl + adapt | chunk = resolved `--chunked-prefill-size` (already divided by attn DP in 6be0d75ae `arg_groups/parallel_hook.py`); **adapt**: ops recorded inside the DSv4.1 late-layer SWA tail are replayed inside it (`enter_late_layer_tail` + `enter_late_layer_dp` re-entered around the call) -- without it t2_p2 skipped late ops on shape checks and then hit an HSA hardware exception | t2_p1b (4 chunks), t2_p2 (failure before the fix) |
| 46 | prefill: Chunk timeline (+ attention path) | adapt | GLM's MHA/MLA switch does not exist; the column is the late-layer SWA tail row count | t2_p1b (tail 128 rows) |
| 47 | prefill: growth table across chunks | impl | unchanged | t2_p1b |
| 48 | `main.py` e2e (real text, chunked prefill, 256 teacher-forced steps) | impl + adapt | string rids; `--text-file` for offline nodes | t2_p1d | prefill #5034 ISL 32768 (16384 rows/rank) with `--mem-fraction-static 0.7` | 2 chunks completed: TTFT 874.6 ms, roll-up 872.6, ratio 0.998, 37,465 tok/s | 2 (the 16384-row shape) |
| t2_e2e1 |
| 49 | Kineto `kernel_breakdown` per hotspot + Top kernels with top-5 targets (deepseekv4; glm52 TODO) | impl | function call sites get a Kineto child range so parent samples stay exclusive; no `aklr` routes in the image -> grouped by innermost API | t2_d2 |
| 50 | shape export `AKLR_EXPORT` / `--export-only` | adapt | `aklr` absent from the image (Versions: `aklr absent`): `shape_export.py` records, during the one eager recording step only, (a) every aiter callable bound into an `sglang.*` module namespace, (b) every non-aten dispatcher op (TorchDispatchMode), (c) every hooked op's input/weight specs, to `<result>/aklr_export/rank<r>/calls.pid<p>.<run>.jsonl` + summary; AKLR's recorder is used when importable. Kernels launched from functions imported locally inside sglang functions are not seen (as AKLR records only proxied entry points) | t2_ex1, t2_ex2, CPU |
| 51 | capacity refusal `[BLOCKED_CAPACITY]` (deepseekv4) | impl | full KV, SWA, request slots, a2a cap | CPU, t2_d3 |
| 52 | `ck_moe_stage1` M<16 hole (deepseekv4 README) | n/a (not hit) | the Top-kernels section shows DSv4.1 experts in `aiter::fused_moe_` -> FlyDSL `mfma_moe1_*` kernels here, not `ck_moe_stage1`; CONC 16 (4 requests/rank) ran | t2_d2 |
| 53 | `--no-kernels` (Kimi-K3 port) | impl | extra, not a glm52 feature | not run on GPU |
| 54 | `validate_model` (reject other archs before loading) | impl | DeepseekV41/V4 ForCausalLM | all |

## GPU evidence

All points: DeepSeek-V4.1-Flash real weights, `fp8_e4m3` KV, image sha256:6a0b7952..., container `yihou-inproc-t2`
(GPUs 4-7, `--pid host`), provenance files in each point dir (command/env/versions/kit hashes/inspect, redacted).
"#5034" = `presets/dsv41_tp4_dpa4_ep4.args` (TP4 DPA4 EP4) with `presets/dsv41.env`. Decode context ISL 1024
unless noted; random KV; balanced routing unless `native`.

| point | entry / setup | result | exercises rows |
|---|---|---|---|
| t2_d1 | decode #5034 C16 | first run: `ImportError get_world_group` (6be0d75ae moved it to `distributed.parallel_state`); rank exit 1 -> launcher exit 1 | 4 (dead-rank gate) |
| t2_d2 | decode #5034 C16, reps 5 | step 9.944 ms, roll-up 9.816, ratio 0.987, 1609 tok/s; 8 kinds/40 layers; Coverage + Top kernels present; Not replayable empty | 1,3,8,12-15,17-20,22-29,31-37,49 |
| t2_d3 | decode #5034 C16 + C64 one load, reps 10, repeat 2 | C16: 9.914 / 9.776 / 0.986; C64: 15.800 ms / 15.654 / 0.991, 4051 tok/s; C64 hotspots `mlp.experts` 51.2%, `layer.<rest>` 24.6%, `attn.<rest>` 9.2%; `attn.sources` timed (1.3%) | 7,8,30 |
| t2_fs1 | decode #5034 `--full-step 32` C16 + C64 | C16 9.922 ms; C64 13.706 ms (fastest of 34; decode-graph `execute` 34/34) | 6,9 |
| t2_fsn1 | same, `--routing native` | C16 9.864 ms; C64 13.694 ms | 19 (diagnostic) |
| t2_fs8k / t2_fsn8k | `--full-step 32`, ISL 8192, C64, balanced / native | 13.798 / 13.746 ms | 9 |
| t2_d8k | decode #5034 ISL 8192 C64, reps 10 | step 15.931 ms, roll-up 15.773, ratio 0.990, 4017 tok/s | 8 |
| t2_tp4d1 | decode plain TP4 (`presets/dsv41_tp4.args`) C16 | step 8.375 ms, roll-up 8.268, ratio 0.987, 1910 tok/s; `mlp.experts` 41.8% | layouts |
| t2_tr1 | decode #5034 C16 `--trace` | 109 distinct kernels, 8747 launches, eager; `trace.json` (6.1 MB, rank 0) | 10 |
| t2_ex1 / t2_ex2 | decode #5034 `--export-only` (C16+C64 / C16) | per-rank JSONL; t2_ex2 rank 0: 95 distinct calls, 288 aiter-API calls (4 APIs), 375 dispatcher-op calls (8 ops incl. `aiter::fused_moe_`, `aiter::gemm_a8w8_blockscale`, `sgl_kernel::deepseek_v4_topk_transform_512`), 70 module calls | 50 |
| t2_mtp1 | MTP #5034 + DSPARK, sim 3.51, C16 | failed: `batch.forward_iter` None (the scheduler stamps it per launch); fixed in `spec_dspark._forward` | 38 |
| t2_mtp2 | MTP #5034 + DSPARK block 5, `SGLANG_SIMULATE_ACC_LEN=3.51` (match-expected, real-draft-token), C16, reps 5 | per-op pass: forwards 19.271 ms (target verify 18.177 + draft 1.094), roll-up 19.125, ratio 0.992, 2914 tok/s, ITL floor 5.49 ms; 3 draft stage `layer` rows Not replayable (24 vs 20 rows; glm52 also had 3 unreplayable draft ops); implied `--full-step 3`: step 15.295 ms, 3672 tok/s at accept 3.51; phases propose 1.687 / verify 13.212 / accept+finalize 0.115 ms | 21,38-42 |
| t2_p1 | prefill #5034 ISL 32768 (chunk 65536 -> 16384/rank) | aborted: `hipErrorLaunchOutOfResources` surfaced during the first `layer` replay (C++ terminate, not catchable); see open problems | 2 |
| t2_p2 | prefill ISL 16384, hit 0.5, chunk 4096/rank | before the tail fix: late-layer ops failed shape checks (`Tensor<4096>` vs tail 128 rows), then HSA hardware exception -> abort | 45 |
| t2_p1b | prefill #5034 ISL 16384, chunk 16384 (4096/rank), reps 3 | TTFT 491.3 ms (4 chunks, 121.9-123.4 ms each), roll-up 488.7, ratio 0.995, 33,350 tok/s; tail rows 128 | 2,16,25,45-47 |
| t2_p2b | same, `--prefix-cache-hit 0.5` | 2 chunks on prefixes 8192/12288; TTFT 246.6 ms, roll-up 244.9, ratio 0.993; prompt 66,446 tok/s, computed 33,223 tok/s | 11 |
| t2_p1dbg | prefill ISL 16384, chunk 16384/rank, `HIP_LAUNCH_BLOCKING=1 AMD_SERIALIZE_KERNEL=3` | completed (ratio 0.819 under launch blocking: the eager reference is slowed, not comparable) | diagnostic for t2_p1 |
| t2_p1c | prefill #5034 ISL 32768, chunk 16384/rank, after the tail fix, no debug env | chunk 1 (prefix 0) fully timed; chunk 2 (prefix 16384) aborted in the `moe+c0+engram\|layer` replay: `HSA_STATUS_ERROR_OUT_OF_RESOURCES`, hang analysis names a dispatch with grid 12288x256 and private_seg_size 1024 | 2 |
| t2_e2e1 | `main.py dsv41 decode` #5034 C16 ISL 1024, `--text-file .../srt/server_args.py` (sha256 b524acc6...) | TTFT 137.2 ms, median teacher-forced step 8.718 ms, 1835 tok/s | 48 |

### Cross-checks of the decode step (TP4 DPA4 EP4)

| shape | tool 2 per-op full forward (graph of `model.forward`, cuda graph disabled config) | tool 2 `--full-step` (sglang decode graph) | kit step_bench TPOT (VALIDATION.md, real prefill, median) | tool 2 e2e (real text, median) |
|---|---|---|---|---|
| C16, ISL 1024 | 9.914 ms (t2_d3) | 9.922 balanced / 9.864 native | 9.77 ms (sgl_s3_c16_1k, OSL 32) | 8.718 ms (t2_e2e1) |
| C64, ISL 1024 | 15.800 ms (t2_d3) | 13.706 / 13.694 | -- | -- |
| C64, ISL 8192 | 15.931 ms (t2_d8k) | 13.798 / 13.746 | 13.06 ms (sgl_s4_c64_8k, OSL 128) | -- |

Observations only (no mechanism claimed): at C16 the per-op forward, `--full-step` and step_bench agree within
1.5 %; at C64 the per-op full forward is 15 % above `--full-step` at both ISL 1024 and 8192, while balanced vs native routing moves
`--full-step` by < 0.4 %. The e2e median step (real text, real KV) is 12 % below the synthetic `--full-step`.

### MTP cross-check with tool 3 (TP4 DPA4 EP4, #5034 + DSPARK block 5, simulated accept 3.51)

Tool 3 numbers come from tool3-port's message of 2026-11-25. They are second-hand here; the details are in
`../tool3_spec_step/VALIDATION_tool3.md`. All runs are single runs.

| shape | tool 2 | tool 3 (in-process whole iteration, real prefill) | real #5034 server |
|---|---|---|---|
| C16, ISL 1024 | `--full-step`: 15.295 ms per step -> 4.36 ms/user at the assumed 3.51; per-op forwards 19.271 ms (t2_mtp2) | TPOT 4.651 ms/user, realized accept 3.667 (~17.1 ms per iteration) | -- |
| C64, ISL 8192 | not run (GPUs released) | TPOT 6.300 ms/user, accept 3.5625 (~22.4 ms per iteration); verify graph 19.7 ms, draft graph 1.9 ms per iteration | client TPOT median 7.31 ms; server-log steady interval 7.14 ms |

The tool 2 C16 step (15.30 ms) is 10 % below tool 3's iteration (~17.1 ms). Tool 2's step times only
`forward_batch_generation`; the commit happens outside the clock. That difference in clock boundary is not
isolated by a measurement. The C64 ISL 8192 MTP point is the next tool 2 run to make for a direct comparison
with the server.
