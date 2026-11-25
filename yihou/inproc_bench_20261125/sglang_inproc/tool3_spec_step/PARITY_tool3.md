# PARITY_tool3.md — GLM-5.2 tool 3 -> DeepSeek-V4.1 port, feature by feature

Reference: `/home/yihou/dev/git.16-19/infera.dev.yihou.sglang.bench.fast.script/sglang_decode_internal_bench_and_profiling/`
(GLM-5.2-MXFP4, sglang 402df1e1, EAGLE 5/6/1) and KB handoff `kbk-tool-inprocess-step-bench`.
Port: this directory (DeepSeek-V4.1-Flash, sglang 6be0d75ae, DSPARK block 5). Status values:
**implemented** (same behaviour), **adapted** (same function, changed where the model/engine forces it; how + why
with the first-hand evidence), **N.A.** (not applicable; why). GPU evidence = iteration names in
`VALIDATION_tool3.md` (all on smci355-ccs-aus-n03-33, GPUs 0-3, `/data/yihou/inproc_bench_20261125/work_t3/iterations/`).

Engine facts the adaptations rest on (code read of the pinned image source, sglang 6be0d75ae):
- E1 `speculative/dspark_components/dspark_worker_v2.py`: `DSparkWorkerV2` replaces `EAGLEWorkerV2`; its decode step is
  draft (`DraftBlockProposer.propose`, draft runner decode graph) -> target verify (target decode graph) -> accept ->
  `commit_hidden` (draft KV injected from target hidden). There is no draft-extend forward or graph runner.
- E2 `arg_groups/speculative_hook.py:833-865`: `--speculative-dspark-block-size 5` -> gamma 5 -> verify window
  (speculative_num_draft_tokens) 6; accept lengths are bonus-inclusive in [1, 6] (same bound as EAGLE 5/6/1).
- E3 `speculative/spec_utils.py:268-283` + `dspark_verify.py:172,240-252`: simulated acceptance is drawn from a private
  `torch.Generator` seeded from the server seed, one scalar per iteration (`sample_simulated_acc_len`), not the process RNG.
- E4 `managers/scheduler.py:596-626, 1192-1270`: per-rank construction order (bootstrap.init_parallel_runtime ->
  init_layer_runtime -> moe/fp8/fp4/bf16 configs -> TpModelWorker -> spec worker -> pools -> attention backends ->
  graphs -> prewarm/post-capture -> build_kv_cache tree cache). ParallelState is built by bootstrap, not passed in.
- E5 `managers/scheduler.py:5418-5442` (non-overlap spec-v2 branch) and `batch_result_processor.py:989`: snapshot/restore
  of ScheduleBatch fields around the worker call, `spec_info = next_draft_input`, `seq_lens = new_seq_lens`,
  `req.kv.commit_decode(accept)`.
- E6 `managers/scheduler_components/dp_attn.py:586` `prepare_mlp_sync_batch_raw` is the DP-metadata gather the Scheduler and
  `bench_one_batch` call; ScheduleBatch fields were renamed (`can_run_decode_cuda_graph`, `can_run_dp_draft_cuda_graph`,
  `can_run_dp_prefill_cuda_graph`).
- E7 `mem_cache/deepseek_v4_memory_pool.py`: `DeepSeekV4TokenToKVPool` = SWA ring + C4/C128 compressed pools with
  compress-state pools + FP4/FP8 indexer pools (+ engram host table in the model runner); several KV layouts
  (V4 packed fp8+bf16 rope+scales, V4.1 fp4, uniform fp8). GPU: target pool 12 contiguous buffers, 46.4 GB/rank (t3_e3_eager).
- E8 `server_args` resolution: `--enable-dp-attention --dp-size 4` resolves to `attn_dp_size=4, dp_size=1`
  (probe in the container, PARITY section "main() asserts"); `page_size` resolves to 256.
- E9 `model_executor/runner/decode_cuda_graph_runner.py:2094-2101`: graph `execute` is wrapped in
  `device_timer_ctx(runner.device_timer, forward_mode.name.lower())`; eager runner uses decode/target_verify/extend/idle.

## bench/profile_decode.py — CLI flags

| GLM-5.2 flag | status | port | evidence |
|---|---|---|---|
| `--model-path` (required) | implemented | same | all |
| `--batch-size` (global C, default 16) | implemented | same | t3_g1_graph (16), tp4_ep4_dpa_on_c64_yihou (64) |
| `--input-len` (default 70000) | adapted | default 8192 (this port's reference ISL; GLM's 70000 was its own sweep point); mutually exclusive with spec | all uniform points |
| `--input-len-spec` uniform/bimodal/normal/list/@file | implemented | `bench/isl_spec.py` copied verbatim | t3_h1_hetero |
| `--output-len` (default 10000) | adapted | default 1024 (reference OSL of this port) | c64 point |
| `--accept-length` (default 3.61, bound [1,6]) | adapted | default 3.51 (#5034 `SGLANG_SIMULATE_ACC_LEN`), bound [1, verify window 6] (E2) | c64 point |
| `--accept-method` match-expected / multinomial | implemented | sets `SGLANG_SIMULATE_ACC_METHOD` (used by DSPARK via `sample_simulated_acc_len`, E3) | c64 (match-expected) |
| `--accept-token-mode` real-draft-token / fixed | implemented | sets `SGLANG_SIMULATE_ACC_TOKEN_MODE`; recorded. Code read: DSPARK's accept path does not read it (only EAGLE's `eagle_utils.py` does) | config_yihou.json of every point |
| (new) `--accept-greedy` | added | DSPARK-only `SGLANG_SIMULATE_ACC_GREEDY` (environ.py:543-549), off as in #5034 | config |
| `--tp-size` (default 8) | adapted | default 4 (#5034 shape) | all |
| `--ep-size` (default 1) | implemented | same | c64 (4) |
| `--enable-dp-attention` | implemented | dp = tp, local batch = C/dp | all (DPA4) |
| `--result-dir` (required) | implemented | same | all |
| `--disable-cuda-graph` | implemented | same | t3_e3_eager |
| `--max-steps` (0 = full OSL) | implemented | same | t3_e3_eager (8), t3_p1_profile (40) |
| `--warmup-steps` (default 3) | implemented | same, followed by release + re-prefill (see bootstrap) | all |
| `--seed` | implemented | prompt ids, ISL generator, `--random-seed` of the engine, acceptance generator re-seed | all |
| `--log-interval` | implemented | same | all |
| profiling flags (`--profile --profile-start-step --profile-num-steps --profile-activities --profile-ranks --profile-dir --profile-with-stack --[no-]profile-record-shapes --[no-]device-timer --profile-top-n --profile-graph-capture-trace`) | implemented | `bench/profiling_yihou.py` (GLM copy; categories list updated, E9) | t3_p1_profile |
| pass-through ServerArgs (`--enable-profile-cuda-graph`, `--mem-fraction-static`, ...) | implemented | `parse_known_args` -> ServerArgs; `presets/dsv41_ix5034.args` carries the #5034 engine flags | all |
| (new) `--prefill-tokens` | added | max prompt tokens per bootstrap prefill forward (bootstrap is a real prefill here) | all |
| (new) `--dp-metadata gather|static` | added | gather = engine's MLP-sync (E6, default); static = GLM-style hand-set fields | all (gather) |

## bench/profile_decode.py — behaviour

| GLM-5.2 feature | status | port (how / why) | evidence |
|---|---|---|---|
| `--help` stdlib-only; plain python (rejects torchrun via WORLD_SIZE) | implemented | same; `tests/test_batch_state.py::test_help_requires_no_torch_or_sglang` | CPU test |
| `validate_args` (topology, positivity, accept bound, exclusive ISL flags, profile args) | implemented / adapted | accept bound = verify window 6 (E2) | CPU tests |
| `resolve_input_lens` once in parent, shipped to ranks (Plan 1) | implemented | same | t3_h1_hetero |
| `server_cli`: bench owns topology + graph flags, raises otherwise | adapted | also owns the 6be0d75ae aliases `--tensor/--data/--expert-parallel-size` | `tests/test_topology.py::test_cli_rejects_conflicting_forwarded_topology` |
| injected defaults EAGLE 5/6/1, fp8 kv, flydsl DSA backends, graph bs = local batch, random seed, radix off, skip tokenizer, no overlap, trust-remote-code | adapted | DSPARK + block 5 (#5034); kv dtype / attention backends left to the engine's DSv4.1 resolution (`kv_cache_dtype fp8_e4m3`, `attention_backend dsv4`, E8 probe); graph bs = max bs = local batch (resolved `cuda_graph_config.decode.bs=[16]` at C64, captured_decode_bs in every result); other defaults same | config_yihou.json, result `captured_decode_bs` |
| `configure_acceptance` before spec_utils import, in parent and every rank | implemented | env set before any sglang import + `envs.*.set`, in parent and ranks | config `simulated_acceptance_env` |
| `create_workers` in scheduler order | adapted | `Engine` follows E4 verbatim (incl. `account_preloaded_weights`, `resolve_decode_retraction_backup`, `init_hicache_draft_plan`, `prewarm_sampling`, `post_capture_resize_kv_pool`, `agree_on_stage_request_capacity`, startup weight-load overlap); spec worker from `SpeculativeAlgorithm.create_worker` | all GPU points |
| ParallelState from the topology | adapted | built by `bootstrap.init_parallel_runtime` (E4); the bench compares `get_parallel()` with `topology.parallel_state_kwargs(rank)` field by field and raises on mismatch | every point (no mismatch raised), `tests/test_topology.py::test_resolved_parallel_state_comparison` |
| `initialize_kv` (random fp8 bytes into every MLA kv_buffer + DSA index keys/scales; refuse metadata-only KV) + `validate_kv_layout` | adapted | prefix KV is written by a **real prefill of the synthetic random prompts** (E7: the DSv4.1 pool has stateful C4/C128 compressors, a SWA ring, FP4 indexer and several byte layouts a byte fill cannot make consistent). `validate_kv_layout` now refuses an unknown pool class or a pool with no physical buffers; the pool inventory (class, buffer count, bytes) is in `memory.kv_pools`. Result field `prefix_kv_source` says so; `synthetic_prefix` stays true (token ids are synthetic) | memory.kv_pools in every result |
| `allocate_batch` (uniform-max reservation of ISL+OSL+reserve, capacity / row / context checks) | adapted | same three checks up front (`reserved_tokens` vs `available_tokens_before_prefill`, `req_to_token_row`, `context_len`); reserve = 2 x verify window = 12 (DFlash `prepare_for_decode`, same 12 as GLM). Allocation itself is done by the engine: prefill allocates ISL, `DFlashDraftInputV2.prepare_for_decode` allocates decode headroom page by page, as in serving (GLM pre-reserved everything) | every result `memory` |
| string request ids offset by attn-DP rank | implemented | `bench-<offset+i>` (ints break DP sync, kit VALIDATION.md) | all |
| `bootstrap`: one-token untimed target+draft forward over the synthetic prefix, asserts draft state | adapted | untimed real prefill through `DSparkWorkerV2._forward_prefill` (target prefill + draft KV injection), grouped by `--prefill-tokens` and merged with `ScheduleBatch.merge_batch`; asserts bonus tokens per request and post-prefill seq_lens == ISL. Draft state fields differ (E1: bonus_tokens / new_seq_lens, no hidden/topk) | all |
| warmup then reset to exact ISL | adapted | warmup iterations, then `release_kv_cache` of every request (server path) + `clear_cache_pool` + re-prefill | all (warmup 2-10) |
| reseed acceptance coins before the measured loop | adapted | re-creates the engine's private acceptance generator (E3) instead of `torch.manual_seed` only; realized accept is then independent of warmup length (eager 8-step and graph runs start 4,4,3) | t3_e3_eager vs t3_g1_graph; 3.5625 bit-identical in c / e / EP1 |
| `prepare_dp_metadata` | adapted | default: engine MLP-sync gather (E6) + assert gathered decode counts == [local]*dp; `--dp-metadata static` sets the renamed fields by hand | all (gather) |
| `commit_result` | adapted | E5 path: snapshot/restore, `spec_info = next_draft_input`, seq_lens from `new_seq_lens`, `kv.commit_decode(accept)`, output_ids extended with the accepted tokens; committed <= allocated is the engine's own invariant | all |
| cross-rank progress check every warmup + measured iteration (7-int signature, digest, encoded invalidity) | implemented | same; accept bound parameter = 6 | all |
| `validate_worker_progress` (new = prev + accept) | implemented | same | all |
| `count_graph_executions` (target / draft / draft_extend execute counters) | adapted | target = target `decode_cuda_graph_runner.execute` (verify graph), draft = DSPARK draft runner `decode_cuda_graph_runner.execute`; `draft_extend` kept as key, null runner (E1) | t3_g1_graph 18/18/0, c64 point |
| `target_graph` per step, `target_graph_iterations` | implemented | `result.can_run_cuda_graph` | all |
| terminal policy (pay terminal compute, count only useful) | implemented | same | all complete points (`terminal_extra_tokens`) |
| per-rank JSON, steps JSONL, all_gather aggregate, rank-0 result + steps | implemented | same file names | all |
| result fields (`expected_accept_length ... draft_extend_graph_available`) | implemented / adapted | all GLM fields kept; `pinned_sglang` = 6be0d75ae; `draft_extend_graph_available` = null; added `prefix_kv_source`, `speculative_algorithm`, `dspark_gamma`, `verify_num_draft_tokens`, `dspark_verify_mode`, `captured_decode_bs`, `accept_greedy`, `dp_metadata` | result_yihou.json |
| `module_source_paths` provenance | adapted | server_args, dspark_worker_v2, tp_worker (was eagle_worker_v2) | all |
| `main()` asserts: resolved topology, PP1/CP1/DCP1/MoEDP1/one node, moe-a2a none, spec config, no dummy weights | adapted | topology checked on the resolved view's `attn_dp_size` (E8); spec check = (DSPARK, 6) | all |
| config_yihou.json byte-compatible when features off (profile / ISL-spec keys omitted) | implemented | same rule; server_cli passed through `common/redact.py` | all |
| `--profile` env set in parent before spawn | implemented | same | t3_p1_profile |
| rank spawning `mp.spawn` | adapted | `multiprocessing.Process` per rank with `maybe_reindex_device_id` (the pinned bench_one_batch pattern); parent stops waiting as soon as one rank fails (otherwise peers hang in a collective) | t3_e1_eager (failure path exercised) |

## bench/batch_state.py, topology.py, isl_spec.py

| GLM-5.2 | status | note |
|---|---|---|
| `DecodeAccounting` (+ summary fields) | implemented | `max_accept_len` = 6 passed explicitly |
| `validate_worker_progress`, `count_graph_executions`, `module_source_paths`, `required_token_capacity` | implemented | |
| `validate_kv_layout` | adapted | see initialize_kv row |
| `DecodeTopology`, `request_id_offset`, `summary`, `representative_ranks` | implemented | |
| `parallel_state_kwargs` | adapted | used for comparison with the engine's state (`compare_parallel_state`) |
| `prepare_dp_metadata` | adapted | see above |
| `lens_digest`, `rank_progress_signature`, `validate_rank_progress`, `aggregate_rank_summaries` | implemented | accept bound parameter |
| `isl_spec.py` (all modes, private RNG, exact Fraction bimodal, list file, describe) | implemented | verbatim copy |

## bench/profiling_yihou.py

| GLM-5.2 feature | status | note / evidence |
|---|---|---|
| torch.profiler window over measured iterations, start/stop latch, close at loop end | implemented | t3_p1_profile |
| `_ProfilerTorch` not reused (rank-subset barrier deadlock); its trace filename convention | implemented | `trace_filename` gets `get_parallel()` |
| DeviceTimer per runner attached after warmup | adapted | runners = target model runner + DSPARK draft model runner (GLM: target, EAGLE draft runner, draft_runner_list); categories = E9 | t3_p1_profile device_timer_rank_0 |
| top-N tables (device kernels only / all / CPU), annotation exclusion, text tables | implemented | t3_p1_profile |
| MEM / CUDA_PROFILER activities, memory snapshot | implemented | code path same as GLM; not exercised on GPU |
| `profile_meta` (is_performance_measurement false, env snapshot, capture caveat) | implemented | t3_p1_profile |
| `--profile-graph-capture-trace` / `--enable-profile-cuda-graph` (capture inventory) | implemented | code path; not exercised on GPU |

## bench/compare_server.py, bench/check_compare_paths.py

| GLM-5.2 | status | note / evidence |
|---|---|---|
| compare_server: fixed waves, warmup requests, per-wave + aggregate TPOT, graph bs = actual wave sizes | implemented | `wave_sizes`, `summarize_waves`, per-wave files; CPU tests `tests/test_comparison.py` |
| initial_state `sikl` | implemented | fresh real prefill per wave; OSL decode tokens |
| initial_state `fake-server` (zero KV, one-hot EAGLE draft state, OSL-1 decode tokens) | adapted | replaced by `server-like`: real prefill emits output token 1 (as a served request), OSL-1 decode tokens. The fake-input decode server it mirrored does not exist for DSv4.1 (tool 1 waived by the user) and DSPARK has no hidden/topk draft state to fake (E1) |
| DP1-only restriction | adapted | DPA supported (#5034 server is DPA4) |
| rejects `--input-len-spec` | implemented | same |
| check_compare_paths: graph / graph_repeat / eager / eager_repeat from identical initial state, per-step equality + float diffs | adapted | eager forced by detaching both decode graph runners; compares accept lens, new seq lens, emitted ids, bonus tokens (exact) and verify logits (max-abs, rel-L2, argmax) instead of EAGLE hidden/topk (E1) | see VALIDATION_tool3.md |

## scripts/

| GLM-5.2 script | status | port |
|---|---|---|
| `create_container.sh` (pinned digest, owner label, ro model, host fixes, image-keyed JIT dir, env) | adapted | wraps `common/lib.sh kit_create_container` (name `yihou-inproc-*`, label owner=yihou, ro model/kit, caches under /cache, `--pid host`, `ROCR_VISIBLE_DEVICES`) + asserts the image digest; env = `presets/dsv41.env` (#5034). GLM's four host fixes were for an NFS-root-squash bare-metal host; here the kit is copied to local disk (`sync_kit.sh`) and the container runs as root |
| `run_decode.sh` (--dry-run, digest + owner asserts, no-clobber, command/git_head/code.diff/code_hashes/bench_snapshot/container-inspect, launch_status) | implemented | + `launch_started.json`, redacted container-inspect, `common/provenance.py` (command/env/versions/kit hashes, redacted), ARG CHECK (trap #10), `ARGS_FILES` presets; git head / diff come from `sync_kit.sh` since the node copy has no .git |
| `run_profile.sh` (env knobs CONC ISL OSL ACCEPT MEMFRAC WARMUP EP_SIZE MAX_STEPS PROFILE_* GRAPH GRAPH_PACKET_CAPTURE CAPTURE_PROFILE, MAX_STEPS=0 guard, launch_started, is_performance_measurement false, writable CWD for capture profile, knobs json) | implemented | defaults re-pinned to C64/8192/1024/3.51, TP_SIZE added; delegates to run_decode.sh |
| `sweeps/run_tp8_ep8_dpa.sh`, `sweeps/run_tp8_ep1_dpa.sh` | adapted | `sweeps/run_tp4_ep4_dpa.sh`, `sweeps/run_tp4_noep_dpa.sh` (#5034 shape TP4/DPA4; single-variable EP pair) |
| `verify_point.py` | adapted | same gates; topology parameterized (default TP4/DP4/EP4/DPA); accept constant parameter (default from this port's reference run); added graph-execute == iterations gate and profiled-run rejection |
| `collect_sweep.py` | implemented | GLM copy's broken import (trap #17) fixed; expectations forwarded |
| `graph_visibility_probe.py` | implemented | verbatim |
| `equivalence_smoke.sh` (patched without --profile vs pristine, identical args) | implemented | PRISTINE = a bench_snapshot dir; compares deterministic fields + config key set, prints EQUIVALENCE ok/FAILED |
| (new) `sync_kit.sh`, `arg_check.py`, `server_reference.sh`, `server_tpot.py`, `lib_t3.sh` | added | kit copy + provenance; ARG CHECK; real #5034 server + fixed-length client; TPOT comparison |

## tests/

| GLM-5.2 test file | status | port |
|---|---|---|
| `test_isl_spec_yihou.py` | implemented | `tests/test_isl_spec.py` verbatim |
| `test_batch_state_yihou.py` | implemented / adapted | layout test -> pool inventory test; ParallelState import test -> bootstrap + comparison test; added accept-bound test |
| `test_topology_yihou.py` | implemented / adapted | renamed graph fields; gathered-metadata check; parallel-state comparison; default TP4; DSPARK defaults; alias flags rejected |
| `test_comparison_yihou.py` | adapted | `fake-server` budget -> `server-like` |
| (new) `tests/test_scripts.py` | added | verify_point gates, collect_sweep import (trap #17), ARG CHECK, run_profile MAX_STEPS=0 guard (trap #6) |

## README sections

| GLM-5.2 README | status |
|---|---|
| what it does / does not; quick start; where things go; scripts table; knobs; heterogeneous ISL; profiling + five rules; two traps; verifying; published results | implemented (README.md), with DSv4.1 traps found in this port; results in VALIDATION_tool3.md |

## Status counts

Rows above (one per GLM-5.2 feature, plus port additions): implemented: 47, adapted: 30, added: 4, implemented / adapted: 4.
No GLM-5.2 feature is N.A.; the one mechanism replaced outright (`fake-server` initial state) is listed as adapted with its
replacement. GPU evidence: VALIDATION_tool3.md (points a-f all PASS their gates; f reports the measured gap, not a pass/fail).
