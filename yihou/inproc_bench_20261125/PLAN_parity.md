# Plan: SGLang tools 2 and 3 with strict functional parity to the GLM-5.2 tools (user, 2026-11-25 ~13:40Z)

User requirement: tool 1 (fake-input server) may stay absent; tools 2 and 3 must be functionally STRICTLY the same as
the GLM-5.2 ones, for DeepSeek-V4.1-Flash on MI355X, on the SGLang image already validated here
(`lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111`, sglang 6be0d75ae).

## References (read-only)
- Tool 3 (ours, GLM-5.2): `/home/yihou/dev/git.16-19/infera.dev.yihou.sglang.bench.fast.script/sglang_decode_internal_bench_and_profiling/`
  (bench/ scripts/ tests/, README) + KB handoff `kbk-tool-inprocess-step-bench` (build history, design decisions,
  equivalence evidence) + packups under that repo's `packups/`.
- Tool 2 (teammate, AKLR @33689332): `research/aklr_src/` (copies of `aklr/benchmarks/models/{sglang_bench,kernel_breakdown,main}.py`,
  `glm52/`, `deepseekv4/`) + KB handoff `kbk-tool-perop-microbench` (mechanism table, checklist, traps).
- Overview: KB `kbk-decode-bench-tools-overview`; task `decode_bench_tool_build` (validators = the bar).

## Deliverables (in this kit, `sglang_inproc/`)
- `tool3_spec_step/`: port of the GLM-5.2 in-process whole-step spec-decode bench: target `TpModelWorker` + the
  DSv4.1 spec worker (DSPARK V2, block 5) built in the scheduler's order, synthetic physically-valid KV (all DSv4.1
  pools: SWA / C4 / C128 / C4-indexer + engram host table as the server builds it), one-token bootstrap, warmup+reset,
  simulated acceptance via the engine's own hook (accept length / method / token mode), full OSL loop, uniform and
  heterogeneous ISL (`--input-len-spec`), TP/EP/DPA topology module, profiling extension (torch.profiler window,
  DeviceTimer, capture inventory, graph-visibility probe), compare_server + check_compare_paths, runners
  (create_container, run_decode, run_profile, sweeps), verify_point / collect_sweep gates, equivalence_smoke, CPU tests.
- `tool2_perop/`: port of the AKLR per-operator micro-bench to v0.5.23 + DSv4.1, glm52 feature set: decode + prefill
  entry points, record/replay per module in its own CUDA graph, cold-weight rotation, seeded balanced routing, layer
  kinds + roll-up + ratio, Coverage / Not replayable, `--full-step`, `--trace`, Kineto `kernel_breakdown` targets,
  `--export-only` / shape export, MTP (DSPARK) path like glm52's EAGLE path (draft model one row), report sections.
- `PARITY.md`: one row per GLM-5.2 feature (flag, output file/field, script, gate, test) -> implemented / adapted
  (how) / not applicable (why, first-hand), with the GPU evidence that exercises it.

## Order
1. Research inventories (feature lists of both GLM-5.2 tools; DSv4.1 model + DSPARK worker + KV pools + one_batch API on
   the pinned image).  2. Port tool 3, then tool 2 (parallel teammates).  3. GPU validation on openai n03-33
   (GPUs 0-3 / 4-7 split), incl. the equivalence check against the real #5034 server at the same shape (median ITL),
   as both GLM-5.2 tools did.  4. PARITY.md, VALIDATION.md, push to origin/dev/yihou/inproc_bench.

## Open with the user
- Does "tools 2 and 3" also cover the vLLM kit (a vLLM per-op micro-bench and a vLLM whole-step spec bench), or SGLang only?
  Proceeding with SGLang (unambiguous) meanwhile.
