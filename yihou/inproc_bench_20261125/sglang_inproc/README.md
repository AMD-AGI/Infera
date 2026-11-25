# sglang_inproc — SGLang in-process kit

Pinned to the image `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111` (sglang `0.5.23.dev20261111+g6be0d75ae0`).
The internal APIs used here (`sglang.benchmark.one_batch.load_model / extend / decode`, `ScheduleBatch.merge_batch`,
`runtime_context.publish`, `arg_groups.overrides.declare_resolution`) change between SGLang versions. Before trusting
the kit on another image, re-read them in that image (copies of the pinned versions are in `../research/sglang_src/`).

## step_bench.py — ModelRunner step bench

`python3 step_bench.py [any ServerArgs flag] --result-dir R --batch-size C --input-len L --output-len O [--prefill-group G]`

1. Builds the model the same way `one_batch` does: `ModelRunner`, memory pools, attention backends and CUDA graphs.
   The decode graph is captured for exactly the local batch.
2. Runs an untimed warm-up point.
3. Runs the timed point: real prefill of C/dp random-token requests per DP rank (in groups of G merged into one
   `ScheduleBatch`), then O-1 greedy plain-decode steps, each with a device sync before and after.
4. Every rank writes `rank_<r>.json`. Rank 0 gathers the per-step times, takes the max over ranks, and writes
   `steps.jsonl` and `result.json`.

Owned flags (rejected if passed): `--cuda-graph-max-bs-decode` and `--cuda-graph-bs-decode`. All `--speculative-*`
flags are rejected because this bench does not drive speculative decoding. A point stops early with
`KV capacity ... < needed` when `C/dp * (L + O)` does not fit the pool.

Profiling: `--profile --profile-start-step S --profile-num-steps N --profile-ranks 0|all` writes the torch.profiler
trace plus a top-ops table to `profile/`.

## infer_engine.py — offline Engine

`python3 infer_engine.py [ServerArgs flags] generate --chat --prompt "..."` or
`... throughput --num N --isl L --osl O --result-dir R`. This is the full engine, including spec decode. For #5034
spec behaviour, add `presets/dsv41_dspark.args` and recreate the container with
`ENV_FILES="presets/dsv41.env presets/dsv41_dspark_sim351.env"`.

## Presets

- `dsv41.env`: the #5034 env without serving-only knobs.
- `dsv41_tp4_dpa4_ep4.args`: the #5034 topology.
- `dsv41_dspark.args` + `dsv41_dspark_sim351.env`: DSpark with simulated acceptance 3.51 (`infer_engine` only).
