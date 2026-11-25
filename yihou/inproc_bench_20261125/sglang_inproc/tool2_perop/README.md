# Tool 2 — per-operator decode/prefill micro-bench, DeepSeek-V4.1-Flash on SGLang (MI355X)

Port of the teammate's AKLR per-operator micro-bench (`aklr/benchmarks/models/{glm52,deepseekv4}` @ 33689332,
read-only copies in `../../research/aklr_src/`) to DeepSeek-V4.1-Flash on the pinned image
`lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111` (sglang 6be0d75ae). Feature set = the GLM-5.2 port's,
plus what the deepseekv4 port added (CONC sweep, capacity refusal, Kineto top kernels, export-only).
Parity table: `PARITY_tool2.md`. License: Apache-2.0 with the AMD header on every adapted file; each file
names what it was adapted from.

It loads the model through SGLang's own `ModelRunner` (no server/scheduler), builds a synthetic decode (or
prefill-chunk) batch, records every operator's real inputs during one forward (module forward pre-hooks, plus
call-site wrappers where DSv4.1 never calls a module), replays each operator alone inside its own CUDA graph,
multiplies one representative per layer kind by the kind's layer count, and checks the sum ("roll-up")
against a full forward: `ratio = roll-up / full forward`. Near 1 means the totals agree — not that every
operator was covered or that the split inside a layer is right. All inputs are synthetic (random token ids,
random KV bytes, seeded balanced routing): performance measurements, never correctness evidence.

## Layout

| file | holds |
|---|---|
| `dsv41/profile_decode.py` | decode entry: per-op graph timing, roll-up, report; `--CONC` sweep, `--full-step N`, `--trace`, `--export-only`, `--no-kernels`; MTP (DSPARK) when `--speculative-algorithm DSPARK` |
| `dsv41/profile_prefill.py` | prefill entry: every chunk of one request's synthetic TTFT; `--prefix-cache-hit` |
| `dsv41/model_bench.py` | DSv4.1 parts: layer kinds, op tree, recording/replay, call sites, batches, capacity, synthetic KV, memory |
| `dsv41/spec_dspark.py` | DSPARK worker built in the scheduler's order, real-prefill bootstrap, step driver, draft ops, phases |
| `sglang_bench.py` | shared: forward ctx, graph/eager timing, balanced routing, lockstep/agreed, versions, tables |
| `kernel_breakdown.py` | Kineto per-kernel breakdown calibrated to graph time; `Ranking.targets()` = top-5 targets |
| `shape_export.py` | `AKLR_EXPORT=1` shape export (AKLR's recorder when `aklr` is importable; else our own JSONL) |
| `main.py` | e2e stage: real text, real chunked prefill, 256 teacher-forced decode steps -> `{"tokens_per_s","latency_ms"}` |
| `presets/dsv41_tp4.args` | plain TP4 flags (the #5034 preset minus attention DP / EP) |
| `scripts/run_t2.sh` | one point in the kit container with provenance (command/env/versions/kit hashes/inspect, redacted) |
| `tests/test_tool2_cpu.py` | CPU-only tests of the pure logic |

## Usage

Container: `../scripts/create_container.sh` (env `CONTAINER=yihou-inproc-t2 GPUS=4,5,6,7 KIT_DIR=<copy the
docker daemon can read> MODEL_DIR WORK_DIR CACHE_DIR`). Then, from `scripts/` on the node:

```bash
P=../../presets/dsv41_tp4_dpa4_ep4.args       # InferenceX #5034: TP4 DPA4 EP4
D=../../presets/dsv41_dspark.args             # + DSPARK block 5
SIM="SGLANG_SIMULATE_ACC_LEN=3.51 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token"

ARGS_FILES=$P ./run_t2.sh NAME decode  --ISL 1024 --CONC 16 64 --reps 10 --repeat 2    # per-op report per CONC
ARGS_FILES=$P ./run_t2.sh NAME decode  --ISL 1024 --CONC 16 64 --full-step 32          # whole steps, sglang graphs
ARGS_FILES=$P ./run_t2.sh NAME decode  --ISL 1024 --CONC 16 --trace                    # one eager step's kernels
ARGS_FILES=$P ./run_t2.sh NAME decode  --ISL 1024 --CONC 16 64 --export-only           # shapes only
ARGS_FILES=$P ./run_t2.sh NAME prefill --ISL 32768 --reps 3 [--prefix-cache-hit 0.5]
ARGS_FILES="$P $D" EXTRA_ENV="$SIM" ./run_t2.sh NAME decode --ISL 1024 --CONC 16 --reps 5   # MTP: per-op pass, then implied --full-step 3
ARGS_FILES=../presets/dsv41_tp4.args ./run_t2.sh NAME decode --ISL 1024 --CONC 16
```

Inside the container the entry points can also be run directly (`cd /kit/sglang_inproc/tool2_perop/dsv41 &&
python3 profile_decode.py --model-path ... <ServerArgs flags> --ISL ... --CONC ...`). `ServerArgs` flags pass
through, except: `--disable-cuda-graph` is forced (hooks need SGLang's graphs off) unless `--full-step`;
`--kv-cache-dtype auto` becomes `fp8_e4m3`; under `--full-step` the decode graph's max batch is declared as the
local batch (one_batch's mechanism) unless `--cuda-graph-max-bs-decode` is given. The 475 GB checkpoint needs
>= 2 GPUs.

Bench-own flags:

| flag | default | script | meaning |
|---|---|---|---|
| `--ISL` | 4096 | both | input length per request |
| `--reps` | 50 | both | graph replays per operator |
| `--repeat` | 1 | both | timing passes, fastest wins |
| `--out` | auto | both | report path (default: auto-named in `--result-dir`) |
| `--result-dir` | cwd | both | report / trace / export directory |
| `--CONC` | 1 | decode | **total** concurrency, split across attention-DP ranks; several values share one model load; `[UNSUPPORTED]` when not divisible, `[BLOCKED_CAPACITY]` when it does not fit (never reduced) |
| `--full-step N` | 0 | decode | time N+2 whole steps under SGLang's own graphs, synced per step, fastest wins, then exit; spec: phase split via CUDA events |
| `--trace` | off | decode | one eager step's kernels (top 35 printed) + chrome trace (rank 0) instead of timing ops |
| `--export-only` | off | decode | one eager step per CONC with `AKLR_EXPORT=1`, no timing, no report |
| `--no-kernels` | off | decode | skip the Kineto per-kernel breakdown |
| `--prefix-cache-hit` | 0 | prefill | fraction of `--ISL` already cached |
| `--kv-fill` | random | both | `prefill`: decode context written by one real prefill forward instead of random bytes |
| `--routing` | balanced | both | `native`: keep the model's router (diagnostic for server comparison; not the bench's design) |

## Reading a report

Rows are `kind|op`, summed as per-layer cost x layer count; global rows run once per forward.

| row | contents |
|---|---|
| `attn.wqkv_a` / `attn.wq_b` / `attn.wo_b` | fused q/kv down-projection, q up-projection, output projection (FP8 block GEMMs) |
| `attn.sources` | DSv4.1 ratio-1/2 layers' kv-source compressor + indexer, run by the attention backend's `forward_low_ratio_sources` (the modules themselves are in Coverage) |
| `attn.<rest>` | q/kv norm + rope, fused `wo_a`, KV write, sparse attention kernel |
| `mlp.topk` / `mlp.experts` / `mlp.shared_experts` | router (gate fused into topk), FP4 routed experts, shared expert |
| `mlp.<rest>` | shared+routed add and any post-expert collective (a difference; can read negative) |
| `layer.<rest>` | the mHC boundary (hc_post + hc_pre + both RMSNorms, one fused kernel on gfx950), MoE row transport (DP gather/scatter) and collectives |
| `engram` | the Engram gate of layers 1 and 14, run from the model loop before the layer (outside `layer`) |
| `embed_tokens`, `engram_hasher`, `hc_head`, `norm`, `logits` | global; `hc_head` is the final mHC combine (a function call site) |
| `draft model` | MTP only, one row: the draft forward(s) of one step, timed in place; per-op draft rows are a separate table |

Layer kinds (Flash, 40 layers), from the built model: `moe+c0` (0), `moe+c0+engram` (1),
`moe+c2+kvsrc+indexer` (2, 8), `moe+c2` (15), `moe+c2+kvsrc+indexer+engram` (14),
`moe+c1+kvsrc+indexer+late` (20), `moe+c1+late` (15), `moe+c1+indexer+late` (24, 28, 32, 36). `cN` is the
compression ratio, `kvsrc`/`indexer` mark layers owning a compressor/indexer, `late` marks layers at or past
the decoder SWA bounded-replay start (prefill runs them on the tail rows only). No hash-routed layers on
this checkpoint.

**Coverage** lists hooked modules never called: `input_layernorm` / `post_attention_layernorm` (applied inside
the fused mHC boundary), `attn.wo_a` (fused), `mlp.gate` (inside topk), `attn.compressor` / `attn.indexer`
(run through the backend; timed as `attn.sources`). **Not replayable** lists ops that ran but failed alone,
and where their cost sits. **Top kernels** groups Kineto kernels by innermost API (no `aklr` routes in this
image), calibrated to graph time; `target` marks the top 5 non-collective groups plus ties.

DSv4.1-specific adaptations (details in `PARITY_tool2.md`): on gfx950 the model loop runs each layer as
`forward_layer_fused_boundary(model, i, HcState)` and never calls `layer(...)`, so the `layer` row is recorded
by wrapping that call site; recorded inputs are snapshotted (tensors inside `HcState` included) before the
layer mutates them; every scoped forward flag and the active `ForwardContext` are replayed with each op.

## Validation

See `PARITY_tool2.md` ("GPU evidence") for the points run on smci355-ccs-aus-n03-33, GPUs 4-7 (point dirs with
provenance under `/data/yihou/inproc_bench_20261125/work_t2/iterations/` on that node). The markdown reports of
the passing points are copied to `results_20261125/`. Code changes after a point ran (draft-table share column,
Coverage pointing compressor/indexer at `attn.sources`) are cosmetic and were not re-run for every point.

## Open

- Decode C64 (TP4 DPA4 EP4): the per-op pass's full forward (one graph of `model.forward`, built under the
  forced `--disable-cuda-graph` config) is 15.80 ms at ISL 1024 and 15.93 ms at ISL 8192, while `--full-step`
  (sglang's own decode graph) is 13.71 / 13.80 ms; at C16 they agree (9.91 vs 9.92 ms). Balanced vs native
  routing moves `--full-step` by < 0.4 %. Not explained; the measurement that would answer it is a kernel trace
  of the same C64 forward in both configurations. Until then read C64 per-op rows against the per-op full
  forward, not against server TPOT.
- Prefill at 16384 rows/rank (the #5034 `--chunked-prefill-size 65536` over DP4) aborts inside a standalone
  `layer` replay with `hipErrorLaunchOutOfResources` / `HSA_STATUS_ERROR_OUT_OF_RESOURCES` (t2_p1, t2_p1c);
  4096 rows/rank (`--chunked-prefill-size 16384`) runs clean. With `HIP_LAUNCH_BLOCKING=1` the 16384-row chunk
  completed, and with `--mem-fraction-static 0.7` (t2_p1d) the 16384-row chunks run clean (TTFT 874.6 ms at ISL
  32768, ratio 0.998). Use `--mem-fraction-static 0.7` for #5034-size prefill chunks; why 0.85 fails is not established.
- `mlp.<rest>` reads negative (the `mlp` parent replayed alone is shorter than its children's sum); it is a
  difference of two measurements and is not investigated.
- DSPARK draft stage `layer` rows are not replayable standalone (24 vs 20 rows); their cost stays in the
  `draft model` row (glm52 had three unreplayable draft ops too).
- Shape export sees aiter callables bound into sglang module namespaces and dispatcher ops; Triton/gluon kernels
  launched from functions imported inside sglang functions are not recorded.
- No per-row correctness check of attribution exists (upstream TODO); `ratio` cannot catch a mis-split.
