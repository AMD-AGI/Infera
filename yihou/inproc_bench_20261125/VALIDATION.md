# Validation — GPU runs on smci355-ccs-aus-n03-33 (8x MI355X), 2026-11-25

Model DeepSeek-V4.1-Flash (475 GB checkpoint, local copy). Images: SGLang
`lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111` (sha256:6a0b7952...), vLLM
`infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2` (sha256:51ceb17d..., vllm 0.34.0rc1.dev897+g9c1d45c21).
Points live in `/data/yihou/inproc_bench_20261125/work/iterations/` on that node. Every passing point carries the
provenance set (command/env/versions/kit hashes/container inspect/runtime log/launch status).

## Passing points (gates: `common/verify_point.py`)

| point | tool | setup | result |
|---|---|---|---|
| sgl_s3_c16_1k | sglang step_bench | TP4 DPA4 EP4 (#5034 preset), C16 (4/rank), ISL 1024, OSL 32 | prefill 114k tok/s; TPOT 9.77 ms (p10/p90 9.51/10.04); 409 tok/s/GPU |
| sgl_s4_c64_8k | sglang step_bench | same, C64 (16/rank), ISL 8192, OSL 128, prefill groups of 4 merged | prefill 142k tok/s; TPOT 13.06 ms (p10/p90 12.32/13.22); 1225 tok/s/GPU; decode graph executed 127/127 steps on every rank |
| vllm_s5_tp2_c64_8k_o1k | vllm engine_step_bench | TP2, DSpark 5 synthetic acceptance 3.51, C64, ISL 8192, OSL 1024 | 34 prefill steps (16k tokens each) at 54.5k tok/s; 256 steady steps, step 32.5 ms (p10/p90 32.2/33.0); realized 3.5075 tokens/req/step; TPOT 9.28 ms; 3449 tok/s/GPU |
| vllm_fwd1_tp2 | vllm forward_bench (torchrun x2) | TP2, real weights, DSpark drafter included | verify 64x6@32768:5: 29.58 ms (spread 0.13 %); prefill 16384: 278.6 ms (spread 0.03 %) |
| vllm_infer_gen2 | vllm infer_llm generate | TP2, no spec, chat | "17*23" -> `391`; "capital of France" -> `Paris` (both with reasoning prefix) |
| sgl_infer_gen1 | sglang infer_engine generate | TP4 DPA4 EP4 preset, no spec, chat | "17*23" -> `391`; "capital of France" -> `Paris` (reasoning prefix; gate flags `complete` missing: run predates the generate `complete` field) |
| vllm_infer_tput1 | vllm infer_llm throughput | TP2, DSpark synth 3.51, 64 x ISL 8192 / OSL 1024 | 65536 tokens in 19.59 s = 3345 tok/s |

## Cross-checks between tools

- vLLM prefill: forward_bench 16k-token prefill forward 278.6 ms vs engine_step_bench 16k-token prefill steps
  ~286-301 ms (scheduler + CPU + sampler on top of the forward).
- vLLM whole workload (64 x 8k/1k, same engine flags): engine_step_bench wall 18.81 s vs infer_llm throughput
  19.59 s (+4 %, the LLM path adds the multiprocess EngineCore + output processing).
- vLLM realized acceptance 3.5075 tokens/req/step vs configured synthetic 3.51.

## Bugs found and fixed during validation (each one is a reason the gates exist)

| symptom | cause | fix |
|---|---|---|
| sglang ranks crash `'int' object has no attribute 'startswith'` under DPA | `one_batch`-style integer request ids; DP-attention sync calls `rid.startswith(...)` | string rids `bench-<i>` |
| sglang decode 81 ms/step at C16 (sgl_smoke2) | ranks recomputed dp from the resolved ServerArgs view (dp=1) and ran 16 req/rank while the decode graph was captured for 4 -> eager decode | dp/local batch passed from main; per-rank graph-coverage flag; graph `execute` counter; gate fails on replays < steps and warns when unverifiable |
| vLLM TP1 OOM in MoE weight shuffle | 475 GB checkpoint > one 288 GB GPU | presets use TP2 |
| vLLM prefill 17M tok/s (vllm_smoke3) | async scheduling batch queue: step i schedules batch i+1, returns batch i; attribution by `schedule()` is off by one | attribute by `update_from_output` (completed batch); time of schedule-only steps carried forward; unit test with a fake batch-queue engine |
| vLLM step 0 = 1 req / 6 tokens in the timed run (vllm_s4) | warmup's last verify batch still in the batch queue | drain the batch queue after every run |
| `--seed` argparse conflict | kit flag vs EngineArgs `--seed` | `--data-seed` |
| failed run recorded exit 0 | `cmd | tee | grep || true` reset PIPESTATUS | no `|| true`; status of `docker exec` |
| `--api-key=...` not redacted | name regex matched `API_KEY` only | `API[_-]?KEY` (CPU test) |

## Not yet validated / open

- `sglang infer_engine throughput` and the DSPARK spec path through infer_engine: not run.
- `engine_step_bench --profile-steps`, `step_bench --profile`, `forward_bench --profile`: code paths not run on GPU.
- vLLM forward_bench reports graph-vs-eager relative RMS 1.36 (verify) / 1.05 (prefill) for this model. Upstream
  prints it; not investigated here.
- SGLang step_bench drives plain decode only. DSPARK spec decode is not driven (use infer_engine).
