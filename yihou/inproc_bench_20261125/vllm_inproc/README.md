# vllm_inproc — vLLM in-process kit

Pinned to the image `infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2` (vllm `0.34.0rc1.dev897+g9c1d45c21`; the
infera DSv4.1 build lineage). Internal APIs used: `LLM.from_engine_args`, `llm_engine.engine_core.engine_core`
(InprocClient -> EngineCore), `scheduler.schedule` / `SchedulerOutput.num_scheduled_tokens`, and
`vllm.benchmarks.forward` (copy of the pinned source in `../research/vllm_forward_src/`).

## engine_step_bench.py — EngineCore step bench (CPU included)

`python3 engine_step_bench.py [EngineArgs] --result-dir R --batch-size C --input-len L --output-len O`

- Sets `VLLM_ENABLE_V1_MULTIPROCESSING=0`, so the EngineCore runs in this process. With TP>1 the GPU workers are
  still separate processes.
- Adds C requests at once and times each `LLMEngine.step()`. A wrapper on `scheduler.update_from_output` records, per step, the batch that step
  COMPLETED (scheduled tokens, requests, how many still prefilling); with async scheduling the EngineCore runs a
  batch queue, so a step can schedule batch i+1 while returning batch i, and attribution by scheduling would be
  off by one step (observed: step 0 = 1 ms schedule-only, step 1 = 0.31 s prefill), so the chunked-prefill steps and the decode
  steady state are separated exactly (`classify`, `steady_window`).
- Output tokens come from DELTA outputs, so speculative steps that emit k tokens per request are counted correctly.
- `--profile-steps A:B` writes a torch.profiler trace of steps [A, B) on the engine-core side. TP>1 workers are not
  traced.
- DP>1 is rejected. For DP/EP topologies use forward_bench under torchrun, or the serving stack.

## forward_bench.py — wrapper of `vllm bench forward`

Builds exact batches without the scheduler. Times target forward + sampler + drafter as a one-shot CUDA graph. Cases
use the upstream grammar `NAME=[N*]Q[@K][c][:D]`; `--case-file presets/dsv41_agentx_cases.txt` gives DSpark verify
rows (`6@K:5`), plain decode, fresh prefill, chunked prefill and a mix. Upstream defaults: `--load-format dummy` with
zipf routing (pass `--load-format auto` for real weights and real routing), KV zeros, kv-len jitter ±50 %
(`--kv-len-jitter 0` for exact lengths). For more than one GPU, set `TORCHRUN_NPROC=N` on `run_point.sh`.

## infer_llm.py — offline LLM

`generate --chat --prompt ...` / `throughput --num N --isl L --osl O`. Accepts any EngineArgs, spec decode included.

## Presets

- `dsv41.env`: llying's decode/prefill env without the PD/Mooncake knobs.
- `dsv41_tp2.args`: two GPUs (TP2; one GPU OOMs loading the 475 GB checkpoint), aiter MoE, mxfp4 indexer, fp8 KV, block 128, engram CPU offload, 16k batched tokens.
- `dsv41_dspark_synth351.args`: DSpark 5 with synthetic acceptance 3.51, as on llying's decode leg.
- `dsv41_agentx_cases.txt`: the forward cases.
