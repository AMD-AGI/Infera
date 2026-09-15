# GLM-5.2 scheduler-free internal decode report

## Result
**Requested workload completed successfully on2026-09-09:**16 concurrent requests,70000 initial context tokens/request,10000 useful output tokens/request, expected MTP accept length3.61 including the bonus token. A standalone wrapper calls real SGLang target/draft workers without a Scheduler, HTTP serving, or PD deployment.

| Metric | Full run005 |
|---|---:|
| Useful emitted output tokens |160000|
| Final accounting context per request |80000|
| Complete speculative iterations |2768|
| Synchronized measured wall time |78.174991s|
| Aggregate useful output throughput |2046.690351tokens/s|
| Useful output throughput per GPU (8 GPUs) |255.836294tokens/s/GPU|
| Effective token latency per user |7.817499ms|
| Mean full-iteration latency |28.239905ms|
| Median full-iteration latency |28.209135ms|
| Expected bonus-inclusive accept length |3.61|
| Realized bonus-inclusive accept length |3.613439306|
| Target/draft/draft-extension graph executes |2768 /2768 /2768|
| Raw accept tokens, including terminal excess |160032|
| Terminal excess excluded from useful throughput |32|
| Physical mapped KV token capacity |1281024;80064/request|
| Peak PyTorch allocated memory per rank |255176085504bytes (~237.65GiB)|

All8 rank result JSONs are identical. Bootstrap finiteness, raw-FP8 KV layout, physical capacity and worker sequence-progression assertions passed. Logs contain8 `FlyDSL sparse MLA decode engaged` and8 `gfx950 fused DSA indexer enabled` markers. Successful graph execution is counted, not inferred merely from graph-runner availability.

**Interpretation:** these are synthetic-prefix, simulated-acceptance internal compute results. They are not correctness-valid generated-text throughput, an end-to-end serving measurement, or a production workload claim.3.61 is an expected accepted-token count, not a cache-hit probability.

## Method
The entry `bench/profile_decode.py` uses the packup's pinned `TpModelWorker` and `EAGLEWorkerV2`. `ScheduleBatch` serves only as a fixed-batch data structure; no Scheduler object, admission loop or request transport runs.

Initialization loads real target and NextN draft weights, allocates shared request/allocator state and real target/draft KV pools, initializes attention backends and captures the required batch16 graphs. It reserves page-rounded capacity for all16 requests through80000 tokens plus12 speculative reserve tokens. Target KV, draft KV, DSA index keys and scales are physically initialized using the SIKL reference's finite FP8-byte distribution and FP32 scale initialization. This distribution is synthetic, positive and broad-magnitude, not statistically representative of real-prefill KV. Actual MoE routing is preserved; SIKL's synthetic uniform router override is not copied.

A one-token untimed target+draft forward over the populated prefix creates valid recurrent draft state. After10 untimed complete speculative warmup iterations, bootstrap resets measured context to exactly70000. The loop performs real draft, target verification/sampling and draft extension, then updates GPU/CPU lengths and committed state. It checks worker-returned `new_seq_lens = previous_lens + accept_lens` before terminal useful-output clipping.

The pinned simulation uses `match-expected`, `real-draft-token` and one shared random acceptance coin per batch; every request therefore progresses equally. At3.61 it selects3 or4 bonus-inclusive tokens with expected probabilities0.39/0.61. The recorded histogram contains17120 request-iterations of length3 and27168 of length4, totaling44288 request-iterations. This equals1070 and1698 batch iterations respectively.

Timing includes the complete internal speculative loop, required synchronization, graph-execution counters and wrapper bookkeeping. It excludes model loading, physical prefix initialization, capture, bootstrap, warmup, final file serialization and final cross-rank aggregation. The reported elapsed time is the maximum across ranks. It does not time only bare `model.forward`, and is deliberately not computed by multiplying ordinary-decode throughput by3.61.

Terminal computation emits two excess tokens/request in the final iteration. Full terminal compute is charged,32 excess tokens are not counted as useful output, and no subsequent forward consumes the clipped state. Final80000 is the useful/accounting boundary; internal terminal compute reached80002 before clipping.

## Optimized stack and effective configuration
-8 AMD Instinct MI355X (`gfx950`), nodecrsuse2-m2m-055, existing job126175. No node234/036 was accessed and no allocation was requested/cancelled.
- Image `sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d`.
- SGLang fork `402df1e1e453e1e85ec0f5ac4052d36598cc691a`; AITER `2c71811b32c8ce2e1266aedaec199df7d90f597d`.
- Torch runtime2.9.1+rocm7.2.0.git7e1940d4; Triton3.7.0+amd.rocm7.2.0.git89002410; FlyDSL0.3.2; transformers5.12.1; host-visible driver6.14.14.
- TP8/EP1/DP1/PP1, FP8 e4m3 KV, FlyDSL DSA decode/prefill configuration, AITER top-k, fused AITER allreduce and QK norm/rope, EAGLE steps5/draft6/topk1, memory fraction0.85, batch16 graphs.
- Container env: `AITER_USE_FLYDSL_MOE_SORTING=1`, `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1`, `SGLANG_OPT_USE_TOPK_V2=false`. Acceptance knobs set through pinned Envs API before speculative-module import.
- HiCache/prefix-cache/offload and online scheduling are intentionally absent: all synthetic prefix KV remains resident. This is not a replay of the packup's aggregated AgentX serving experiment.
- Missing tuned GEMM configurations use the pinned stack's logged fallback. No claim that every matrix shape uses a hand-tuned kernel. One-token bootstrap does not exercise or prove long-prefill acceleration.

The exact image archive was rehashed, zstd-tested and loaded on055. Actual containerd storage was inspected from an authorized read-only host-path mount (28TiB filesystem,24TiB available), rather than inferred from Spur namespace root space. Specific imported SGLang modules resolve under `/sglang/python/sglang`; AITER under `/aiter`. The existing `python/pyproject.toml` modification is the packaged Dockerfile's install adaptation, not a new benchmark source patch.

Model directory `/shared_nfs/models/GLM-5.2-MXFP4` was mounted read-only. Config SHA256:`8b46225f7afd7181735c2dd97b925bcb70dfebb72b1a105d4ac0be7583a5c726`; safetensors-index SHA256:`fd42188894abe9196fb70a2113c4fd1d0569b29a307a89c0e18e200111456fc6`. Full weight-file hashes/revision were not established.

## Repeat and differential checks
A separate256-step rerun006 used identical code/config/seed and the same70000 starting context. Its acceptance and context trajectories exactly match the first256 steps of005; all8 ranks agree and all three graph counters equal256.

| Matched first256-step metric | Full005 segment | Repeat006 |
|---|---:|---:|
| Sum of recorded step times |7.185985s|7.183488s|
| Median iteration time |27.996378ms|27.928701ms|
| Final context |70929|70929|
| Useful output tokens |14864|14864|

Cumulative step-time difference is−0.034756%. The repeat's overall loop elapsed is7.184349s and throughput2068.941747tokens/s. This is one bounded repeat, not a confidence interval or statistical-equivalence test. Its shorter context range must not be compared as if it were the entire70000→80000 run.

Reference comparison: SIKL `profile_decode.py:24-47` supplies the physical-prefix/direct-forward technique, but replays one fixed context and has no MTP implementation. The wrapper preserves its relevant storage initialization while using pinned SGLang's actual speculative flow. Eager002 and graph003/004 progressively established worker boot, finite state, length progression and actual graph execution; they do not establish tokenwise eager/graph numerical equivalence. No new model-quality evaluation was run.

## Reproduction and artifacts
Workspace: `/shared_nfs/yihou/playground/glm52_decode_internal_yihou_20260909_1057`.

```bash
ROOT=/shared_nfs/yihou/playground/glm52_decode_internal_yihou_20260909_1057
bash "$ROOT/scripts/run_decode.sh" new_full_run_yihou \
  --batch-size 16 --input-len 70000 --output-len 10000 \
  --accept-length 3.61 --tp-size 8 --warmup-steps 10 \
  --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope \
  --mem-fraction-static 0.85
```

Requires the verified image, external read-only weights, an existing authorized allocation and the owned running container. See root `README.md` and `scripts/create_container_yihou.sh`; stale job/node IDs must be revalidated, not automatically replaced by new allocations. Plain Python spawns TP ranks; do not use torchrun.

- Full evidence:`iterations/005_full_target_yihou/{command.txt,config_yihou.json,runtime.log,result_yihou.json,rank_*_yihou.json,steps_yihou.jsonl}` plus source snapshots.
- Matched repeat:`iterations/006_repeat_segment_yihou/`, `results/repeat_comparison_yihou.json`.
- Environment/archive/model provenance:`iterations/001_environment/`.
- Per-iteration hypotheses/results:`working_process.md` and each iteration README.
-13 CPU accounting/state/CLI tests and4 launcher tests passed; saved logs in `results/`.

Packup was read while still evolving, then revisited after environment preparation. All7993 manifest entries passed, with the manifest unchanged during verification. The complete package's later image-validation correction was incorporated; executable Dockerfile/launcher and manifest hashes were unchanged before the final run. No original repository source, model, packup or other experiment was overwritten.

## Remaining limitations
- Simulation forces acceptance independently of target correctness. Real-draft token IDs and real routing do not make these outputs correctness-valid.
- Synthetic KV distribution and fixed synchronized requests differ from natural prompts/serving. No prefill cost, tokenization, scheduling, networking, cache-hit policy or HiCache transfers are measured.
- Timing includes wrapper/CPU synchronization overhead; optional bare-model profiling could answer a different question but was not substituted for this result.
- Small eager smoke included first-use JIT and is not a valid speedup baseline. This report makes no kernel speedup claim against it or the concurrency1 AgentX packup.
- Graph capture and model load recur when a new process starts; the reusable container retains JIT cache but is not a persistent loaded-model service.
- Shared-memory resource-tracker/barrier-device warnings appeared at shutdown/startup. Processes exited successfully and GPU memory drained; no broader resource-leak fix is claimed.
