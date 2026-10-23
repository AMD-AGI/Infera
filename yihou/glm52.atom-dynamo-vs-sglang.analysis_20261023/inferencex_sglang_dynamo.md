# InferenceX SGLang+Dynamo GLM-5.2 vs our MI355X kit (workflow wf_fb75c5df-ae9, 2026-10-23)

# CRITIC (final)

**Review of the InferenceX SGLang+Dynamo GLM-5.2 draft (InferenceX ae21092e, srt-slurm ae3d6877, API dumps in /var/tmp/yihou/ix_api/)**

## 1. Wrong or unsupported claims

1. **Master config path.** The draft gives `configs/nvidia-master.yaml:7549-7666`. The file is at `inferencex-e2e/configs/nvidia-master.yaml:7549-7666`; the line range itself is correct.
2. **B300 agg c48 has a different provenance.** The draft implies it shares the GB300 image and run. It does not:
   - Run 38356572105, dated 2026-10-14.
   - Image `nightly-dev-cu13-20260901`.
   - Recipe `b300-fp4/agentx/agg-tp4-variants.yaml:184-196`, not #4300.
3. **Two decode settings in B(3) apply only to c264.**
   - `mem-fraction-static 0.94` is set only at c264 (recipe L276). The base decode value is 0.90 (L118), and c216 uses it.
   - HiCache 185 GB is also c264-only (L261). The base and c216 set no `hicache-size`.
   - c216 prefill uses chunk 16384 and mem 0.83 (L62-63, L70). c264 uses 8192 and 0.88 (L258-260).
   - So the step from 55.5K to 63.6K changes five settings at once (concurrency, prefill chunk, prefill mem, HiCache size, decode mem). It does not isolate any one of them.
4. **Our prefill settings already match c264.** Our S1 prefill uses chunk 8192 and mem 0.88 (`S1.argv.txt:168-173`), the same as c264. The draft's statement "ours inherits 32768" is true only for decode (`S1.argv.txt:341-342`), not L357; L357 is `--speculative-algorithm`.
5. **Wrong argv line numbers for decode EP1.** The draft cites L131 and L383. L131 is in the prefill block, and the file has only 379 lines. Decode `--ep-size 1` is at L333-334, `--max-running-requests 256` at L343-344, and graph bs 256 at L372-373.
6. **Wrong path prefix.** `fields/memory.py` and `fields/serving.py` are at `python/sglang/srt/arg_groups/fields/...`.
7. **SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION is used on the PD-decode path.** The draft says this was not verified. A code read shows it is used:
   - Read at `disaggregation/decode.py:141`.
   - Used in admission sizing at `decode.py:483-498` (`_required_admission_tokens`), `decode.py:1540-1543` and `decode.py:1984-1990`.
   - Whether it actually limits admission at our KV occupancy is still unknown. Decode-queue and KV-usage metrics would settle it.
8. **KV events are not passed when unset.** The draft says the recipe's KV events are unknown. In srt-slurm at HEAD ae3d6877, `role_kv_events` returns None when `kv_events` is unset (`backends/base.py:140-146`), so no `--kv-events-config` is passed (`backends/sglang.py:422-427`). What remains unknown is which SHA was actually used and how the Dynamo KV router behaves without events.
9. **One prefill PP4 worker per node.** #4300's PR body ties this to sgl-project/sglang#39378, and GB300 has 4 GPUs per node (recipe L13). Each PP4 prefill worker is therefore inside one node, as ours is. The DEP8 decode spans 2 nodes over MNNVL (L263-264).

## 2. Missed items

- **Decode KV usage was saturated at the 50K points [first-hand, API].** `gpu_kv_cache_usage_pct` is 1.0 at c216 and c264, 0.96 at c256, and 0.78 at c56. Which role this metric comes from is unknown; `request_metrics` or the collector source would settle it.
- **Output tokens [first-hand].** At c264, mean `output_tokens_actual` is 1,005 and `output_tokens_expected` is 1,293. Our ratio has not been read.
- **Cache hit [first-hand].**
  - GB300: server GPU hit 0.64 plus CPU hit 0.31, against a theoretical 0.970.
  - ATOM MI355X c48: GPU hit 0.93.
- **The ATOM 34,195 point uses DCP4 [first-hand, API].** It is agg TP4 with `dcp_size 4`, c48, image nightly_202609211553.
- **Synthetic acceptance matches.** InferenceX sets the same three variables we set: `SGLANG_SIMULATE_ACC_LEN`, `SGLANG_SIMULATE_ACC_METHOD=match-expected` and `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token` (`synthetic_acceptance.py:36-38,219-228`). Ours are at `worker.sh:107-108`.
  - The AL 3.61 comes from `golden_al_distribution/glm5.2_mtp.yaml` (5 steps, thinking_on). That file says it was measured on glm-5.2-fp8 B300 vLLM.
- **Window definition [first-hand].** The window runs from the first request start to the last request end over completed records (`request_metrics.py:213-241`).
- **The B300 agg c48 point (42K, 4 GPUs, no PD) is the closest single-node comparator** (`agg-tp4-variants.yaml:13-196`). Its settings:
  - `enable-dp-lm-head`; `enable-prefill-delayer` with 60 passes.
  - Max-running 128 (32 per rank), graph bs 64, chunk 32768.
  - HiCache 270 GB with `page_first_direct`; kv fp8; `bf16-gemm-backend cutedsl`.
  - `AGENTIC_WARMUP_GRACE_PERIOD 1800`.
  - Both flags exist in our branch (`arg_groups/fields/parallel.py:199`, `schedule.py:228`); S1 uses neither.
- **KV events differ by role on our side.** Prefill publishes them (`S1.argv:178-182`); decode has them off (L376-378).
- **The #4300 PR body states the dram-utilization reason.** It was raised to 0.85 so the 185 GB HiCache plus the DSA-indexer host pool fit.

## 3. Corrected A-C

### A. Which points reach ~50K, and how comparable they are

**The points [first-hand]**

- Source: API dump `/var/tmp/yihou/ix_api/glm52_latest.dec.json`.
- GB300 points: run 39187209967 attempt 2 (2026-10-20), image `nightly-dev-20261016-a93f16f0`, Dynamo 1.6.0.dev20261016, NIXL, recipe #4300 bc6a0446.

| Point | GPUs (P+D) | tok/s/GPU | TTFT | TPOT | GPU/CPU cache hit | KV use |
|---|---|---|---|---|---|---|
| GB300 3×PP4 + 1×DEP8, c216 | 12+8 | **55,488** | 3.15 s | 10.8 ms | 0.69 / 0.26 | 1.0 |
| GB300 3×PP4 + 1×DEP8, c264 | 12+8 | **63,647** | 4.34 s | 11.1 ms | 0.64 / 0.31 | 1.0 |
| GB300 4×PP4 + 1×DEP16, c256 | 16+16 | 44,453 | 2.42 s | 8.8 ms | 0.72 / 0.23 | 0.96 |
| B300 agg TP4 DP4-attn, c48 (run 38356572105, image cu13-20260901) | 4 | 42,047 | 6.41 s | 19.0 ms | 0.52 | n/a |

**How comparable the number is**

- **Metric [first-hand]:** same definition as ours (`request_metrics.py:213-241`). Check: 1,272,944 / 20 = 63,647.
- **Acceptance [first-hand]:** both sides use the same simulated AL 3.61 with the same three variables. Neither number is a correctness result.
- **Not comparable on these axes:**
  - Hardware and precision: GB300 NVFP4 vs MI355X MXFP4.
  - GPU count: 20 vs 8.
  - Window: about 3,630 s vs 1,800 s.
  - Client: aiperf vs InferenceX client 918524ff.
  - Warm-up: unknown on their side (the GB300 recipe sets no `AGENTIC_WARMUP_GRACE_PERIOD`) vs 10 per lane on ours.
- **MI355X reference points [first-hand]:**
  - SGLang agg TP4 EP4, v0.5.20: 18,327 at c16.
  - ATOM agg TP4 DCP4: 34,195 at c48.
  - There is no published SGLang MI355X disagg point.

### B. Differences from our S1 (34,509 tok/s/GPU)

None of the effects below has been measured.

**(1) Hardware and kernels [first-hand]**
- Theirs: MNNVL, `modelopt_fp4`, `flashinfer_cutedsl` and `flashinfer_trtllm` (recipe L50-51, L58, L66, L76, L113).
- The hardware share of the gap is unknown.

**(2) Topology [first-hand]**

| | Theirs (c264) | Ours (S1) |
|---|---|---|
| P:D GPUs | 12:8, 3 prefill PP4 workers (one per node) | 4:4, 1 prefill PP4 worker |
| Decode | DEP8 across 2 nodes, `moe-a2a flashinfer`, 32 requests per rank (L267-278) | TP4 DP4 EP1, 64 requests per rank, graph bs 256 (argv L333, L343, L372) |
| Concurrency per GPU | 13.2 | 10 |

- `decode_dcp_size` is 1 at all GB300 points.
- On ROCm the a2a option would be `mori` (`layers/moe/utils.py:41`). Whether it works with MXFP4 GLM-5.2 is unknown; a launch test would settle it.

**(3) SGLang flags [first-hand]**
- **Prefill:**
  - Theirs: max-running 16, cuda graph off, `hicache-io-backend direct`, overlap and policy at defaults (L54-86).
  - Ours: 256, `kernel` with `page_first`, `--disable-overlap-schedule`, `shortest-prefill-first` (argv L141, L167, L174, L186-195).
  - Chunk and mem fraction already match c264 (8192 / 0.88).
- **Decode:**
  - Theirs: `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8` (L95). It feeds PD-decode admission (`decode.py:483-498`).
  - Theirs: `stream-interval 30`, `skip-tokenizer-init`, chunk 64, mem 0.90 (0.94 at c264).
  - Ours: chunk 32768, mem 0.92.
- **Same on both sides:** extra-slots is 0, and decode radix is off.
- **Unknown on their side:** kv dtype, page size and DSA backend; the `server_args` log of the run would settle them. Ours uses fp8_e4m3 KV and triton DSA.

**(4) Router [first-hand]**
- **Their routing setup:**
  - Up to 10 frontends with nginx consistent hashing on the session header (srt-slurm `schema.py:2591,2630-2634`, `nginx.conf.j2:18-38`).
  - `router-mode kv` with a prefill session TTL of 3600 s and a decode active-request weight of 8192.
- **KV events:** none are passed at srt-slurm HEAD (`base.py:140-146`). The SHA actually used is unknown; that run's artifacts would settle it.
- **Ours:** Infera with Mooncake. Prefill KV events are on and decode KV events are off.
- **DEP8 DP-rank selection:** unknown; the `dynamo.sglang` logs would settle it.

**(5) Protocol [first-hand]**
- 3,600 s window; mean ISL 138.7K.
- `temperature 1`, `top_p 0.95`, thinking on (L136).
- `dram-utilization 0.85`.
- Our cache-hit rates and actual/expected output ratio: unknown.

### C. Candidates to try on MI355X

The evidence is only that a published recipe uses each item; no effect is predicted.

| # | Item | Evidence | ROCm status |
|---|---|---|---|
| 1 | Decode DEP4 with `mori` | GB300 L267-275 | Launch test needed |
| 2 | P:D ratio above 1:1 | c216/c264 use 12:8 | Needs multi-node RDMA |
| 3 | Decode 32 requests per rank, graph bs 32 | L277-278 | Supported |
| 4 | `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8` | L95; `decode.py:483-498` | Supported |
| 5 | `stream-interval 30`, chunk 64, `skip-tokenizer-init` | L111-115 | Supported; token-ID path in Infera unknown |
| 6 | Prefill max-running 16, overlap and policy at defaults | L64 | Supported |
| 7 | HiCache `direct` (and B300's `page_first_direct`) | L74; B300 L85-86 | Supported (`arg_groups/fields/memory.py:128`) |
| 8 | `enable-dp-lm-head`, `enable-prefill-delayer` | B300 c48 L184-196 | Fields exist |
| 9 | Protocol parity: 3,600 s window, read cache-hit rates | API | n/a |

**One-variable runs** (all from S1, C80, AL 3.61, one change per run):
1. Read S1 cache-hit, KV-usage and output-ratio metrics, then rerun S1 at 3,600 s.
2. CLIP=8.
3. `stream-interval 30`.
4. Decode max-running 128 with graph bs 32.
5. Decode chunk 64.
6. Prefill max-running 16.
7. Prefill overlap schedule back on; then, as a separate run, the default schedule policy.
8. HiCache `direct`.
9. `enable-dp-lm-head`.
10. DEP4 with `mori`, after a launch test.
11. 2-node P:D ratio change, with concurrency scaled proportionally.

Report tok/s/GPU, TTFT, TPOT, cache hit and KV usage for each.

**Remaining unknowns:**
- Their unset SGLang defaults: the `server_args` log of run 39187209967.
- The srt-slurm SHA: that run's `srt-slurm-sha.txt` artifact.
- The c216 HiCache size: resolved `server_args`.
- The source role of the KV-usage metric: InferenceX collector source.

# SYNTHESIS (draft)

# InferenceX SGLang(+Dynamo) GLM-5.2 AgentX: where the ~50K comes from and how it differs from ours

## A. Which points reach ~50K, and how comparable they are

**The points [first-hand]**

Source is the InferenceX API (`/api/v1/benchmarks?model=GLM-5.2`), dumped to `/var/tmp/yihou/ix_api/`. Recipe is #4300 (bc6a0446), file `gb300-fp4/agentx/disagg-variants.yaml`. Master config is `configs/nvidia-master.yaml:7549-7666`.

| Point | GPUs (P+D) | tok/s/GPU | TTFT | TPOT | GPU/CPU cache hit |
|---|---|---|---|---|---|
| GB300 3×PP4 prefill + 1×DEP8 decode, c216 | 12+8 | **55,488** | 3.15 s | 10.8 ms | 0.69/0.26 |
| same, c264 | 12+8 | **63,647** | 4.34 s | 11.1 ms | 0.64/0.31 |
| GB300 4×PP4 + 1×DEP16, c256 | 16+16 | 44,453 | 2.42 s | 8.8 ms | 0.72 |
| B300 agg TP4 DP4-attn, c48 | 4 | 42,047 | 6.41 s | 19.0 ms | 0.52 |

- Image `lmsysorg/sglang:nightly-dev-20261016-a93f16f0`, Dynamo 1.6.0.dev20261016, KV transfer over NIXL.
- Run 39187209967 attempt 2, dated 2026-10-20.

**How comparable the number is**
- **Metric [first-hand]:** identical definition to ours. It is (Σ input including cache hits + Σ output) / wall time / all GPUs (`infx/results/agentic/request_metrics.py:216-241`). Check: 1,272,944 / 20 = 63,647.
- **Acceptance [first-hand]:** simulated on both sides, AL 3.61 with EAGLE 5/1/6 (`infx/srt_slurm/synthetic_acceptance.py:213-221`; ours `kit/engine/worker.sh:107-108`). Neither number is a correctness result.
- **Not comparable on these axes:**
  - Hardware: GB300 NVL72 with NVFP4 (flashinfer cutedsl/trtllm) vs MI355X with MXFP4.
  - GPU count: 20 GPUs vs 8.
  - Window: about 3,630 s vs 1,800 s.
  - Warm-up: client default (unknown) vs 10 per lane.
  - Client: aiperf with Dynamo session header vs InferenceX client 918524ff.
- **MI355X reference points [first-hand]:**
  - InferenceX's own SGLang MI355X result (agg TP4 EP4, v0.5.20) peaks at 18,327 (c16).
  - ATOM's best on MI355X is 34,195.
  - There is no published SGLang MI355X disagg point.

## B. Differences from our S1 (34,509 tok/s/GPU)

I spot-checked our side against `llkit/render/S1.argv.txt`. None of the effects below has been measured.

**(1) Hardware and precision [first-hand]**
- Theirs: GB300 with MNNVL (`MC_FORCE_MNNVL=1`, `NCCL_MNNVL_ENABLE=1`), `modelopt_fp4`, `flashinfer_cutedsl` / `flashinfer_trtllm` kernels (recipe L54-127).
- None of those kernels exists on ROCm.
- The hardware share of the gap is unknown. A same-recipe A/B of B300 agg c48 against MI355X would settle it.

**(2) Topology and parallelism [first-hand]**

| | Theirs (c264) | Ours (S1) |
|---|---|---|
| P:D GPUs | 12:8 (1.5:1), 3 prefill workers | 4:4 (1:1), 1 prefill worker |
| Decode | TP8 DP8 **EP8** with dp-attention, `moe-a2a-backend flashinfer` | TP4 DP4 **EP1** (argv L131, L383) |
| Running requests per decode rank | 32 (graph bs 32) | 64 (graph bs 256) |
| Prefill | PP4 TP1, 20,20,20,18 | same |
| Concurrency per GPU | 13.2 | 10 |

- On ROCm the a2a option is `mori` (`layers/moe/utils.py:41`; `deepep` is also listed there). Whether mori runs with MXFP4 GLM-5.2 on our image is unknown; a launch test would settle it.
- DCP and prefill-CP are not used at the 50K points: the API reports `decode_dcp_size 1`.

**(3) SGLang flags [first-hand, recipe vs our argv]**
- **Prefill:**
  - `max-running-requests 16`; ours is 256.
  - `disable-cuda-graph`; ours is not set.
  - HiCache 185 GB with io `direct`; ours is 250 GB with `kernel` and `page_first` (argv L186-195).
  - Overlap schedule and schedule policy are left at defaults; ours sets `--disable-overlap-schedule` and `shortest-prefill-first` (L167, L174).
  - `context-length 1048576`.
  - All of these are supported in our branch, e.g. `fields/memory.py:128`.
- **Decode:**
  - `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8`; ours uses the default 4096 (`environ.py:620`, read at `schedule_policy.py:87`). Whether it gates PD-decode admission on our path has not been verified by a code read.
  - `stream-interval 30` and `skip-tokenizer-init`; both absent in ours (`fields/serving.py:256,58`).
  - `chunked-prefill-size 64`; ours inherits 32768 (L357).
  - `mem-fraction-static 0.94`; ours is 0.92.
  - All of these exist in our branch.
- **Same on both sides:**
  - Decode radix cache is off. Theirs sets it explicitly; ours is off by the PD default (`pd_disaggregation_hook.py:120-126`).
  - `disaggregation-decode-extra-slots` resolves to 0 on both.
- **Ours only:** `index_share_for_mtp_iteration`.

**(4) Dynamo and router [first-hand]**
- **Their routing setup:**
  - Up to 10 Dynamo frontends behind nginx, with a consistent hash on `X-Dynamo-Session-ID` (srt-slurm `schema.py:2591,2630`, `templates/nginx.conf.j2:18-38`).
  - `router-mode kv`, `router-replica-sync`.
  - Prefill session TTL of 3600 s (L80-83).
  - Decode `router-decode-active-request-weight 8192` (L120-124).
  - Busy thresholds set to `None` (L33-35).
- **KV events:** the GB300 recipe sets no `kv_events`, so what its KV router actually does is unknown. Dynamo router source or the frontend log would settle it.
- **Transfer:** NIXL for theirs, Mooncake over RDMA for ours.
- **Ours:** Infera kv-aware routing with prefill session affinity, `--decode-input-demand`, and guard release at completion.
- **How much routing can matter here [inference]:** with 3P1D and session pinning they have little routing freedom, and our S1 (1P1D) has none.
- **DP-rank selection inside DEP8 is unknown.** It could be the Dynamo router or SGLang's internal balancer; the `dynamo.sglang` logs would settle it.

**(5) Benchmark protocol [first-hand]**
- Duration 3,600 s vs 1,800 s.
- Mean ISL at c264 is 138.7K.
- Client sampling: `temperature 1`, `top_p 0.95`, `enable_thinking` (L136).
- Host DRAM: `dram-utilization 0.85`.
- Our server cache-hit rate has not been read. The `sglang:cache_hit` metrics in our S1 logs would settle it.

## C. Candidates to try on MI355X

Ranked by relevance. The evidence is that the recipe uses each one; no effect is predicted.

| # | Item | Evidence | ROCm status |
|---|---|---|---|
| 1 | Decode EP via a2a (DEP4 with `mori`) | Recipe L101-127. Every 50K+ point is DEP. | Field exists; MXFP4 compatibility unknown |
| 2 | P:D GPU ratio above 1:1 (e.g. 2 PP prefill + 1 DEP decode across nodes, or P6:D2) | 12:8 at c216/c264 | Supported; multi-node needs RDMA |
| 3 | Decode per-rank batch 32 (max-running = 32·DP, graph bs 32) | L220-221 | Supported |
| 4 | Decode `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8` | L95 | Env exists; admission effect on our path needs a code read |
| 5 | Decode `stream-interval 30`, `skip-tokenizer-init`, `chunked-prefill-size 64` | L114-115 | Supported; skip-tokenizer needs a token-ID path in Infera (unknown) |
| 6 | Prefill `max-running-requests 16`, overlap and policy at defaults | L54-86 | Supported |
| 7 | Prefill HiCache io `direct`, default layout | L54-86 | Supported (`fields/memory.py:128`) |
| 8 | 3,600 s window and our cache-hit rates, for protocol parity | API `duration_seconds` | n/a |

**One-variable experiment list**

All runs start from S1, 1,800 s, C80, AL 3.61, one change per run. Report tok/s/GPU, TTFT, TPOT and server cache-hit rate.

1. Read our S1 cache-hit metrics, then rerun S1 at 3,600 s to quantify the protocol effect.
2. Decode: set `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8`.
3. Decode: `stream-interval 30`.
4. Decode: max-running 128 (32/rank), graph bs 32.
5. Decode: `chunked-prefill-size 64`.
6. Prefill: `max-running-requests 16`.
7. Prefill: re-enable the overlap schedule, then the default schedule policy (two separate runs).
8. Prefill: `hicache-io-backend direct`.
9. Decode: EP4 with `moe-a2a-backend mori`. Launch test first.
10. P:D ratio: 2 nodes, 2 prefill PP4 workers plus 1 DEP decode worker. Changes topology only; scale concurrency proportionally.

Remaining unknowns, each with what would settle it:
- SGLang defaults that the recipe leaves unset (kv dtype, page size, DSA backend): the `server_args` log of run 39187209967.
- The actual srt-slurm SHA used: that run's `srt-slurm-sha.txt` artifact.
- The c216 HiCache size: the resolved `server_args` of the same run.

Files: API dumps are in `/var/tmp/yihou/ix_api/` and PR bodies in `/var/tmp/yihou/ix_prs.txt`. Our argv is `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp/yihou_ws/glm52_dcp_pd_20261021/llkit/render/S1.argv.txt`.

# READER results

# InferenceX SGLang(+Dynamo) GLM-5.2 AgentX results: where the ~50K tok/s/GPU comes from

## 1. Sources (first-hand)
- **Published numbers:** the InferenceX public API, `https://inferencex.semianalysis.com/api/v1/benchmarks?model=GLM-5.2` (latest curve per key, 118 rows). Saved to `/var/tmp/yihou/ix_api/glm52_latest.dec.json`.
- **PR bodies:** #4300, #4286, #3953, #3659, #2642, #4277, #4227, #4370, #4394, #4398, #4064. Saved to `/var/tmp/yihou/ix_prs.txt`.
- **Recipes:** InferenceX clone at `ae21092e`.

**How tok/s/GPU is computed** (`inferencex-e2e/infx/results/agentic/request_metrics.py:216-241`, `agentic/__init__.py:223-226`):
- Throughput is (Σ input_sequence_length + Σ output) / (last request end − first request start).
- The full prompt is counted, so cache-hit tokens are included. The total is then divided by all GPUs, P+D.
- Checked against the API rows: 445405 gives 1,272,944 / 20 = 63,647, and 444868 (agg TP4) gives 168,187 / 4 = 42,047.
- This is the same definition as ours (total / 8).

**Measured window:** `duration_seconds` is about 3,600–3,630 s (workflow default `duration: 3600`, `.github/workflows/benchmark-multinode-tmpl.yml:87-91`). Ours is 1,800 s.

**Warm-up:** the non-fast mode passes no `--warmup-requests-per-lane` (`infx/bench/agentic/replay.py:60-63,145`). The client default is unknown. Ours is 10 per lane. The exact client (aiperf) version is also unknown; the run logs of 39187209967 would settle both.

**MTP acceptance:** all InferenceX points use forced acceptance, not real acceptance. `infx/srt_slurm/synthetic_acceptance.py:213-221` injects `SGLANG_SIMULATE_ACC_LEN=<golden AL>`, `METHOD=match-expected`, `TOKEN_MODE=real-draft-token`. The golden AL for 5 steps is 3.61 (`infx/golden_al_distribution/glm5.2_mtp.yaml`). Our `kit/engine/worker.sh:107-108` sets the same three variables.

## 2. Published SGLang(+Dynamo) GLM-5.2 AgentX points (selected)
Dyn = Dynamo. Disagg = P/D disaggregated, agg = aggregated. DEP = DP-attention + EP. Columns P/D and GPUs come from the API row fields.

| HW / prec | Mode | P / D | GPUs | Conc | tok/s/GPU | TTFT mean (s) | TPOT mean (ms) | Date / image |
|---|---|---|---|---|---|---|---|---|
| GB300 fp4 Dyn | disagg | 3×PP4 / 1×DEP8 | 12+8=20 | 264 | **63,647** | 4.34 | 11.1 | 10-20, nightly-20261016-a93f16f0 |
| GB300 fp4 Dyn | disagg | 3×PP4 / 1×DEP8 | 20 | 216 | **55,488** | 3.15 | 10.8 | same |
| GB300 fp4 Dyn | disagg | 4×PP4 / 1×DEP16 | 32 | 256 | **44,453** | 2.42 | 8.8 | same |
| B300 fp4 Dyn | agg TP4 DP4-attn | – | 4 | 48 | **42,047** | 6.41 | 19.0 | 10-14, nightly-cu13-20260901 |
| B300 fp4 Dyn | agg TP4 DP4-attn | – | 4 | 32 | 37,390 | 2.09 | 15.9 | same |
| GB300 fp4 Dyn | disagg | 1×PP4 / 1×DEP8 | 12 | 56 | 28,407 | 1.70 | 8.8 | 10-20 |
| B200 fp4 Dyn | agg TP8 DP-attn | – | 8 | 32 | 20,014 | 2.07 | 16.4 | 10-22, v0.5.22 |
| GB300 fp4 Dyn | disagg | 1×PP4 / 3×TP4 | 16 | 40 | 16,924 | 1.34 | 6.2 | 10-20 |
| GB200 fp4 Dyn | disagg | TP4 / TP4 | 8 | 10 | 8,138 | 1.42 | 4.4 | 10-21, v0.5.22 |
| B300 fp4 SGLang | agg TP8 | – | 8 | 28 | 23,239 | 1.47 | 10.2 | 10-22, v0.5.20 |
| H200 fp8 Dyn | agg TP8 | – | 8 | 12 | 2,769 | 30.9 | 6.3 | 09-18 |

**Server GPU cache-hit rate** (API field `server_gpu_cache_hit_rate`; theoretical ≈0.97 for all rows):

| Point | Server hit rate |
|---|---|
| GB300 c264 | 0.644 |
| GB300 c216 | 0.693 |
| GB300 DEP16 c256 | 0.720 |
| B300 agg c48 | 0.523 |

**Not yet on the dashboard:** PR #4398 (GB200 PP8-prefill, OPEN) reports from run 39765601214:
- 3P1D DEP16 c336: 42,272 tok/s/GPU.
- 2P1D c224: 37,666.
- 1P1D DEP8 c96: 32,057.

**InferenceX's own SGLang MI355X GLM-5.2 numbers** (API, 2026-10-06, `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260930`, agg TP4 EP4 with HiCache `write_through` 180 GB):

| Conc | tok/s/GPU | TTFT (s) | TPOT (ms) |
|---|---|---|---|
| 16 | 18,327 | 2.59 | 16.8 |
| 14 | 18,042 | – | – |
| 10 | 16,664 | – | – |

- The recipe is `benchmarks/single_node/srt-slurm-recipes/glm5.2/sglang/mi355x-fp4-mtp/agentic.yaml:138-149`.
- The MI355X ATOM best is 34,195 (TP4 DCP4 c48, 09-22).
- The DCP4 SGLang MI355X PR #4064 is OPEN with results "TBD", and needs sglang #44322.

## 3. The ~50K points
The points near 50K are **GB300 3P1D c216 (55,488)** and **c264 (63,647)**. **DEP16 c256 (44,453)** and **B300 agg c48 (42,047)** are the next ones down.

- **Commit:** `bc6a0446` (#4300, merged 2026-10-20).
- **Run:** 39187209967, attempt 2 (API `run_url`).
- **Recipe file:** `inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/glm5.2/sglang/gb300-fp4/agentx/disagg-variants.yaml`
  - base: lines 3-157
  - `override_3p1d_pp4_dep8_c216`: lines 223-247
  - `override_3p1d_pp4_dep8_c264`: lines 249-278
- **Master config:** `inferencex-e2e/configs/nvidia-master.yaml:7549-7666`. c216 is at 7617-7633 and c264 at 7634-7649. It also sets `dram-utilization: 0.85`, NIXL and the Dynamo router 1.6.0.dev20261016.

**Configuration as written in the recipe (first-hand):**
- **Prefill** (3 workers, one GB300 node each, 12 GPUs):
  - `tensor-parallel-size 1`, `pipeline-parallel-size 4`, `SGLANG_PP_LAYER_PARTITION=20,20,20,18`.
  - Chunk 16384, or 8192 at c264. `max-running-requests 16`. CUDA graph disabled.
  - `mem-fraction-static` 0.83, or 0.88 at c264.
  - HiCache `write_back`, io `direct`; `hicache-size 185` only at c264, not set at c216.
  - MoE/FP4 GEMM `flashinfer_cutedsl`. EAGLE 5/1/6 (lines 54-86).
- **Decode** (1 worker, DEP8, spanning 2 nodes, 8 GPUs):
  - TP8, DP8, EP8, `enable-dp-attention`.
  - `moe-a2a-backend flashinfer`, `moe-runner-backend flashinfer_trtllm`.
  - `max-running-requests 256`, `cuda-graph-max-bs-decode 32` per rank.
  - `disable-radix-cache`. `mem-fraction-static` 0.90, or 0.94 at c264.
  - `stream-interval 30`, `skip-tokenizer-init`, `disaggregation-decode-extra-slots 0`.
  - `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8`.
  - `MC_FORCE_MNNVL=1`, `NCCL_MNNVL_ENABLE=1` (lines 87-127, 263-278).
- **Routing:**
  - Dynamo KV router with `router-replica-sync` and nginx session affinity on `X-Dynamo-Session-ID`.
  - Prefill workers advertise `router-session-affinity-ttl-secs 3600`.
  - Decode workers advertise `router-decode-active-request-weight 8192` (lines 23-35, 80-83, 120-124).
- **Client sampling:** `enable_thinking: true`, `temperature 1`, `top_p 0.95` (line 136).
- **Observability:** Tachometer and the OTel collector are off.

## 4. Differences from our S1 (34,509, 8× MI355X, P PP4 + D TP4/DPA4)
These are facts read from the recipes. I have not measured which of them cause the gap.

| Item | GB300 c264 | Ours S1 |
|---|---|---|
| GPUs (P:D) | 12 : 8 (1.5:1), across 5 nodes over NVL72 MNNVL | 4 : 4 (1:1), one node |
| Decode | DEP8: DPA8 + EP8 with flashinfer all-to-all, bs 32/rank (256) | TP4 + DPA4, EP1, graph bs 256 |
| Prefill | 3 independent PP4 workers, with session-affinity KV routing across them | 1 PP4 worker |
| Concurrency per GPU | 13.2 | 10 |
| TTFT / TPOT | 4.34 s / 11.1 ms | 5.5 s / 18.3 ms |
| Decode radix cache | disabled | not checked in this pass (see `render/S1.argv.txt`) |
| KV transfer | NIXL over MNNVL | Infera router + our transfer backend |
| Window / warm-up | ≈3,630 s / client default | 1,800 s / 10 per lane |
| Accelerator / precision | GB300, NVFP4, flashinfer/cutedsl kernels | MI355X, MXFP4 |

- Throughput per decode GPU is about 159K for GB300 c264, against about 69K per decode GPU for ours (34,509 × 8 / 4). This is arithmetic on the published numbers, not a measured cause.
- There is no published SGLang MI355X disagg point to compare with directly.

## 5. Open items
- **Hardware share of the gap:** unknown. A same-hardware A/B (B300 agg TP4 DP4 c48 recipe vs. the same on MI355X) would settle it.
- **Our server cache-hit rate:** not checked here. The `sglang:cache_hit` metrics in our S1 run logs would settle it.
- **c216 HiCache size:** `hicache-size` is not set at c216, so it falls to the SGLang default. The resolved `server_args` in run 39187209967 would settle the actual value.

# READER recipe

# InferenceX SGLang+Dynamo GLM-5.2 AgentX: which points reach ~50K, and what the recipes set

## 1. Which points reach ~50K (first-hand)

I pulled the published numbers from the InferenceX API (`/api/v1/benchmarks?model=GLM-5.2&date=2026-10-20..22`) and saved them under `/var/tmp/yihou/ix_glm52/bench_*.json`. All three top points come from run 39187209967 attempt 2 on 2026-10-20, config key `glm5.2-fp4-gb300-dynamo-sglang-agentic-disagg`.

| Point | GPUs (P+D) | tok/s/GPU | TTFT mean | TPOT mean | output tok/s/GPU |
|---|---|---|---|---|---|
| 3P1D PP4 + DEP8, c264 | 12+8=20 | **63,647** | 4.34 s | 11.1 ms | 458 |
| 3P1D PP4 + DEP8, c216 | 12+8=20 | **55,488** | 3.15 s | 10.8 ms | 393 |
| 4P1D PP4 + DEP16, c256 | 16+16=32 | 44,453 | 2.42 s | 8.8 ms | 314 |
| 1P1D PP4 + DEP8, c56 | 4+8=12 | 28,407 | 1.70 s | 8.8 ms | 203 |
| 1P3D PP4 + TP4, c40 | 4+12=16 | 16,924 | 1.34 s | 6.2 ms | 109 |

- **How tok/s/GPU is computed:** it is total_tput_tps (input including cache hits, plus output) divided by all GPUs. For c264, 1,272,944 / 20 = 63,647. This is the same definition as ours.
- **Which points match "~50K":** c216 (55.5K) and c264 (63.6K). Both use 5 GB300 nodes.
- **Run conditions (c264):**
  - duration_seconds is 3630, versus our 1800.
  - Mean input is 138.7K tokens and mean output is 1005 tokens.
  - Server cache hit rate is 0.64 on GPU and 0.31 on CPU; the theoretical hit rate is 0.97.
- **Agg points are much lower:**
  - The best Dynamo+SGLang agg point is B200 TP8+DPA8 c32 at 20,014 (2026-10-22, run 2747).
  - The best plain-SGLang point is B300 c28 at 23,239.
  - The GB300 agg key only has c1 points.

## 2. Image and machinery (first-hand)

- **SGLang image:** `lmsysorg/sglang:nightly-dev-20261016-a93f16f0@sha256:0b3e39…` (nvidia-master.yaml:7550). The recipe reports sglang `0.0.0.dev1+gc15edf314` (disagg-variants.yaml:10).
- **Dynamo:** wheel `1.6.0.dev20261016` (disagg-variants.yaml:9,17). Router field in nvidia-master.yaml:7556.
- **Source PR:** #4300 (commit bc6a0446). It moved all disagg points to one PP4 prefill worker per node, which relies on sgl-project/sglang#39378 "Fix disagg PP MTP for GLM-5.2" (merged 2026-09-19).
- **Host DRAM:** `dram-utilization: 0.85` (nvidia-master.yaml:7563).
- **Synthetic MTP acceptance:** `infx/srt_slurm/synthetic_acceptance.py:218-228` sets three env vars on every role with spec decoding, prefill included:
  - `SGLANG_SIMULATE_ACC_LEN=<golden AL>`
  - `SGLANG_SIMULATE_ACC_METHOD=match-expected`
  - `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`
  - #4300 gives AL = 3.61 at 5 steps, the same value we use.
- **srt-slurm commit:** the submodule pin at current HEAD is `ae3d6877` (NVIDIA/srt-slurm). The SHA actually used on 10-20 is unknown; it is recorded in that run's `srt-slurm-sha.txt` artifact (`infx/launch/drivers/srt/checkout.py:93-96`).

## 3. Server flags per role (first-hand, `gb300-fp4/agentx/disagg-variants.yaml`)

**Common env (L43-53, L89-100):**
- `SGLANG_DISAGGREGATION_{HEARTBEAT_MAX_FAILURE,BOOTSTRAP_TIMEOUT,WAITING_TIMEOUT}=100000`
- `SGLANG_USE_MESSAGE_QUEUE_BROADCASTER=0`
- `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True`
- `MC_FORCE_MNNVL=1`, `NCCL_MNNVL_ENABLE=1`, `NCCL_CUMEM_ENABLE=1`, `NVSHMEM_REMOTE_TRANSPORT=none`
- Decode only: `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8` (L95)

**Prefill (base L54-86, c264 override L249-261), 3 workers:**
- **Parallelism and quantization:** `tp 1, pp 4`, `SGLANG_PP_LAYER_PARTITION=20,20,20,18`, `quantization modelopt_fp4`.
- **MoE / GEMM backends:**
  - `moe-runner-backend flashinfer_cutedsl`
  - `fp4-gemm-backend flashinfer_cutedsl`
- **Batching and memory:**
  - c264 sets `chunked-prefill-size 8192` and `max-prefill-tokens 8192`; the base is 16384.
  - `max-running-requests 16`.
  - `disable-cuda-graph true`.
  - c264 sets `mem-fraction-static 0.88`.
- **HiCache:** `enable-hierarchical-cache`, `hicache-size 185` (c264), `hicache-write-policy write_back`, `hicache-io-backend direct`.
- **MTP on prefill:**
  - EAGLE 5/1/6 (`speculative-num-steps 5`, `speculative-eagle-topk 1`, `speculative-num-draft-tokens 6`).
  - `speculative-moe-runner-backend flashinfer_trtllm`.
- **KV transfer:** `disaggregation-transfer-backend nixl`.
- **Router hints:** `router-mode kv`, `router-replica-sync`, `router-session-affinity-ttl-secs 3600`.
- **Not set:** overlap, schedule policy, kv dtype, page size and DSA backend, so SGLang defaults apply.

**Decode (base L101-127, c264 override L262-278), 1 worker of 8 GPUs (2 nodes):**
- **Parallelism:** `tp 8, dp 8, ep 8`, `enable-dp-attention`.
- **MoE backends:**
  - `moe-a2a-backend flashinfer`
  - `moe-runner-backend flashinfer_trtllm`
  - `speculative-moe-a2a-backend none`
  - `speculative-moe-runner-backend flashinfer_trtllm`
- **Batching and memory:**
  - `max-running-requests 256` (32 per DP rank).
  - `cuda-graph-max-bs-decode 32`.
  - c264 sets `mem-fraction-static 0.94`; the base is 0.9.
  - `chunked-prefill-size 64`.
  - `disaggregation-decode-extra-slots 0`.
- **Caching:** `disable-radix-cache`.
- **MTP and output:** EAGLE 5/1/6, `stream-interval 30`, `skip-tokenizer-init`.
- **Router hints:** `router-decode-active-request-weight 8192`.

**Other DEP8 points:**
- **c216 (L223-247):** identical to c264 except that the prefill side keeps the base settings (16k chunks, mem 0.83, `hicache-size` not set) and decode keeps mem 0.9.

**TP4-decode points (c30, c40):**
- Prefill sets `hicache-size 100`.
- Decode sets `disaggregation-decode-enable-radix-cache` and 24 running requests.

## 4. Dynamo side (first-hand)

- **Frontend:** `type dynamo`, `router-mode kv`, `router-replica-sync true`, and all three active-* thresholds set to `None` (L23-35).
- **Number of frontends:** `enable_multiple_frontends: true` (L27) puts nginx in front of the Dynamo routers.
  - srt-slurm's default `num_additional_frontends` is 9, so there are 10 routers (srt-slurm `src/srtctl/core/schema.py:2591,2630` at ae3d6877).
  - The recipe does not override this count.
- **Session affinity:**
  - `nginx_session_affinity: true` consistently hashes on `X-Dynamo-Session-ID` (L25-26; schema.py:2596).
  - The client sets that header from its correlation id: `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true` (L133).
- **Client sampling:** `enable_thinking: true`, `temperature 1`, `top_p 0.95` (L136).

## 5. Differences from our S1 (first-hand diffs; the effect of each is not measured)

| Item | IX c216/c264 | Ours S1 |
|---|---|---|
| HW / quantization | GB300 NVL72, NVFP4, MNNVL | MI355X, MXFP4 |
| GPUs, P:D | 20, 12:8 (3 prefill workers) | 8, 4:4 (1 prefill worker) |
| Decode | DEP8 (EP8, flashinfer a2a, trtllm MoE), 32/rank graph | TP4 + DPA4, EP1, graph bs 256 |
| Decode radix | disabled | default (not set in S1.argv) |
| Prefill overlap / policy | defaults | `--disable-overlap-schedule`, `shortest-prefill-first` |
| Prefill HiCache | 185 GB/rank, io `direct`, max-running 16 | 250 GB, io `kernel`, `page_first`, max-running 256 |
| KV transfer | NIXL over MNNVL | Mooncake (RDMA) |
| Router | Dynamo KV router ×10 behind nginx session hash; prefill affinity TTL 3600; decode request-count balancing | Infera router |
| Benchmark | 3600 s, mean ISL 138.7K | 1800 s |

The PP4 TP1 prefill, the 20,20,20,18 layer split, 8k chunks, write_back, EAGLE 5/1/6 and AL 3.61 match ours.

**Derived arithmetic on published numbers (inference, not measured):**
- c264 input throughput per prefill GPU is 1,263,781 / 12 ≈ 105K tok/s.
- Ours is about 34,509 × 8 / 4 ≈ 69K total tok/s per prefill GPU.
- Our output-per-decode-GPU figure is not in my inputs, so I have not compared decode.

## 6. Unknowns and what would settle them

- **Defaults not set in the recipe** (kv-cache dtype, page size, DSA prefill/decode backend, attention backend, overlap): read the `server_args` line in the worker logs of run 39187209967, or run `python -m sglang.launch_server --help` in image a93f16f0.
- **Where the 63.6K vs 34.5K gap comes from:** these items are listed only as differences, with no attribution. Settling it needs one-variable runs on our side, for example a 3:1 P:D GPU ratio or decode with EP.

# READER diff

# SGLang+Dynamo GLM-5.2 "~50K" points diffed against our S1/S2 (MI355X)

## 1. The ~50K points (first-hand, from the InferenceX API)

Source is the InferenceX API `/api/v1/benchmarks?model=GLM-5.2&date=2026-10-20&exact=true`, saved at `/var/tmp/yihou/ix_api/b_2026-10-20.json`. Both points come from workflow run 2690 on GB300, image `lmsysorg/sglang:nightly-dev-20261016-a93f16f0`, Dynamo 1.6.0.dev20261016. The recipe is from #4300 (bc6a0446).

| Point | Topology | tok/s/GPU | TTFT mean | TPOT mean | GPU / CPU cache hit |
|---|---|---|---|---|---|
| c264 | 3x PP4 prefill (12 GPU) + 1x DEP8 decode (8 GPU) | **63,647** | 4.34 s | 11.1 ms | 0.64 / 0.31 |
| c216 | same | **55,488** | 3.15 s | 10.8 ms | 0.69 / 0.26 |
| c256 | 4x PP4 + 1x DEP16 (32 GPU) | 44,453 | 2.42 s | 8.8 ms | – |
| B300 agg TP4+DP4 c48 (2026-10-14) | single 4-GPU engine | 42,047 | 6.41 s | 19.0 ms | – |

- The per-GPU number is total throughput divided by 20 GPUs: 1,272,944 / 20 = 63,647. Input including cache hits is 99.3% of that (input_tput 1,263,781).
- The benchmark ran for 3,630 s. Ours runs 1,800 s.
- Acceptance is simulated at AL 3.61 by the launcher, not by the recipe. Sources: `b300-fp4/agentx/agg-tp4-variants.yaml:5-9` and the #4300 body. This matches our `SGLANG_SIMULATE_ACC_LEN=3.61`, so throughput is comparable on that axis. Neither side's number is a correctness result.
- Answer to "which point is 50K": it is c216 (55.5K), and c264 is 63.6K. The single-engine aggregated number closest to it is B300 c48 at 42K.

Recipe file: `.../glm5.2/sglang/gb300-fp4/agentx/disagg-variants.yaml` (base plus `override_3p1d_pp4_dep8_c264` / `_c216`). Master config is `configs/nvidia-master.yaml:7549`, with `dram-utilization 0.85`.

## 2. Topology

- **Theirs:** GB300 with 4 GPUs per node. Each prefill node runs one PP4 TP1 worker. Decode is one DEP8 worker across 2 nodes, using MNNVL (`MC_FORCE_MNNVL=1`, `NCCL_MNNVL_ENABLE=1`). P:D is 12:8. Total is 5 nodes (+infra).
- **Ours (S1):** one 8-GPU MI355X node. One PP4 TP1 prefill worker on GPUs 0-3, one TP4 DP4 decode worker on GPUs 4-7. P:D is 4:4.
- Two differences in kind:
  - They have 3x the prefill workers per decode worker.
  - Their decode EP group spans 8 GPUs over NVLink.

## 3. Flag-by-flag: prefill (c264 vs our S1 prefill argv)

| Recipe setting | Ours | Exists in our branch |
|---|---|---|
| PP4 TP1, `SGLANG_PP_LAYER_PARTITION=20,20,20,18` | present, identical | – |
| chunked-prefill / max-prefill 8192 | present (8192; an earlier 32768 is overridden by the later flag) | – |
| mem-fraction-static 0.88 | present, 0.88 | – |
| HiCache write_back, size 185 GB/rank | present, 250 GB/rank (ratio 1.5) | – |
| hicache-io-backend `direct`, default mem-layout | ours is `kernel` + `page_first` | yes, `fields/memory.py:128` (choices direct/kernel, default kernel) and `:135` |
| `max-running-requests 16` | ours 256 | yes |
| `disable-cuda-graph: true` | ours not set | yes, `fields/exec_.py:636` (`no_cli=True`; how the recipe's CLI form maps to it is unknown) |
| `--disable-overlap-schedule`, `schedule-policy shortest-prefill-first` | present in ours, absent in recipe | – |
| EAGLE 5/1/6 on prefill | present | – |
| `speculative-moe-runner-backend flashinfer_trtllm` | absent | field exists at `fields/spec.py:172`; the backend is CUDA-only, so not applicable on ROCm |
| `moe/fp4-gemm-backend flashinfer_cutedsl`, `quantization modelopt_fp4` | absent; ours is MXFP4 via aiter/triton | `fp4_gemm_runner_backend` at `fields/exec_.py:250` is NVFP4-only, not applicable on ROCm |
| Transfer: NIXL + `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True` | ours is mooncake over ionic RDMA; env absent | env exists at `environ.py:829` |
| `context-length 1048576` | absent (model default) | yes |
| Router hints: `router-mode kv`, session-affinity TTL 3600 | ours: Infera `--router-policy kv-aware --session-affinity prefill` (`llkit/kit/engine/up.sh:50-51`) | functional analogue; TTL equivalence unknown |
| `enable-kv-events` | ours on (Infera) | – |
| Page size | neither sets it (default) | `fields/schedule.py:139` |

## 4. Flag-by-flag: decode (c264 DEP8 vs our S1 and S2)

| Recipe setting | S1 | S2 | Branch |
|---|---|---|---|
| TP8 **DP8 EP8**, dp-attention | TP4 DP4 **EP1** | TP4 EP4 DCP4, no DP | – |
| `moe-a2a-backend flashinfer` (MNNVL a2a) | absent (EP1) | absent (EP4, a2a `none`) | field at `fields/exec_.py:802`; ROCm option is `mori` (`layers/moe/utils.py:41`), `deepep` also listed. Whether mori runs with MXFP4 GLM-5.2 on our image is unknown; a launch test would settle it |
| `moe-runner-backend flashinfer_trtllm` (NVFP4) | ours: aiter MXFP4 | – | not applicable on ROCm |
| `speculative-moe-a2a-backend none` | default | default | `fields/spec.py:180` |
| `disable-radix-cache: true` on DEP decode | radix off by default in PD decode (`pd_disaggregation_hook.py:120-126`) | same | same behaviour |
| `disaggregation-decode-enable-radix-cache` on TP4 decode (c30/c40 points only, ≤17K) | absent | absent | `fields/disagg.py:85`; with DP attention it is "EXPERIMENTAL" (`pd_disaggregation_hook.py:110-113`) |
| HiCache on decode | absent in recipe | absent | `disaggregation_decode_enable_offload_kvcache` at `fields/disagg.py:89`, unused on both sides |
| max-running 256 = **32/rank**, graph bs 32 | 256 = 64/rank, graph bs 256 | 96, graph bs 96 | `cuda_graph_max_bs_decode` at `fields/exec_.py:595`. Whether 256 is per-rank or global in ours: unknown (needs a capture-log read) |
| mem-fraction 0.94 | 0.92 | 0.85 | – |
| `chunked-prefill-size 64` on decode | absent (our base 32768 is inherited) | absent | – |
| `disaggregation-decode-extra-slots 0` | resolves to 0 anyway, since per-worker 64 > 32 (`pd_disaggregation_hook.py:133-142`) | 96 is ≤32? no, DP1, so 96 > 32 and it is 0 | `fields/disagg.py:110` |
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8` | absent (default 4096, `environ.py:620`, read at `managers/schedule_policy.py:87`) | absent | exists. Whether it gates PD-decode admission on our path: not verified by code read |
| `stream-interval 30`, `skip-tokenizer-init` | absent (default 1, `fields/serving.py:256`) | absent | `fields/serving.py:256/58` |
| `router-decode-active-request-weight 8192` | ours `--kv-decode-overlap-weight 2 --decode-input-demand on` | – | Infera-specific; equivalence unknown |
| `index_share_for_mtp_iteration` | absent in recipe | ours on (S1, S2) | – |
| `enable-dp-lm-head`, `enable-prefill-delayer` (B300 agg c48 only) | absent | absent | `fields/parallel.py:199`, `fields/schedule.py:228` |

## 5. Features asked about that are not used in the 50K points

- **Prefill context parallel:** used only in the H200 FP8 recipes (`h200-fp8/agentx/disagg-1p1d-pcp8-...yaml:62-64`: `attn-cp-size 8`, `enable-prefill-cp`, `cp-strategy interleave`, `page-size 64`). Our branch has `enable_prefill_cp` (`fields/parallel.py:153`) and `attn_cp_size` (`:104`).
- **DCP:** the API reports `decode_dcp_size 1`, `prefill_pcp_size 1` for c216/c264. No DCP is used. Our S2 (DCP4) scored lower than S1 (25.9K vs 34.5K).
- **DeepEP:** not used; they use flashinfer a2a.

## 6. Inference (low confidence, not measured)

These are the differences that can be stated; how much each contributes is unknown.

1. **Prefill GPU ratio:** 12:8 vs our 4:4. Our S1 TTFT is 5.5 s at C80; theirs is 4.3 s at C264.
2. **Decode EP:** DEP8 with EP8 a2a vs our EP1 MoE on 4 GPUs.
3. **Per-rank decode batch:** 32 vs our 64.
4. **Decode-only knobs we lack:** clip-estimation 8, stream-interval 30, skip-tokenizer-init, decode chunk 64.
5. **Benchmark length and cache-hit rate:** 3,630 s vs our 1,800 s, and the cache-hit rate shifts the input share. Our server cache-hit rates are not in this note; reading them from our run's metrics would settle the comparison.

To attribute the gap, change one variable at a time against S1: decode EP/a2a (mori), then clip-estimation/stream-interval, then the P:D ratio.

API dumps are in `/var/tmp/yihou/ix_api/`.

# READER dynamo

**Dynamo in the InferenceX GLM-5.2 SGLang runs compared with our Infera router**

Tags: **[D]** = documented (recipe text, launcher code, PR body or API data). **[I]** = my inference. **[U]** = unknown, with what would settle it.

## 0. Where the ~50K comes from [D]
I pulled this from the InferenceX API for 2026-10-20 (saved as `/var/tmp/yihou/glm52_api_2026-10-20.json`). All three points are GB300 runs with framework `dynamo-sglang`, `router_name: dynamo-router 1.6.0.dev20261016` and `kv_p2p_transfer: nixl`.

| Point | tok/s/GPU | GPUs | TTFT mean (s) | TPOT (ms) | Server cache hit, GPU / CPU | Duration (s) |
|---|---|---|---|---|---|---|
| 3P(PP4) + 1D(DEP8), c264 | **63,647** | 12 P + 8 D | 4.34 | 11.1 | 0.64 / 0.31 | 3630 |
| 3P(PP4) + 1D(DEP8), c216 | **55,488** | 12 P + 8 D | 3.15 | 10.8 | 0.69 / 0.26 | 3630 |
| 4P(PP4) + 1D(DEP16), c256 | 44,453 | 16 P + 16 D | 2.42 | 8.8 | — | — |

- Throughput is total tokens over all GPUs: c264 is 1,272,944 tok/s ÷ 20 = 63,647.
- The recipe is `gb300-fp4/agentx/disagg-variants.yaml`, overrides `override_3p1d_pp4_dep8_c216` (L223) and `_c264` (L249), from commit bc6a0446 / PR #4300.

**These runs use 2.5x our GPU count, a 1.5:1 prefill:decode GPU ratio, and 3600 s instead of our 1800 s.** None of those differences comes from Dynamo.

## 1. What the Dynamo stack is [D]
- **Frontend.** `python3 -m dynamo.frontend` runs as the HTTP entry point (srt-slurm `frontends/dynamo.py:225`). srt-slurm is the `ae3d6877` submodule, cloned to `/var/tmp/yihou/src/srt-slurm`.
- **Workers.** Workers run as `python3 -m dynamo.sglang`, not `sglang.launch_server` (`backends/sglang.py:345`). Request plane is tcp (`schema.py:2422`). Discovery is etcd. The recipe also starts NATS with a 64 MB payload limit and disables OTel (recipe L137-151).
- **Multiple frontends.** `enable_multiple_frontends: true` puts nginx in front of several frontend replicas; the default is 1 + 9 (`schema.py:2630`). `nginx_session_affinity` hashes `X-Dynamo-Session-ID` consistently across those replicas (`templates/nginx.conf.j2:18-38`), and `router-replica-sync: true` shares router state between them (recipe L32).
  - [U] How many frontend replicas actually ran depends on node placement. The job log `*_frontend_*.out` would settle it.
  - Our setup has a single Infera router process (`kit/engine/up.sh:46-52`).

## 2. KV-aware / prefix-aware worker selection

**Theirs [D]:**
- Frontend runs `router-mode: kv` (L31).
- Prefill workers advertise `router-session-affinity-ttl-secs: 3600` to the prefill router (L80-83). PR #4300 moved this setting from the frontend to the worker.
- Decode workers advertise `router-decode-active-request-weight: 8192` to the decode router (L120-124). The recipe comment says this makes decode "balance request counts".

**KV events on GB300 [D]:**
- The GB300 recipe sets **no `kv_events`** on either role. srtctl passes `--kv-events-config` only when a role has `kv_events` set (`backends/sglang.py:422-427`, `backends/base.py:140-148`, `docs/topology.md:82`).
- By contrast, the GB200 recipe does set `kv_events: true` and `router-kv-events: true` (`gb200-fp4/.../disagg-mtp-variants.yaml:23,108,173`).

**[U] What the GB300 KV router does with no events.** It may fall back to approximate prediction from its own routing decisions, or to pure load balancing. Settle with the Dynamo 1.6.0.dev20261016 router source or the frontend log.

**Ours [D]:** `--router-policy kv-aware --kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill`. Session affinity is keyed on `X-Session-ID` (infera e54e9c62 / PR #188), and the client sends that header (`AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID`, `agentx/agentx.sh:57`). Prefill publishes KV events over ZMQ; decode runs `--kv-events off` (`render/S1.argv.txt`).

**[I] On the 50K points, worker-level selection is mostly moot on both sides.**
- They have 3 prefill workers and 1 decode worker. With a 3600 s TTL, a session stays pinned to one prefill worker.
- Our S1 has 1 prefill and 1 decode worker, so there is nothing to choose between.
- So I would not attribute the gap to the routing algorithm until it is measured.

## 3. PD bootstrap and KV transfer [D]
- **Theirs.** `disaggregation-transfer-backend: nixl` on both roles (L59, L106), over MNNVL (`MC_FORCE_MNNVL=1`, `NCCL_MNNVL_ENABLE=1`). The PR body says "KV transfer uses NIXL; no recipe sets UCX_*".
  - Dynamo does not do the bootstrap. SGLang's own bootstrap server still runs on each prefill worker; srtctl only assigns its port (`backends/sglang.py:396-402`).
  - Dynamo does choose which prefill/decode pair serves each request.
- **Ours.** Mooncake over RDMA (`--disaggregation-transfer-backend mooncake` plus the ionic IB map). Same-node transfers use a Mooncake filter that matches no HCA (`kit/env.sh:101`).
- [U] How much transfer time contributes to TTFT on either side. The per-request transfer spans would settle it; theirs are disabled together with OTel.

## 4. Decode prefix reuse [D]
- **50K+ DEP points: decode prefix reuse is off.** They set `disable-radix-cache: true` (L245, L275).
- **Only the TP4 1P3D/1P5D points enable it** (`disaggregation-decode-enable-radix-cache: true`, L174, L193). Those points score 8.9K and 16.9K.
- **Our S1 decode is the same:** no decode radix cache and no decode KV events.
- **So decode prefix reuse is not a differentiator for the 50K number.**

## 5. Request concurrency control

**Theirs [D]:**
- Busy thresholds on the router are all `None` (L33-35, and per-role at L165-167). [I] By the flag names this turns off busy-worker gating, so admission is left to the engines.
- Engine-side admission limits:
  - prefill `max-running-requests: 16`
  - decode `max-running-requests: 256`, i.e. 32 per DP rank × 8, with `cuda-graph-max-bs-decode: 32` (L220-221)
  - `disaggregation-decode-extra-slots: 0`
  - `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8`

**Ours [D]:**
- `--pd-prefill-guard-release completion`: releases the prefill reservation when prefill completes rather than when decode ends (infera PR #185).
- `--decode-input-demand on`: chooses a decode DP rank by input-token demand (PR #188, `rust/router/src/decode_demand.rs`).
- Decode `max-running-requests 256` with graph batch size 256.
- The effects below are second-hand, from `infera.glm52.pd/llying.analysis.md:78,80,82`, measured on other topologies; I did not reproduce them:
  - guard release: +5.4% on P8D8
  - prefill session affinity: +12.1% on 3P1D
  - decode input demand: +3.2% / +9.3%

**[U] How Dynamo picks a DP rank inside one DEP8 decode worker.** It could be the Dynamo router or SGLang's internal DP balancer. The `dynamo.sglang` registration log or the per-DP running counts would settle it.

## 6. Other things in the Dynamo path that could affect throughput

**Front-end and dataflow [D]:**
- Decode runs `skip-tokenizer-init: true` (L114), so the decode worker has no tokenizer; [I] the Rust frontend presumably handles tokenize/detokenize.
- Ours: the router tokenizes for routing and decode runs `--tokenizer-worker-num 4`. [U] Whether Infera forwards token IDs to the engine. A code read of `infera/router/disagg.py` would settle it.
- Decode sets `stream-interval: 30` (L115); ours uses the default.

**Prefill and benchmark configuration [D]:**
- `SGLANG_PP_LAYER_PARTITION=20,20,20,18` on prefill (L231).
- Prefill HiCache 185 GB/rank at c264.
- `dram-utilization` raised to 0.85 (PR #4300 body).
- Client sampling: `temperature 1, top_p 0.95, enable_thinking` (L136).
- Simulated acceptance is the same as ours: AL 3.61 at 5 steps (PR #4300).

**Not Dynamo [D]:** NVFP4 with flashinfer cutedsl/trtllm MoE (L64-66, L214-217) and the GB300 hardware. These fall outside the router comparison.

## 7. Summary
- **Documented Dynamo contributions:**
  - multiple frontends behind nginx with session-header stickiness and state sync between router replicas
  - KV-mode routing with a 3600 s prefill session TTL
  - decode balanced by request count (weight 8192)
  - router busy thresholds disabled
  - a Rust frontend with tokenizer-less decode
  - NIXL transfer
- **Not attributable to Dynamo on the evidence I have.** The 50K+ points have 1 decode worker and session-pinned prefill. That leaves little routing freedom, and our S1 has none (1P1D).
- The documented differences that are large are topology (20 GPUs at 1.5:1 P:D vs our 8 at 1:1), the hardware/kernel stack, and run duration.
- **Two things are open:** whether GB300 KV routing runs with or without KV events, and how the DP rank is chosen inside DEP8.