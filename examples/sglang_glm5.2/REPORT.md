# GLM-5.2-FP8 1P1D on MI325X — experiment log

Two 8×MI325X (gfx942) nodes, one prefill leg and one decode leg, KV over Mooncake
RDMA on a Broadcom bnxt_re RoCEv2 fabric, fronted by the Infera router. Everything
runs inside live containers; the image cannot be rebuilt, so image-time fixes are
reproduced by scripts here.

Chinese version: [`REPORT.zh.md`](REPORT.zh.md).

| | |
|---|---|
| prefill | `10.32.17.210:30001` (`tus1-p15-g46`) |
| decode | `10.32.17.209:31001` (`tus1-p15-g45`) |
| router / etcd | `10.32.17.210:8000` / `:2379` |
| model | `/wekafs/models/GLM-5.2-FP8`, TP8 + DP-attention, fp8_e4m3 KV, tilelang DSA |
| SGLang | v0.5.16 |
| Mooncake | upstream #2682 (`01d1eb2a`) + HIP-transport gate |

---

## 1. Blockers and known landmines

### 1.1 Mooncake forces HIP IPC on a cross-node peer

Every request through the pair died at the KV hand-off:

```
E0728 12:41:38 hip_transport.cpp:70] HipTransport: hipIpcOpenMemHandle failed
                                     (Error code: 17 - invalid device pointer)
[DP0 TP0] Session 10.32.17.209:16680 failed.
[DP0 TP0] Prefill transfer failed ... Failed to send kv chunk
```

`hipIpcOpenMemHandle` is intra-node GPU P2P; it can never open a handle from
another host. Upstream Mooncake #2682 installs the HIP transport unconditionally
and `selectTransport` prefers it over RDMA.

Measured: the `MC_USE_HIP_IPC=0` runtime knob does **not** help — with it set, both
legs still logged `HIP transport installed for intra-node GPU P2P` eight times
each. It tunes the transport, it does not stop it being installed and preferred.
Only the source gate works.

Fix: `patch_mooncake_hip.sh` applies `deploy/docker/patches/mooncake_cpp/
transfer_engine_impl.diff` (gates the install behind `MC_ENABLE_HIP_TRANSPORT`),
rebuilds `engine.so` incrementally (~30 s, one translation unit), installs it over
the pip module, and caches the result on shared storage keyed by the stock .so's
hash so the second node installs instead of rebuilding. Both nodes reported the
same stock hash `10f75b32e43501e9`, which also confirms they run the same image.

After: `HIP transport installed` 0×, `installTransport, type=rdma` 8× on both legs.
Both leg scripts now refuse to start if the gate is missing, rather than loading
for 20 minutes and failing on the first request.

The auto-chunk-MR patch from the same directory was deliberately **not** applied:
it fixes a ~2 GiB `max_mr_size` truncation on ionic NICs, and this fabric's
bnxt_re reports `max_mr_size = 512 GiB`, well above the ~118 GiB KV pool.

### 1.2 MTP deadlocks the decode leg during PD warmup

The decode leg finished loading, served `/model_info`, then hung at
`Start of pd disaggregation warmup` for 25 minutes. `/health` returned 503
throughout because `server_status` never left `Starting`.

py-spy on all eight schedulers put every rank at the same frame, and six samples
over twelve seconds produced an identical stack — a real deadlock, not a hot loop:

```
torch.distributed.all_reduce
  _all_reduce_in_place (parallel_state.py:884)
  _dp_gather_via_all_reduce (dp_attention.py:463)
  dp_gather_replicate (dp_attention.py:629)
  forward (deepseek_nextn.py:273)          <- MTP draft model
  _execute_idle (eager_runner.py:406)
  _draft_extend_for_decode (eagle_worker_v2.py:945)
  event_loop_overlap_disagg_decode (decode.py:2057)
```

Why eager: on ROCm the draft-extend CUDA graph is only captured when the draft
attention backend is `AiterMultiStepDraftBackend`
(`eagle_worker_v2.py:421-430`); `supports_cuda_draft_extend_graph` additionally
requires `_is_cuda or _is_musa` (`eagle_worker_v2.py:463-465`), and although
`DeepseekSparseAttnBackend` *is* in `graph_supported_backend_types`, it is only
appended inside an `if _is_cuda or _is_musa` branch. On HIP neither holds, so
draft extend always runs eager. The log confirms only `target verify` and
`draft decode` graphs were captured.

#### The missing graph is not the cause

An earlier revision of this report blamed the eager draft-extend path. That is
wrong, and the single-node run disproves it. `inference_glm5p2_sglang/tmp/mtp/
server_safe_dp8.log` is a DP8 + MTP run on one node, and it:

- captured exactly the same two graphs, `target verify` (`num_tokens_per_req=4`,
  `bs=[1..6]`) and `draft decode` (`num_tokens_per_req=1`, `bs=[1..6]`);
- logged **zero** occurrences of "draft extend", i.e. no graph there either;
- landed on the same `max_running_requests=6` per rank;
- and served **197 requests, all HTTP 200**, at `accept len 3.80–3.85` out of 4
  draft tokens.

So eager draft extend under DP attention with MTP is a perfectly working
configuration. Something PD-specific is required to make it deadlock.

#### What is actually PD-only

The decode leg of a PD pair schedules a `PREBUILT` batch — the "fake completed
prefill" standing in for work done on the other node
(`disaggregation/decode.py:get_new_prebuilt_batch`). `PREBUILT` exists nowhere
else, and DP MLP sync has a special case for it:

```338:341:/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/dp_attn.py
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()
```

That nested idle batch is unwrapped only in the PD decode loop
(`disaggregation/decode.py:2080-2084`, the sole consumer of `inner_idle_batch`),
and the hang sits exactly on that path: the py-spy frame `decode.py:2057` is
`self.run_batch(batch)` inside `event_loop_overlap_disagg_decode`, and
`eager_runner.py:406` is the `model.forward(...)` inside `_execute_idle`. Every
rank was running an *idle* draft extend nested inside a prebuilt batch, and the
DP gather's `all_reduce` never returned.

Single-node DP8 never constructs a `PREBUILT` batch, so no amount of single-node
DP-attention + MTP testing could have covered this interaction. The next step is
to find out why the gather does not complete when all eight ranks appear to be
inside it — the likely candidates are a per-rank disagreement on
`global_num_tokens` (mismatched all_reduce sizes hang) or two ranks being in
different collectives despite identical Python frames. `NCCL_DEBUG=WARN` plus
logging `mlp_sync_info.global_num_tokens` per rank should separate the two.

#### MTP has to be on both legs

Worth recording because it is not obvious: `MTP` is a must-match parameter across
the pair, like `page_size` and `kv-cache-dtype`. Both leg scripts pass identical
`MTP_ARGS`, and that is required, even though no speculation happens during
prefill — the prefill leg emits exactly one token and never runs a draft/verify
loop. What it does instead is set up everything the decode leg's *first* draft
step needs:

1. Runs `_draft_extend_for_prefill` (`eagle_worker_v2.py:1151`) so the draft
   model's own `draft_token_to_kv_pool` is populated over the prompt.
2. Appends that draft pool to the registered RDMA buffers
   (`disaggregation/prefill.py:186-194`, "We should also transfer draft model kv
   cache"), so Mooncake ships target KV *and* draft KV.
3. Saves the EAGLE seed per request — `output_topk_p`, `output_topk_index`, and
   `hidden_states_tensor` (`disaggregation/prefill.py:661-667`) — into aux buffers
   that exist for exactly this purpose (`disaggregation/utils.py:298-306`,
   commented `# For PD + spec decode`).

Point 3 is the hard requirement: EAGLE drafts from *(last token embedding, that
token's target-model hidden state)*, and prefill is the only place that hidden
state is produced. Enabling MTP on decode alone would leave the two legs
registering different numbers of KV buffers and would never deliver the seed.

Consequence for debugging: MTP on prefill cannot be switched off to narrow down
the decode-side deadlock — doing so breaks the pair instead of isolating the bug.
The cost of carrying it is real, though: loading the draft model re-reads the
704 GiB checkpoint a second time (~2× startup), plus draft-pool memory and one
extra draft-extend forward per prefill, with all of the benefit landing on decode.

Status: **worked around** by running with MTP off (`MTP=0`, the current default in
both leg scripts). Root cause still open — see section 6.

Related observation for when MTP comes back: with MTP on, `max_running_requests`
collapsed to 48 global / 6 per DP rank and the decode CUDA graph captured only
`bs=[1..6]`. Without MTP it is 2048 with `bs=[1,2,4,...,512]`. Note the single-node
run shows the same 6 per rank, so this is an MTP property rather than a PD one —
but it is still a large concurrency ceiling and needs its own explanation.

### 1.3 Not hit yet, but certain to hit: DSA indexer row mismatch under DP attention + MTP at concurrency

Not from our own runs. A colleague on 8×MI355X (gfx950), ROCm 7.2.0, sglang
0.5.15.post1, GLM-5.2-**MXFP4**, single node `--dp-size 8 --enable-dp-attention
--ep-size 8` + EAGLE `steps=3 topk=1 draft-tokens=4`, crashes **on the first
batch**:

```
RuntimeError: Expected lengths.size(0) == B to be true, but got false.
dsa_indexer.py::_get_topk_paged -> metadata.topk_transform
-> torch.ops.sgl_kernel.fast_topk_transform_fused
```

A patch already exists: Infera PR #34,
`deploy/docker/patches/sglang_dsa/dsa_indexer_hip_dp_padded_rows.diff`.

**Recorded here because this has nothing to do with MXFP4 — our configuration sits
in the same minefield.** The assert compares `score.size(0)` against
`lengths.size(0)` (`sgl-kernel/csrc/elementwise/topk.hip:397`): a pure row-count
question, independent of how the weights are stored. Checking both runtimes
field by field, concurrency is the only thing that differs:

| | ours (single-node DP8 + MTP) | colleague (MI355X) |
|---|---|---|
| arch / ROCm | gfx942 / 7.2.0 | gfx950 / 7.2.0 |
| SGLang | v0.5.16 | v0.5.15.post1 |
| weight quant | FP8 | MXFP4 |
| paged-MQA backend | aiter | aiter |
| `_use_aiter_preshuffle` | True (triton 3.6.0) | True |
| `page_size` | 64 | 64 |
| `dsa_topk_backend` | `sgl-kernel` | `sgl-kernel` |
| peak concurrency | **≤ 1** | **64** |

`--dsa-prefill-backend tilelang --dsa-decode-backend tilelang` does not save us:
that selects the sparse MLA attention kernels, while the indexer's paged-MQA
backend is **forced** to aiter on ROCm (`dsa/paged_mqa_logits_backend.py:24-30`
raises for anything but `auto`/`aiter`). Both offending lines are present verbatim
in v0.5.16 — `dsa_indexer.py:977-987` hands the **unsliced** `q_fp8` to
`aiter_paged_mqa_logits`, and the padding restore is gated `not _is_hip`:

```1037:1039:/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
        # Restore possible padding exist in the hidden states.
        if not _is_hip and q_offset < q_fp8.shape[0]:
            pad_len = q_fp8.shape[0] - q_offset
```

The CUDA path (`deepgemm_paged_mqa_logits_split`) slices q/weights to `q_offset`
first and restores the padding afterwards, which is why it was always correct;
aiter does neither.

**Why our run missed it: load shape, not a safer configuration.** Padding only
appears when the DP ranks carry token counts that are *close but unequal*. The
decision is `dp_attention.py:99-106` — decode-class batches pick MAX_LEN (pad
every rank up to the group max) only when `sum*2 >= max*dp_size`, otherwise
SUM_LEN (each rank keeps its own length, no padding). In
`tmp/mtp/server_safe_dp8.log` **all 70 `Decode batch` lines read
`#running-req: 1`**: one request node-wide, so
`global_num_tokens = [4,0,0,0,0,0,0,0]`, `sum*2 = 8 < max*dp = 32` → SUM_LEN → no
rank ever carries a padded row. The colleague ran conc=64 on dp8, 7–9 requests per
rank (28–36 tokens), `sum*2 ≈ 456 ≥ max*dp ≈ 288` → MAX_LEN → every rank below the
max gets padded. **Our 197 green requests only prove that test was too gentle.**

**Why `DRAFT_EXTEND_V2` specifically.** They measured 27 distinct padded shapes in
one conc=64 run: 26 in `DRAFT_EXTEND_V2` (typically `q_fp8=(36,32,128)` vs
`q_offset=32`, i.e. 9 req×4 rows against 8 req×4) and none in `TARGET_VERIFY`.
That matches the code: in eager draft extend, `_pad_inputs_to_size`
(`forward_batch_info.py:1364-1416`) pads `input_ids` to the DP-group max token
count and pads the `extend_seq_lens` device tensor to `bs` with zeros, but never
touches `extend_seq_lens_cpu` — and `q_offset = sum(metadata.get_dsa_extend_len_cpu())`
(`dsa_indexer.py:922`) reads exactly that CPU list. `TARGET_VERIFY` is safe because
it recomputes `batch_size` as `num_tokens // num_tokens_per_req`
(`forward_batch_info.py:1336-1342`), so B and the row count are padded together.
Note §1.2 already established that draft extend **always runs eager on HIP**, so
we have been on this path the whole time; only the load shape spared us.

**The exposure is direct.** With MTP on, the ceiling is 6 requests per rank, so any
distribution like `[6,6,6,5,6,6,6,6]` (tokens `[24,…,20,…]`, `sum*2=376 ≥
max*dp=192` → MAX_LEN) pads the 20-token rank to 24 rows while `q_offset` stays 20
— same assert. In other words, **once §1.2's deadlock is fixed, the first
concurrent dp8 + MTP run will die here**; apply the patch before load testing.

**One thing still unexplained — do not read this as "low concurrency is safe".**
One of the 27 shapes was in `IDLE` (`q_fp8=(4,32,128)` vs `q_offset=1`), and 4 rows
implies a single request node-wide — yet by the heuristic above `[4,0,…,0]` should
land on SUM_LEN with idle ranks at 0 rows (which is what our run did). That
contradiction is unresolved; static reading did not settle it. Until the patch is
applied, do not conclude from "our low-concurrency run passed" that low
concurrency cannot trip it.

A checkable environment difference worth noting: our image exports
`SGLANG_USE_ROCM700A=1` (no script in this repo sets it), which makes every
cuda-graph path use SUM_LEN instead of MAX_LEN (`dp_attention.py:108-115`, a RCCL
work-around for ROCm 7.0.0-alpha). It does not participate in
`get_dp_padding_mode`, so it is **not** why we escaped — but if part of their
padding comes from the graph path, that variable is worth comparing.

---

## 2. Working configuration

`MTP=0`, `IB_DEVICE=rdma0`, both legs otherwise identical:

```
TP8 / DP8 + --enable-dp-attention
--kv-cache-dtype fp8_e4m3, page_size 64
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang
--chunked-prefill-size 131072   (-> 16384 per DP rank)
--mem-fraction-static 0.85
--disaggregation-transfer-backend mooncake --disaggregation-ib-device rdma0
```

Both legs agree on `max_total_num_tokens=2194496`, `max_running_requests=2048`,
`page_size=64`, `context_len=1048576`.

CUDA graph state (checked because PD decode is where it matters): decode graph is
**on**, `backend=full`, `bs=[1, 2, 4, 8, ..., 496, 512]`, and every decode batch
logged `cuda graph: True`. The prefill-phase graph is off, which is SGLang's own
default (`cuda_graph_config` resolves `prefill.backend='disabled'`), not something
these scripts set.

---

## 3. Correctness

`inference_glm5p2_sglang/verify_correctness.py` against the router, full suite.
Raw output: `verify_pd_1p1d.log` / `.json`.

| check | result | notes |
|---|---|---|
| weights | 2/2 | indexer FP8 dequant self-consistency, offline |
| basic | 7/7 | short QA |
| determinism | 1/1 | greedy reproducible over 3 runs |
| idle | 3/3 | the post-idle first-request corruption seen on vLLM does not reproduce |
| **needle** | **7/9** | both 58k failures at depth 10% and 50%; see below |
| humaneval | 20/20 | short context |
| humaneval-long | 18/20 | 8k real-source padding |
| code-retrieval | 2/2 | 14k cross-file symbol lookup |
| deep-api | 3/3 | fabricated API buried at 50% depth |

HumanEval A/B delta −10% (IndentationError and KeyError — model noise, the suite
itself reports "no long-context regression"). Zero KV transfer errors during the
whole run.

### 3.1 Open: 58k needle retrieves partial digits then degenerates

```
FAIL ~64k depth 10%  ptok=58695  want=2183762  got='2183</think>2183</think>218</think>218</think> the'
FAIL ~64k depth 50%  ptok=58594  want=7544440  got='7549.8.8.</think>7549. The</think>7549.8.8759</thi'
ok   ~64k depth 90%  ptok=58611
```

Not a plain retrieval miss: the leading digits are right (2183 exactly; 754 of
7544440), then the output degenerates into repeated `</think>` even though the
request sets `enable_thinking: false`. All nine 4k and 16k cases pass; only ~58k
fails, and only at shallow/mid depth.

#### Isolated: the trigger is chunked prefill, not context length

Sweeping the prompt length across the per-rank `chunked_prefill_size` boundary
(16384 tokens) puts the break exactly on the chunk boundary:

| prompt tokens | prefill chunks | result |
|---|---|---|
| 14,642 | 1 | pass |
| 18,364 | 2 | fail |
| 29,293 | 2 | fail |
| 36,537 | 3 | fail |
| 43,932 | 3 | fail |
| 58,695 | 4 | fail |

Nothing fails while the prompt fits in one chunk; everything fails once it does
not. 64k was never the relevant number — 18k already breaks.

Two further facts pin it down. First, the failure is not a miss but a truncation:
the model returns the leading 3–4 digits of the 7-digit needle and then
degenerates into repetition — `4196795 → '4196'`, `1186494 → '1186'`,
`2183762 → '2183'`, `9706003 → '97075...'`. The needle's content is clearly
reachable, so the MLA KV itself is intact; what fails is which tokens get
selected. Second, depth matters in a specific way: at 58k, a needle at 90% depth
(inside the *final* chunk) is retrieved correctly, while needles at 10% and 50%
depth (earlier chunks) are not. Only the last chunk's contribution survives.

That points at the DSA indexer's K cache rather than the MLA KV pool: the indexer
picks top-k 2048 tokens out of the full context, and it behaves as though the
index-K written by chunks 1..n-1 is not where the top-k gather expects it. A
[read of the SGLang v0.5.16 DSA path](0cc83d68-75c7-48df-9c6f-949adeaba4b1)
confirms the indexer is *designed* to select over the full cumulative prefix
during scheduler chunked prefill (`dsa_backend.py:811-812, 897-911, 1089-1118`
use cumulative `seq_lens` for keys and chunk-local `extend_seq_lens` only for
queries), so the intent is right and the symptom is a bug in that path, not a
design limit.

#### Fix, and confirmation

Make the prompt fit in one chunk. `CHUNK=524288` gives 65536 tokens per rank,
which covers 58k. That alone OOMs at `MEM_FRAC=0.85` — a single 65536-token chunk
needs activation headroom that the KV pool has taken (`Tried to allocate 3.58 GiB,
2.05 GiB free`), and DP attention makes it worse by replicating attention weights
per rank instead of sharding them. `MEM_FRAC=0.80` frees ~12.8 GiB/GPU and costs
11% of the KV pool (`max_total_num_tokens` 2,194,496 → 1,949,568).

With `CHUNK=524288 MEM_FRAC=0.80` on the prefill leg, the byte-identical prompts
that failed above all pass:

| prompt tokens | before | after |
|---|---|---|
| 18,364 | fail | pass |
| 29,293 | fail | pass |
| 58,695 | fail | pass |
| 58,594 | fail | pass |

Needle is 9/9 at all three lengths × all three depths, and the whole needle set
also got 3.3× faster (90.8 s → 27.8 s) since it no longer re-runs prefill
metadata per chunk.

Full suite on the fixed configuration (`verify_pd_1p1d_singlechunk.log`):

| check | before | after |
|---|---|---|
| weights | 2/2 | 2/2 |
| basic | 7/7 | 7/7 |
| determinism | 1/1 | 1/1 |
| idle | 3/3 | 3/3 |
| **needle** | **7/9** | **9/9** |
| humaneval | 20/20 | 20/20 |
| humaneval-long | 18/20 | 19/20 |
| code-retrieval | 2/2 | 2/2 |
| deep-api | 3/3 | 3/3 |

Verdict line: `结论: 全部通过`.

This is a workaround, not a repair — the bug is still in SGLang's DSA chunked
prefill path, and the cost is that prompts longer than the per-rank chunk still
break silently. Two caveats before treating it as production config: the KV pool
is 11% smaller, and a large chunk under high concurrency is the OOM risk that bit
us at 0.85, so the c=256 load test needs re-running on this configuration. The
real fix belongs upstream in how index-K from earlier chunks is addressed by the
top-k gather.

The same read ruled out the first hypothesis: `--disable-chunked-prefix-cache` is
a no-op here. That flag controls a different, MHA-only mechanism, and `dsa` is
absent from `CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS`
(`server_args.py:133-142`), so `maybe_disable_chunked_prefix_cache`
(`misc_utils.py:16-37`) already forces it off for this model. Confirmed
empirically: a run with the flag set reproduced the failures unchanged.

#### Why the single-node recipe never saw this

`run_sglang_mtp.sh` defaults to `ENABLE_DP_ATTENTION=0`, and SGLang only divides
`chunked-prefill-size` by `dp_size` when DP attention is on. That recipe therefore
runs with a 131072-token chunk and prefills any prompt below that in one pass —
it never exercises chunked prefill at all. Our PD setup enables DP attention, the
chunk becomes 131072/8 = 16384, and every prompt over ~16k starts chunking. So
this is very likely not a PD-disaggregation bug; it is a DSA + chunked-prefill bug
that DP attention exposes by shrinking the chunk.

---

## 4. Throughput baseline, and whether one RDMA rail is a bottleneck

`bench_rails.sh`, `sglang.bench_serving`, 8192-token inputs / 128-token outputs,
`--random-range-ratio 1.0`, five prompts per concurrency slot, through the router.
Single rail (`IB_DEVICE=rdma0`), MTP off. Raw: `bench_rail_rdma0/`.

| concurrency | req/s | out tok/s | mean TTFT | p99 TTFT | mean TPOT |
|---|---|---|---|---|---|
| 16 | 1.00 | 128 | 6.6 s | 12.7 s | 65.4 ms |
| 64 | 1.84 | 235 | 22.8 s | 39.6 s | 75.1 ms |
| 256 | 2.09 | 268 | 100.4 s | 124.0 s | 77.0 ms |

Zero RDMA transfer failures on either leg across all three runs.

Same load against auto-discovered rails (`IB_DEVICE=` unset, each rank finds all
eight HCAs and opens contexts on rdma0–rdma7 on both legs):

| concurrency | req/s | out tok/s | mean TTFT | p99 TTFT |
|---|---|---|---|---|
| 16 (cold) | 0.57 | 73 | 17.7 s | 37.3 s |
| 16 (warm, repeat) | 0.53 | 68 | 19.1 s | 29.0 s |
| 64 | 1.64 | 210 | 25.1 s | 38.0 s |

Also zero RDMA errors, so multi-rail does establish QPs fine on this fabric — but
it is about **half the throughput** of a single pinned rail at c=16 (0.53 vs 1.00
req/s, TTFT 19.1 s vs 6.6 s). The warm repeat rules out JIT warmup as the
explanation. With KV needing only 3% of one rail, spreading it over eight rails
buys no bandwidth and costs 8× the QP setup and per-transfer slicing. Pinning one
rail is the faster option here, not merely the safer one.

Request throughput saturates near 2.1 req/s while TTFT grows roughly linearly with
concurrency — the prefill leg is the bottleneck and everything else queues behind
it. TPOT is flat at ~77 ms, so the decode leg is nowhere near loaded.

This settles the original question about pinning `IB_DEVICE=rdma0`. At saturation
the prefill leg produces `2.09 req/s × 8192 tok = 17,120 tok/s`, and GLM-5.2's KV
is 53.6 KB/token (MLA latent `(512+64)×78 = 43.9 KB` + DSA indexer K
`128×78 = 9.8 KB`, fp8). So KV egress peaks at **0.92 GB/s = 7.3 Gb/s**, against
the **229.5 Gb/s** a single rail measures here with `ib_write_bw`: about **3.2%**
of one rail. Saturating one rail would need ~535,000 prefill tok/s, roughly 30×
what this hardware can produce for a ~700B MoE.

So a single rail is not a throughput bottleneck at 1P1D. The reason to pin one is
correctness, not performance: each of the eight rails is its own /31 to a leaf
switch (`10.115.46.101/31`, `.111/31`, … on the prefill node; `10.115.45.x` on the
decode node), and Mooncake's auto-discovery assigns each GPU its NUMA-local HCA
(rdma0-3 → NUMA 0, rdma4-7 → NUMA 1, mirrored on both nodes). If a prefill GPU and
its decode peer land on rails that cannot route to each other, the RoCE QP never
reaches RTR and times out under load. Infera has a guard for exactly this
(`apply_mooncake_topology_default`) but it buckets by /24, so all eight rails here
look like one subnet and the guard never fires.

---

## 5. Routing: round-robin → KV-aware

Scripted but **not yet validated on hardware** — the leg restart needed for this
was not run.

Infera exposes exactly two policies, `round-robin` and `kv-aware`
(`infera/router/policy/factory.py:56-67`); `kv-aware` is in fact the upstream
default and these scripts were overriding it to `round-robin`.

The thing worth knowing: with 1P1D there is only one worker per role, so it looks
like there is nothing to route between. There is. Rank-multiplexed workers fan out
to one target per DP rank (`infera/router/policy/target.py:45-54`), so with DP8 the
policy chooses among 8 prefill and 8 decode ranks by prefix locality and load —
`DisaggRouter.dispatch` calls `policy.pick()` once per leg with a `role_hint`
(`infera/router/disagg.py:130-136`). That is exactly the knob that matters for
agentic traffic, where consecutive turns share a long prefix.

It scores `overlap_weight × (request_blocks − hits) + active_blocks`, where hits
come from a router-side mirror of each worker's cached block hashes, fed by ZMQ
KV events from the engines (`infera/router/kv_event/client.py:79-103`) — not from
a router-side radix tree. So the legs have to publish, and ours were launched
with `--no-enable-kv-events --kv-events off`.

Changes now in the scripts:

- `infera_1_server.sh`: `--router-policy ${ROUTER_POLICY:-kv-aware}` plus
  `--kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0` (prefill
  weighted 10× higher — a prefill hit skips the whole prefill, a decode hit only
  saves a little load).
- `infera_2_sglang_prefill.sh`, `infera_3_sglang_decode.sh`: `KV_EVENTS=1` swaps
  the off-flags for `--enable-kv-events --kv-events on --kv-event-transport zmq`.
  Default stays off.

Decode is deliberately left off by default: Infera appends
`--disaggregation-decode-enable-radix-cache` to a mooncake decode worker whenever
kv events are on (`infera/engine/sglang/args.py:257-263`), and SGLang rejects that
flag alongside speculative decoding, so it would collide with MTP. With decode
events off, `kv_block_size` is `None`, decode hits score 0, and decode routing
degrades to load-only — prefill still gets full prefix-aware routing, which is
where the win is.

To verify once enabled: `infera_router_pick_cache_hits` and
`infera_router_pick_request_blocks` on `/metrics`, the `pick policy=kv-aware ...
cache_hits=N request_blocks=M` router log line, and
`GET /v1/admin/cache-view/{worker_id}?dp_rank=N` for per-rank block counts.

---

## 6. Open work

1. **MTP.** Still off. Needs the draft-extend DP-gather deadlock fixed (§1.2).
   The lead is the PD-only `PREBUILT` batch and its `inner_idle_batch`, not the
   missing draft-extend CUDA graph — single-node DP8 + MTP lacks that graph too
   and works fine. First experiment: log `mlp_sync_info.global_num_tokens` per
   rank during PD warmup with `NCCL_DEBUG=WARN` and check whether the eight ranks
   agree. **Prerequisite: apply the §1.3 patch** (PR #34) first, or the first
   concurrent run after MTP comes back dies on the DSA indexer row mismatch — an
   independent defect from this deadlock. Also unexplained:
   `max_running_requests` collapsing to 6 per rank with
   MTP on, on single node as well as in PD. Note this interacts with §5, since
   decode-side kv events cannot be enabled at the same time as MTP.
2. **Upstream the chunked-prefill indexer bug.** §3.1 is a clean reproducer:
   any prompt over the per-rank chunk size, needle before the final chunk.
   Worth filing against SGLang with the length-sweep table.
3. **Re-run the load test on the fixed config.** §4's numbers were taken at
   `MEM_FRAC=0.85 / CHUNK=131072`, which is no longer what we serve. c=256 is the
   case to watch, since bigger chunks plus a smaller pool is the OOM direction.
4. **Validate kv-aware end to end** with an agentic-shaped workload (shared long
   prefix across turns) and measure the hit rate, rather than assuming it.
5. **Rail pinning is settled** — one rail, empirically faster (§4). No further
   work unless the deployment grows past 1P1D, at which point the /31-per-rail
   routing question in §4 comes back.
