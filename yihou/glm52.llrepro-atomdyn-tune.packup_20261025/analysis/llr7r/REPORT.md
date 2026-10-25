# llr7r bottleneck report: GLM-5.2 PD, PP4 prefill (135 GPU4-7) + DPA4 EAGLE decode (136 GPU4-7), C80

Config: llr7 (41,679 tok/s/GPU). Run: llr7r, a replicate on 2026-10-24 with captures in the profiling phase. The phase started 08:21:10Z per the jsonl; the brief says 08:21:19.
Sources: decode torch trace 08:46:57-59 (40 steps x 4 ranks); prefill torch trace from 08:51:57 (20 micro-batches per PP stage); 1 s probes 08:46-08:48; py-spy x3; run logs.
Verifier outputs are under `analysis_llr7r.yihou/{verify_decode,verify_prefill,verify_system,verify}/`.
Labels: **M** = measured; **I** = INFERENCE; **U** = UNKNOWN.

## 1. Executive summary (ranked)

| # | Limiter | Supporting measurement | Conf. |
|---|---|---|---|
| 1 | **The decode verify forward fills the decode step.** Each step on all 4 ranks is spent mostly inside one graph replay. | Step period p50 50.93 ms. The verify graph ends at most p50 42.3 ms / mean 42.5 ms after its launch, which is 84 % of the step. The other recorded GPU work is draft 4.12 ms, draft_extend 2.21 ms and non-graph 0.55 ms. rocm-smi on 136 reads 99-100 % util. | Window and bound **M**; "GPU-bound inside verify" **I**. What runs inside verify is **U**: the verify graph has 0 kernel records. |
| 2 | **Decode runs at a small batch regime and is capped by KV capacity.** About 10-12 running requests per rank. Token usage is 0.81-0.83 on average and is >= 0.95 for 23-28 % of the time. | When usage >= 0.95, P(prealloc>0) is 0.53-0.60 and transfer-req is 5.2-6.5, against 3.3-4.4 below 0.95. Context is about 180k tokens per request. | **M** (logs). Whether capacity limits throughput, rather than only accompanying it, is **I**. |
| 3 | **Prefill PP pipeline: about 26-29 % of each stage cycle is idle.** The idle sits in 88-101 ms gaps between steps, filled by a serial CPU handshake between stages, not by link time. | Real GPU work is 71-74 % of each clean cycle. The PP p2p transfer is about 3.8 ms per batch per link. In 33 of 33 clean batch/link pairs, the long send kernel ends 1.3-1.6 ms after the receiver's CPU enters `_pp_recv_proxy_tensors`. | Idle share **M**; "CPU handshake chain" **I**. Whether the blocking `synchronize` is device-wide is **U**. |
| 4 | **The decode host sync costs GPU idle each step.** It is the blocking `.item()` in `graph_variants.select`, called from `eagle_prepare_for_verify`. | GPU idle from the draft end to the verify launch call is p50 0.56-0.61 ms per step, plus about 0.30 ms between steps. Total is at least about 0.86 ms per step (1.7 %) and plausibly up to about 3.8 %. Py-spy finds the schedulers in `select` in 10 of 12 samples. | **M** (lower bound); upper end **I**. |
| 5 | **Late decode steps happen when prefilled requests arrive.** In 7 of 39 steps, all ranks start 2.5-3.5 ms late. | 4 of the 7 coincide with rank 0's 40-43 ms blocking `.item()` in `build_eagle_disagg_draft_input`. The other 3 are **U**. | **M** for coincidence; lockstep across ranks **I**. |
| 6 | **KV transfer and RDMA are not saturated.** | 135 tx sum is mean 16.35 / p90 33.45 / max 65.9 GB/s, which is 8.2 % of the 4 x 400 Gb/s line rate. It is above 50 % of line rate in 0-0.8 % of samples. retx and ack_timeout deltas are 0. KV transfer p50/p90 is 1.99/4.72 s (llr7). | **M** for link load. Why transfer latency is seconds when the link is idle is **U**. |
| 7 | **Decode request balance across DP ranks is uneven.** | Trace annotations show bs mean 11.95 / 10.70 / 6 (constant) / 8.88 on ranks 0-3, with the verify window equal on all ranks within 0.03 ms. | **M** for bs. The rows each rank actually computes (padding under DP-attention) is **U**. |

## 2. Decode (136, TP4/DP4 DPA, EAGLE 5/1/6)

### 2.1 Step anatomy (rank 0, 40 steps; verify-graph bound on all 4 ranks)

| Segment | p50 / mean (ms) | Notes | Conf. |
|---|---|---|---|
| Step period | 50.93 / 50.49 (p10 48.72, p90 52.41) | | M |
| Draft span | 4.31 (graph kernels 4.12) | 178 kernels on rank 0, 170 on rank 2 | M |
| Idle, draft end to verify launch call | 0.56 (range 0.49-0.64); ranks 1-3 have p50 0.58-0.61 | No GPU events on any stream. The CPU blocks in `select` (`hipMemcpyWithStream`) | M |
| Verify graph | at most 42.3 / 42.5 | 0 kernel records. Bounded by the first same-stream kernel after it, assuming in-order streams. Same on all ranks within 0.25 ms | bound M; semantics I |
| Post-verify serialized non-graph | about 0.25 | `VerifyTreeGreedy`, ArgMax, `_verify_commit_outputs`, draft_extend metadata. The CPU launched them about 1.6 ms after the draft ended; they queue behind verify | M |
| Draft_extend span | 2.38 (graph 2.21) | 54/58 kernels | M |
| Between-step idle | p50 0.019, mean 0.30, max 2.53 | | M |
| **Measurable idle** | **about 0.86 ms per step (1.7 %)** | Idle inside verify is U. If the GPU starts only when the launch call returns (p50 0.87-1.0 ms), the bubble is about 1.5 ms per step (about 3 %) | M / I |

Overlapping verify on stream 7 is about 0.1 ms per step of scheduler work (`process_batch_result`, `get_next_batch_to_run`, `alloc_extend`, `assign_req_to_token_pool`).

The trace's recorded GPU events cover only 273.7 ms of the 2053.6 ms span (rank 2). Why the verify graph is unrecorded is **U**.

### 2.2 Recorded kernels outside verify (6.88 ms per step, rank 0)

| Category | ms per step |
|---|---|
| MoE | 2.97 (0.53 per draft forward; 0.85 in draft_extend) |
| Dense GEMM | 1.27 |
| RCCL collective (`ncclDevKernel_Generic_1`, type unverified) + custom AR | 0.77 (nccl 0.631 / 15 calls; custom AR 0.134 / 5) |
| memcpy / fill / index (includes a rocprim scan of unknown role, 0.042) | 0.67 |
| Indexer compute | about 0.54 (+0.09 metadata), about 0.49 of it in draft_extend |
| Sparse MLA | 0.25 |
| Norm / rope / quant | 0.16 |
| Sampling | 0.14 |

Collective wait plus skew above p10, outside verify, is 0.40 / 0.26 / 0.27 / 0.30 ms per step on ranks 0-3 (**I**). Inside verify it is **U**.

### 2.3 Host side

- Per step, the scheduler blocks in `graph_variants.select`, reached through `run_batch -> verify -> run_eagle_verify -> eagle_prepare_for_verify -> load_batch (decode_cuda_graph_runner.py:1574) -> _resolve_attention_variant (:615) -> select (graph_variants.py:39)`. In 33 of 40 steps the block lasts 37-46 ms (it waits for the draft); in 7 it lasts 3.8-4.5 ms (**M**). The next `hipGraphLaunch` follows the return by p50 0.16 ms.
- Rank 0 also has four 40.3-43.2 ms syncs in `get_next_batch_to_run` / `build_eagle_disagg_draft_input`. They coincide with late steps 5, 6, 20 and 26, and each ends 0.26-0.27 ms after the draft_extend graph ends (**M**). Rank 2 has no such sync but is late at the same steps (lockstep, **I**).

### 2.4 Batch regime and KV capacity (logs, steady window)

| Metric | llr7 | llr7r |
|---|---|---|
| Running per rank | 9.99-11.36 | 10.22-11.63 |
| Token usage (mean) | 0.81-0.83 | 0.81-0.83 |
| Time at usage >= 0.95 | 23-25 % | 25-28 % |
| prealloc frac>0 | 0.25-0.28 | 0.25-0.31 |
| P(prealloc>0 given usage >= 0.95) | 0.55-0.60 | 0.53-0.60 |
| transfer-req per rank (overall) | 4.3-4.8 | 3.9-4.8 |
| transfer-req, usage >= 0.95 vs < 0.95 | 5.2-6.5 vs 3.3-4.4 (both runs) | |
| Running, high vs low usage | change within +/-0.7 | +0.4 to +1.3 at high usage |

- Usage plus pre-allocated is >= 0.95 in 60-65 % of samples. Pre-allocated usage is 0.18-0.22.
- 40-52 % of the prealloc wait time falls at usage below 0.95. Capacity therefore does not explain all of the admission wait (**M**); the cause of the rest is **U**.
- Verify rows: 33.3 % of steps have 48 rows or fewer (built-in gfx950 indexer), 65.2 % have 49-128 (KF), and 1.6 % have more than 128 (llr7 log). The meaning of `bs=` in `step[VERIFY bs=N]`, and whether the DPA verify graph pads to the largest bs across ranks, are **U** (code read needed).

## 3. Prefill (135, PP4 x TP1, layers 20/20/20/18, chunk 8192)

The profiling artifacts are excluded: the trace-save stalls (PP2 49.1 s; PP3 49.1 s + 45.9 s) and a disrupted interval 08:51:58.5-08:52:03.06, which has 1.25-1.33 s kernels in simultaneous pairs across stages and 10-1135 ms memcpys. Clean batches are b10-b20, excluding b16 and b17.

### 3.1 Stage balance and utilization

| | PP0 | PP1 | PP2 | PP3 |
|---|---|---|---|---|
| Real GPU work, % of clean cycle (excl. memcpy > 1 ms and nccl > 2 ms) | 72.5 | 73.6 | 76.3 | 75.3 |
| ...also excluding HiCache backup on s16 | 70.7 | 70.7 | 73.3 | 73.1 |
| Mean inter-step gap (ms), about 3.7 ms of compute inside | 101.0 | 98.7 | 88.0 | 87.7 |
| `batch_transfer_sync` union, % of active window | 25.8 | 27.1 | 26.4 | 27.8 |

- Step time: PP0 clean p50/p90 is 257.9/267.1 ms. PP3 is target p50 218.5 ms plus about 22.5 ms of draft.
- PP0 is the heaviest stage, by +8 to +12 ms in kernel sum and +12 to +17 ms in step p50. The spread across stages is about 7 % (**M**). Imbalance is therefore small compared with the gaps.
- PP0 kernel sum is 251 ms per clean batch. The hipBLASLt `Cijk` kernels take 34.4 ms and `_gemm_a16_w16` 13.3 ms; which projections they serve is **U**. PP0 has 7 indexer-cache layers against 5 on PP1-3.
- Forward-pass DtoD copies (`_get_topk_ragged`) run at about 1.1-1.2 TB/s.

### 3.2 Bubbles: what fills the 88-101 ms gaps

- The PP p2p transfer is about 3.8 ms per batch per link (kernels of 1.1, 1.6 and 1.1 ms), the same on all three links (**M**).
- The long send kernels (median 10-53 ms) end 1.3-1.6 ms after the receiver's CPU enters `_pp_recv_proxy_tensors`, in 33 of 33 clean pairs. They are waiting, not transferring (**M**).
- Each downstream post-step recv completes exactly at the upstream stage's next step end. Each `recv_requests` ends 0.3-1.0 ms after the upstream `_pp_send_dict_to_next_stage` (**M**).
- Blocking `synchronize` calls appear on the CPU: in the event loop, 2980 ms over 15 calls (about 199 ms each); in `writing_check`, PP0 536.6 + 196.1 ms and PP1 338.4 + 152.9 ms. In b10 and b13 the `writing_check` sync ends 0.6-0.8 ms after the last send kernel (**M**).
- **I**: stage k cannot launch batch b+1 until stage k+1's CPU has finished result processing, `recv_requests` and its own `synchronize`, and has posted the recv. The gap is a serial CPU handshake chain. Whether that `synchronize` is device-wide (and so waits for the comm stream) is **U**.
- Removing the gaps entirely would raise PP throughput by at most about +35 %, to about 23.6k tok/s (**I**, arithmetic). Unprofiled llr7 PP0 logs show an input throughput p50 of 21,479 tok/s. How much profiler CPU overhead inflates the gaps is **U**.

### 3.3 HiCache (write_back)

- Per stage in the window: backup 3.32-3.72 GB on stream 16, 127-147 ms of kernel time, with 46.8-57.5 ms of it outside steps. Load is about 34 ms of per-layer MLA kernels (19 layers on PP3). The effective rate is about 47 GB/s (**I**, arithmetic).
- The backup token count (299,200 per stage in about 9.5 s) exceeds the new tokens prefilled in the window (about 163k). The reason is **U**.

### 3.4 Prefill queue (PP0 log, steady window)

| | llr7 line-weighted | llr7r line-weighted | llr7 carry-forward | llr7r carry-forward |
|---|---|---|---|---|
| queue-req | 2.97 | 3.33 | 3.82 | 3.95 |
| pending-token | 0.35M | 0.40M | 0.45M | 0.45M |
| bootstrap-req | 1.82 | 2.35 | 2.40 | 2.74 |
| inflight-req | 8.50 | 8.55 | 9.18 | 8.98 |

- Batch rates are 1.69 / 1.79 per s; new tokens 11.9k / 12.6k per s; cached tokens 336k / 338k per s.
- PP0 has no batch in 12.0 % / 10.4 % of seconds. queue-req > 0 holds in 66.9 % / 68.2 % of the time (carry-forward).

## 4. Transfer, RDMA, CPU and host threads

### 4.1 KV transfer and RDMA (probes 08:46-08:48, about 133 s)

- Only ionic_4..7 are active, at 400 Gb/s per port.
- 135 tx sum: mean 16.35, p50 12.18, p90 33.45, max 65.90 GB/s (8.2 % of line rate on average). 136 rx sum: mean 16.42, p90 36.40, max 55.26 GB/s.
- Load is never above 80 % of line rate, and above 50 % in only 0-0.8 % of samples. retx and ack_timeout deltas are 0. Rails are uneven: ionic_7 / ionic_5 = 1.52x, and per-sample max/mean is p50 2.27.
- On the prefill side, there are 125-129 DtoH page-index copies per stage. Their real copy time is about 0.6 ms; the recorded totals of 486-1566 ms include blocking. No pack kernels appear in the prefill step kernel lists.
- KV transfer latency (llr7) is p50/p90 1.99/4.72 s against a nearly idle link. Which stage of the transfer path holds that time is **U** here (see the hist json for per-stage values). The mapping from rail to PP stage is **U**.

### 4.2 CPU and host threads

- **135:** 17.2 cores busy, max core window-average 37 %. GPU use 72-82 %, 747-772 W. A use dip on card5/6 at 08:47:10-20 has an unknown cause.
- **136:** 10.0 cores busy, max core window-average 76 %. GPU4-7 at 100 % with one 0 % sample at 08:47:01; the trace-export timing coincidence is **I**. Power 867-894 W.
- **136 threads:**
  - The DP controller (`sglang::data_pa`, NSpid 303) runs at 99.9 %, mostly user time. It is in `zmq/error.py` in 2 of 3 samples; a busy-poll is **I**.
  - The scheduler main threads run at 89-96 %, mostly user time. The DP0 main thread has 9.1M voluntary context switches over its lifetime, so the wait yields rather than spins; how the wait consumes CPU is **U**.
  - A native `python3` thread (NSpid 624) runs at 98.7 %, 83 % of it kernel time (lifetime ratio). Its identity is **U**: it is absent from py-spy.
  - `decode_thread` blocks in a zmq `recv_multipart` (conn.py:2973).
- The CPU file holds window averages, so "no core at 100 %" with at least 9 threads at 95-99 % means the threads migrate between cores (**I**).

## 5. Request accounting (where the 80 requests sit)

| Location | llr7r steady (approx.) | Source |
|---|---|---|
| Decode running (sum of 4 ranks) | about 41-46 (10.2-11.6 per rank); 33-40 in the trace window | decode log, trace |
| Decode transfer-req (KV in flight / prealloc) | about 16-19 (3.9-4.8 per rank) | decode log |
| Prefill inflight-req (PP0) | about 8.5-9.0 | prefill log |
| Prefill bootstrap-req | 2.35-2.74 | prefill log |
| Prefill queue-req | 3.3-4.0 | prefill log |
| Client side (tool and think time) | remainder | I |

- The sum can exceed 80; the jsonl contains `source_kind: weka_subagent` records, so subagent requests are a possible source (**U**).
- The time-bucketed trend (8 buckets; minutes 15-25 missing because of the exclusion):
  - Decode running sum: 47 / 37 / 46 at the start, 54-56 at minutes 25-35, and 40 / 36 at the end.
  - At the end of the run, prefill queue-req reaches 10.1 (llr7 9.2), pending tokens 1.42M (llr7 1.50M), decode transfer sum 26.2 (llr7 27.8), and usage about 0.90-0.93.
  - New tokens per second stay at 11-16k in llr7r. In llr7 they fall to 10.0k and then 8.4k in the last two buckets.
- llr7r vs llr7:
  - Profiling-phase completions are 9,497 vs 9,944 (-447): -624 in the profiling window, +234 in the steady window, and about -57 elsewhere.
  - The profiling window explains about 82 % of the score gap (**I**, assuming the score equals whole-run input tok/s/GPU).
  - Steady-window ISL means differ: 121,893 vs 126,290.
- gpu_hit is 0.371 in llr7r vs 0.493 in llr7. Hit tokens fell 7.7 %, but the denominator `prompt_total` rose 23 % (4.425G vs 3.604G). Why it rose is **U**; the `--server-metrics` URL order also differs between the runs.

## 6. Implications: candidate next experiments (ranked by expected impact)

Each is one variable against llr7. The expected direction is **I** until measured.

| # | Action | Evidence it targets | Confirming measurement |
|---|---|---|---|
| 1 | **Get the verify-graph breakdown** (rocprofv3 kernel trace, or a profile with the verify CUDA graph disabled). Diagnostic only; prerequisite for decode kernel work. | 84 % of the decode step is unrecorded (2.1) | Per-kernel time inside verify per rank: MoE vs sparse MLA vs indexer vs collectives, at rows 6 x bs |
| 2 | **Decode kernel work inside verify** (KF MoE / sparse MLA / indexer at verify rows 49-128, covering 65 % of steps). Pick the target after #1. | Verify is at most 42.3 ms of a 50.9 ms step; decode is at 100 % GPU | Verify p50 bound and decode step p50 fall; decode tok/s per rank at the same running count |
| 3 | **Prefill PP handshake:** remove or defer the blocking `synchronize` in the PP event loop and `writing_check`, or post the next recv earlier. Read the code first to see whether the sync is device-wide. | 88-101 ms gaps; 3.8 ms real transfer; sends wait on the receiver's CPU recv (3.2) | Inter-step gap and real-GPU-work share per stage; PP0 batches/s and input tok/s in unprofiled logs; prefill queue-req and pending tokens |
| 4 | **Decode KV capacity:** higher decode mem fraction (the py-spy cmdline shows 0.92), fewer captured graph sizes (the cmdline shows max bs 256 while running is about 12/rank), or a smaller draft KV. | Usage >= 0.95 for 25-28 % of the time; prealloc and transfer-req higher there | Time at usage >= 0.95; P(prealloc>0); running per rank; TTFT. Judge by qps, TTFT and TPOT, not score alone |
| 5 | **Decode DP balance** (router request balance) | Rank bs 11.95 / 10.70 / 6 / 8.88 with lockstep collectives | Per-rank bs spread; collective wait above p10; step p50. First read the code on padding (2.4) |
| 6 | **Remove the per-step `select` `.item()` sync** (decide the attention variant from CPU-side lengths) | At least 0.86 ms per step (1.7 %), up to about 3.8 % | GPU idle draft-end to verify start, in a trace that records verify |
| 7 | **Make the decode admission path non-blocking** (`build_eagle_disagg_draft_input` `.item()` on prefill arrival) | 4 of 7 late steps of 2.5-3.5 ms | Count and size of late steps per 40-step trace |
| - | Not indicated: more RDMA bandwidth or rail changes | Links at 8 % mean, no retx. A latency investigation (transfer-stage histogram stages) is separate | Transfer-stage histograms per stage |

## 7. Caveats

- **Profiling perturbation.** llr7r scored below llr7, mostly because of the profiling window (about 82 %, **I**). The prefill trace has large save stalls and a disrupted interval (excluded), and the profiler's CPU overhead may inflate the prefill gaps (**U**). One 0 % GPU sample on 136 coincides with the trace export.
- **Short captures.** The decode trace is 2 s (40 steps); the prefill trace has 20 batches per stage, with about 9 clean batches used. The probes cover about 133 s and the py-spy dumps are 3 samples.
- **Single replicate.** One llr7 and one llr7r. The closed-loop mix varies (steady ISL differs by 3.5 %), and the gpu_hit difference is not understood.
- **Simulated acceptance.** `SGLANG_SIMULATE_ACC_LEN=3.61` waives correctness. Acceptance is not correctness, and real acceptance would change verify rows and decode throughput.
- **Unrecorded verify graph.** The verify graph has 0 kernel records, so every decode statement about the verify interior is **U**. The 42.3 ms figure is an upper bound that assumes in-order stream semantics.
- **Unverified labels.** The `ncclDevKernel_Generic_1` collective type, the `Cijk` projection mapping, and the first and third decode graphs being draft and draft_extend (by launch order and annotation) are unverified labels.
- **Config details seen in py-spy.** The decode cmdline in the py-spy dumps shows `--mem-fraction-static 0.92`, `--cuda-graph-max-bs-decode 256`, and two `--json-model-override-args` (index_share false, then true). Check these against the intended llr7 config.
- **Verifier side effect.** A verifier left `/tmp/yihou_vdec.yihou.py` on spur007, outside a yihou dir, plus four earlier `/tmp` helper copies there. They were not deleted.
