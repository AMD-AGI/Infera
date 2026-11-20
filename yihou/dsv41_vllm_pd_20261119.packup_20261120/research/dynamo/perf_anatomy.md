# PERF ANATOMY: why Dynamo + vLLM PD on GB300/B300 is 1.9-2.4x per GPU over the MI355X vLLM colocated bar

Date 2026-11-19. Research only (no GPU, no docker). Tags: **[FH]** first-hand (API row / server log / config / code
file:line / PR read here), **[2H]** second-hand (blog, PR author claim, third-party spec site), **[INF]** our arithmetic or
inference, not measured. Sources pulled in this pass are kept under `notes/dynamo_perf/`:

- `rows_selected.json` (API rows, from `notes/ix/bench_dsv41_agentic_20261119.json`, fetched 2026-11-19),
  `derived_table.psv` (table in s1), scripts `scripts/perf_anatomy_{rows,derive,stats}.py`.
- `logs/<row>/` server logs, `config.yaml` / `recipe.lock.yaml`, `aiperf.log`, benchmark commands, downloaded via
  `https://inferencex.semianalysis.com/api/v1/server-log?id=<row>&file=<path>&download=1` (2026-11-19). The 128 MiB decode /
  frontend logs are truncated at exactly 134,217,xxx bytes by the API store; the steady window used below lies inside them.
- `source/pypi_agentx/` = `agentx-harness==1.0.14` + `aiperf-0.13.0` wheels (unzipped, read only).

Rows (all DSv4.1-Flash FP4, AgentX `semianalysis_cc_traces_weka_062126`, DSpark 5 drafts, synthetic AL 3.51, 3600 s):

| Row | HW | Engine / image | Shape | Run |
|---|---|---|---|---|
| 448242 | MI355X | vLLM `vllm-openai-rocm:nightly-60bd6dd2` | 1 server TP1xDP2 attn + TP MoE, UMBP 449 GB, c384 | IX run 43861172833 (2026-11-16 start, published 11-18) |
| 447890 | B300 | vLLM `cu134-nightly-a2bcb565` | 1 server DEP8 (TP1 DP8 EP8), c1280 | IX run 42108214901 (11-06/11-10) |
| 447896 | GB300 | vLLM `cu134-nightly-a2bcb565` | 1 server DEP4, c1024 | IX run 42241164555 (11-10) |
| 448173 | B300 | Dynamo 1.6.0.dev20261010 + vLLM `cu134-nightly-96710245` | 5P x DEP2 (10 GPU, 2 nodes) + 1D x DEP8 (1 node), c3584 | IX run 43769338888 (11-16/11-17) |
| 448172 | B300 | same | 4P x DEP2 (1 node) + 1D x DEP8 (1 node) = 16 GPUs, c2048 | same run |
| 445572 | GB300 | Dynamo + vLLM `cu134-nightly-aarch64-dev-20261019` | 12P x DEP2 (6 trays) + 1D x DEP16 (4 trays) = 40 GPUs, c9888 | IX run 39207089292 (10-19/10-20) |

---

## 0. Bottom line

1. **Per-GPU throughput factors exactly as tput/GPU = (conc/GPU) x (mean ISL+OSL) / (conc/qps).** [FH: identity holds on
   every row, e.g. 7.464 qps x 114,651 / 2 = 427.9k vs 427,451 reported]. Against the MI355X bar the B300 PD best (1.94x) is
   **1.04x concurrency per GPU x 1.58x shorter per-lane cycle x 1.18x larger mean request** [FH: s1]. Concurrency per GPU is
   NOT where NVIDIA wins: MI355X 192/GPU, B300 PD 199, GB300 PD 247, B300 colocated 160 [FH].
2. **The per-lane cycle gap is the decode speed per request.** MI355X c384 mean TPOT 50.0 ms vs B300 PD 11.2 ms; e2e 47.2 s vs
   19.7 s, TTFT 2.2 s vs 8.5 s [FH API]. Engine logs at the steady window: MI355X ~182 running/rank at ~20 output tok/s per
   request; B300 PD decode ~154 running/rank at ~84 tok/s per request (4.2x) [FH logs; INF division]. On the MI355X the
   decode step also carries the prefill (colocated); the split between "prefill interference" and "decode step cost" on
   MI355X is not measured anywhere.
3. **Gap decomposition by measured ratios (not by mechanism)** [FH rows; INF chain]:
   MI355X colocated 427,451 -> x1.464 -> B300 colocated DEP8 625,966 -> x1.323 (PD) -> B300 PD 828,209 -> x1.221 -> GB300 PD
   1,011,514 (product 2.366). Separately, **GB300 4-GPU colocated (863,068) is already 2.02x the MI355X bar** without PD or
   NVL72 scale; PD + NVL72 scale add 1.17x on GB300 (863k -> 1.01M).
4. **Hardware peak ratios are smaller than the observed ratios** [2H specs; INF arithmetic]: per GPU, B300/MI355X dense FP4
   13.5/10.07 = 1.34x, dense FP8 4.5/5.03 = 0.89x, HBM BW ~1.0x (7.7-8 vs 8 TB/s), HBM capacity ~1.0x, scale-up link
   1.8 vs ~1.07 TB/s = 1.68x, scale-up domain 8 vs 8. GB300: FP4 15/10.07 = 1.49x, NVLink domain 72. The observed colocated
   ratio (1.46x B300, 2.02x GB300) exceeds every per-GPU peak ratio except scale-up bandwidth. Which resource binds on either
   side is not measured.
5. **The most transferable reference is B300 PD, not GB300**: it runs on 8-GPU HGX nodes with the KV moved over EFA RDMA
   (`NixlConnector` push, `backends:[LIBFABRIC]`, `FI_PROVIDER=efa`), decode = one DEP8 inside one node [FH:
   logs/448173/config.yaml]. That is the same topology class as an MI355X 8-GPU xGMI node + RDMA. Its 2-node point (4P DEP2 on
   one node + DEP8 decode on the other, 16 GPUs, c2048) scores **738,119 at p90i 120.5** [FH row 448172] = 1.73x the MI355X bar,
   at 7.6x the bar's interactivity; the user's target (~641k) is 87% of it [INF arithmetic].
6. **Decode HBM**: B300 DEP8 decode loads 46.9 GiB weights/GPU and gets 192.4 GiB KV/GPU; MI355X DP2 loads 150.8 GiB weights +
   ~29 GiB non-torch and gets 61.0 GiB KV/GPU (3.15x less) [FH logs]. Wide EP on the decode node is the lever that frees KV.
7. **Prefix tier**: B300 PD prefill reports GPU hit 0.466 + CPU-offload hit 0.510 (sum 0.976 vs theoretical 0.969); MI355X c384
   GPU 0.883 + UMBP 0.057 = 0.940 vs theoretical 0.964 [FH API]. Engine-log computed-prompt rates disagree with these API hit
   numbers (s3.4); the true recompute fraction on each side is an open measurement.
8. **ROCm gaps that block a 1:1 copy** [FH code]: no redundant-expert EPLB in the ROCm DSv4.1 runner
   (`V/vllm/models/deepseek_v41/amd/model.py:167-173`), `symm_mem` EPLB is CUDA-only (`V/vllm/distributed/eplb/eplb_communicator.py:761-762`),
   any non-UMBP connector disables the AITER sparse-MLA fast path (SYNTHESIS C1), monokernel rejects DP/EP
   (`amd/mono_decode.py:316`), DeepSelect top-k is SM100-only (`V/vllm/config/kernel.py:391`).

---

## 1. Measured anatomy of the rows (API metrics, Little's-law form)

Columns from the API row metrics [FH `notes/dynamo_perf/rows_selected.json`]; derived columns [INF arithmetic,
`notes/dynamo_perf/derived_table.psv`]. `cycle` = conc / qps = seconds per completed request per lane. `in-flight/lane` =
qps x mean e2e / conc (Little). `out/GPU` = output tok/s per decode GPU (PD) or per GPU (colocated).

| Row | HW mode | GPUs P+D | conc | conc/GPU | tok/s/GPU | qps/GPU | ISL+OSL | cycle s | e2e s | TTFT s | TPOT ms | p90i | in-flight/lane | out/GPU | hit GPU / CPU / theo | KV use |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 448242 | MI355X agg | 2 | 384 | 192 | 427,451 | 3.732 | 114,651 | 51.4 | 47.2 | 2.20 | 50.0 | 15.8 | 0.92 | 3,381 | 0.883 / 0.057 / 0.964 | 0.86 |
| 448241 | MI355X agg | 2 | 256 | 128 | 401,896 | 3.101 | 129,694 | 41.3 | 30.3 | 1.40 | 30.6 | 23.7 | 0.73 | 2,958 | 0.941 / 0.006 / 0.967 | 0.54 |
| 448240 | MI355X agg | 2 | 128 | 64 | 271,329 | 2.158 | 125,776 | 29.7 | 16.6 | 0.82 | 15.5 | 45.1 | 0.56 | 2,234 | 0.949 / 0 / 0.967 | 0.24 |
| 447890 | B300 agg DEP8 | 8 | 1280 | 160 | 625,966 | 4.732 | 132,311 | 33.8 | 21.5 | 4.53 | 21.5 | 35.9 | 0.64 | 4,692 | 0.946 / 0 / 0.969 | 0.29 |
| 447896 | GB300 agg DEP4 | 4 | 1024 | 256 | 863,068 | 6.697 | 128,901 | 38.2 | 27.9 | 4.42 | 29.6 | 26.1 | 0.73 | 6,466 | 0.944 / 0 / 0.968 | 0.40 |
| 448173 | B300 PD 5P+1D | 10+8 | 3584 | 199 | 828,209 | 6.109 | 135,614 | 32.6 | 19.7 | 8.51 | 11.2 | 83.8 | 0.61 | 13,912 | 0.466 / 0.510 / 0.969 | 0.30 |
| 448172 | B300 PD 4P+1D | 8+8 | 2048 | 128 | 738,119 | 5.159 | 143,116 | 24.8 | 8.8 | 1.41 | 7.4 | 120.5 | 0.36 | 10,529 | 0.478 / 0.501 / 0.972 | 0.13 |
| 445572 | GB300 PD 12P+1D | 24+16 | 9888 | 247 | 1,011,514 | 7.454 | 135,737 | 33.2 | 19.8 | 7.78 | 12.0 | 75.5 | 0.60 | 18,824 | not reported (0/None) | None |
| 448128 | GB200 PD 5P(DEP4)+1D DEP16 | 20+16 | 5600 | 156 | 817,332 | 5.785 | 141,314 | 26.9 | 11.5 | 3.99 | 7.5 | 125.6 | 0.43 | 13,357 | 0.702 / 0.261 / 0.971 | 0.33 |

Observations (no mechanism claimed):
- **O1** Ratio chain B300 PD / MI355X = (199.1/192) x (51.4/32.6) x (135,614/114,651) = 1.037 x 1.577 x 1.183 = 1.93 (reported
  1.94) [INF on FH]. GB300 PD / MI355X = 1.287 x 1.548 x 1.184 = 2.36 (reported 2.37).
- **O2** At equal concurrency per GPU (128/GPU): B300 2-node PD 738,119 vs MI355X c256 401,896 = 1.84x; cycle 24.8 s vs 41.3 s,
  e2e 8.8 s vs 30.3 s, TPOT 7.4 vs 30.6 ms [FH rows 448172/448241].
- **O3** e2e falls 2.4x (47.2 -> 19.7 s) but cycle only 1.58x (51.4 -> 32.6 s): the non-in-flight time per request is 4.2 s on
  MI355X c384, but 11-13 s on MI355X c128/c256 and 10-17 s on every NVIDIA row [FH derived]. AgentX replays recorded
  end-to-start inter-turn delays after each response (`aiperf/timing/strategies/agentic_replay.py:1550-1580`,
  `aiperf/dataset/loader/weka_parallel_convert.py:430-450`; scenario `agentx`: `trace_idle_gap_cap_seconds 300`,
  `system_idle_gap_cap_seconds 10`, `forbid_inter_turn_delay_cap` [FH `aiperf/common/scenario/inferencex_agentx_mvp.py`]).
  Why the MI355X c384 point shows only 4.2 s per request is not established (open Q-A, s6). Consequence for planning [INF]:
  a cut in e2e buys a sub-proportional qps gain because the trace delay stays.
- **O4** Mean ISL+OSL of completed requests is lowest on the MI355X c384 point (114.7k vs 125.8-129.7k at its own c128/c256
  and 128.9-146.4k on NVIDIA rows) [FH]. Mechanism not established (Q-B). It is worth 1.18x of the B300-PD gap.
- **O5** Interactivity: every NVIDIA PD best point sits at p90i 75-120 vs the MI355X bar at 15.8 [FH]. The NVIDIA wins are not
  bought by trading interactivity; at matched interactivity the gap is larger (B300 colocated c1024 586,571 @ 50.5 vs MI355X
  c128 271,329 @ 45.1 = 2.16x) [FH rows 447889/448240].
- **O6** Cache-bust marker is per lane at token 0 of the first turn (`CacheBustTarget.FIRST_TURN_PREFIX`, "diverge at token 0")
  [FH `aiperf/common/enums/enums.py:248-262`]; c3584 samples 3584 lanes from 393 traces ("avg 9.1 lanes per trace") and c9888
  25.2 lanes per trace [FH logs/448173/aiperf.log, logs/445572/aiperf.log]. So high concurrency does not create cross-lane
  prefix sharing; prefix hits are per session on all rows [INF from FH].
- **O7** Power: MI355X 1,729 W for 2 GPUs (865 W/GPU, board max 1,400 W [2H AMD]); B300 PD 17,157 W / 18 = 953 W/GPU; B300
  colocated 870 W/GPU [FH API `avg_total_gpu_power_w`]. Reported only; no utilisation inference.

---

## 2. Hardware ratios (per GPU)

| Item | MI355X | B300 (HGX) | GB300 (NVL72) | B300/MI355X | Source |
|---|---|---|---|---:|---|
| Dense FP4 (MXFP4 / NVFP4) | 10.07 PF | 13.5 PF (108 PF / 8) | 15 PF (1,080 PF / 72) | 1.34 (GB300 1.49) | [2H] AMD MI355X page/brochure; NVIDIA HGX B300 + DGX B300 datasheet via acecloud/pantheon; nvidia.com/gb300-nvl72 |
| Dense FP8 | 5.03 PF | ~4.5 PF (72 PF sparse / 8 / 2) | ~5 PF | 0.89 | same |
| HBM capacity | 288 GB | 270-288 GB (NVIDIA docs disagree) | up to 288 GB | ~0.94-1.0 | same |
| HBM bandwidth | 8.0 TB/s | ~7.7-8 TB/s | ~8 TB/s (576 TB/s / 72) | ~1.0 | same |
| Scale-up link / GPU | 7 x ~153 GB/s = ~1.07 TB/s (AMD: 1,136 GB/s P2P incl. PCIe) | 1.8 TB/s NVLink 5 | 1.8 TB/s, 72-GPU domain (130 TB/s) | ~1.68 | same; 153 GB/s/link is a third-party figure |
| Scale-up domain | 8 | 8 | 72 | 1 | same |
| Scale-out NIC / GPU | openai smci355: **unknown** (U1) | 800 Gb/s CX-8 per GPU on HGX B300 spec; the InferenceX B300 cluster used EFA (`FI_PROVIDER=efa`), rate not stated | MNNVL used for KV (`UCX_TLS=cuda_copy,cuda_ipc,tcp`, `UCX_CUDA_IPC_ENABLE_MNNVL=y`) | - | [2H] spec; [FH] configs |
| Host coupling | PCIe Gen5 x16, EPYC host | PCIe, x86 host | Grace + NVLink-C2C, LPDDR5X (17 TB/rack) | - | [2H] |

Reading [INF, arithmetic only]: on raw peaks a B300 GPU has 1.34x the FP4 compute, parity HBM, 0.89x FP8 and 1.68x link
bandwidth. These are not predictions of the attainable ratio. The vLLM team reports decode MoE-bound and weight-read-bound
(DEP4, 128K, 64 req/rank: MoE 62% of target step) [2H: vLLM blog 2026-10-23, `notes/critic/vllm_blog_2026-10-23_dsv41_pd_wide_ep.md:63,78`],
measured on GB200/GB300 only; whether MI355X decode is bound the same way is not measured (E6).

---

## 3. Per-role anatomy from the engine logs

All from vLLM Rust-frontend `log_stats` lines in the middle 60% of the active span [FH `scripts/perf_anatomy_stats.py` on
`notes/dynamo_perf/logs/*`]. "Running" etc. are totals per engine (all DP ranks of that worker). "prompt tput" is the engine's
own counter; its exact definition (computed vs. scheduled tokens) is not verified.

### 3.1 Decode

| Row / role | Window | gen tok/s (median) | Running (median) | per rank | Waiting / Deferred | KV used | per-request tok/s [INF] |
|---|---|---:|---:|---:|---|---:|---:|
| 448173 B300 D DEP8 | 11-16 21:00-21:58 | 103,741 | 1,230 | ~154 (cap 168) | 54 / 41 | 14.1% | ~84 |
| 445572 GB300 D DEP16 | 10-20 00:16-01:08 | 77,840 | 1,006 | ~63 (cap 253) | 226 / 74 | 23.8% | ~77 |
| 447890 B300 agg DEP8 | 11-06 00:50-01:29 | 24,479 | 756 | ~95 (cap 192) | 94 / 0 | 15.3% | ~32 |
| 448242 MI355X agg DP2 | 11-16 14:14-14:50 | 7,265 | 364 | ~182 (no cap) | 8 / 0 | 64.6% (API 85.8%) | ~20 |

- With AL 3.51 per DSpark step, ~84 tok/s per request corresponds to ~42 ms per decode step on B300 PD at ~154 rows/rank
  (x6 tokens = ~924 tokens/rank/step); MI355X ~20 tok/s corresponds to ~175 ms per AL-3.51 step-equivalent including
  interleaved prefill [INF arithmetic; assumes every running request advances every step].
- On B300 the same software went from ~32 tok/s/request colocated (prefill-schedule-interval 96) to ~84 in PD decode [FH logs;
  INF division]; i.e. on B300 the colocated per-request decode rate is ~2.6x lower than the PD decode rate. No equivalent
  MI355X split exists. Colleague anchor: at TP2 c32 rank-0 device time was decode 78% / prefill 22% [2H: chaox README via
  research/colleagues.md]; nothing at c384.
- GB300 decode at ~63 rows/rank still runs ~77 tok/s/request (~45 ms/step) on the 10-19 image; the blog lists CPU-side
  fixes (async KV admits cap, recursive-doubling DP all-reduce, admission cost) made after/around that image [2H blog table].
  Not attributed here.

### 3.2 Decode memory (startup lines)

| Row / role | Weights loaded / GPU | consumed (weights + non-torch) | KV memory / GPU | engine "GPU KV cache size" tokens / rank |
|---|---:|---:|---:|---:|
| 448173 B300 D DEP8 | 46.94 GiB | 53.43 GiB | 192.39 GiB | 188.4M |
| 445572 GB300 D DEP16 | 28.95 GiB | - | 202.09 GiB | 188.9M |
| 448173 B300 P DEP2 | 149.21 GiB | 157.23 GiB | 88.54 GiB | 47.9M |
| 445572 GB300 P DEP2 | 150.94 GiB | - | 86.34 GiB | 31.2M |
| 447890 B300 agg DEP8 | 47.62 GiB | - | 112 GiB (`kv-cache-memory-bytes 120259084288`) | 24.3M |
| 448242 MI355X agg DP2 | 150.82 GiB | 180.01 GiB | 61.02 GiB | 29.9M |

[FH: `Model loading took`, `Free memory on device ... Actual usage`, `Available KV cache memory`, `GPU KV cache size` lines in
the logs above.] The engine "tokens" figure is not comparable across configs (B300 P: 88.5 GiB -> 47.9M; B300 D: 192.4 GiB ->
188.4M; B300 agg: 112 GiB -> 24.3M); the per-token accounting under prefix caching / SWA groups is not read. Compare GiB.
MI355X non-torch memory ~29 GiB vs B300 decode ~6.5 GiB [FH; cause not read].

### 3.3 Prefill

| Row / role | prompt tput / worker (median, mean) | per GPU (mean) [INF] | Running / Waiting (median) | prefix hit / external hit (median) | KV used |
|---|---|---:|---|---|---:|
| 448173 B300 P w0 (DEP2) | 88.2k, 112.2k tok/s | ~56k | 7 / 18 | 96.2% / 0.0% | 7.0% |
| 445572 GB300 P w0 (DEP2) | 140.8k, 136.1k | ~68k | 15 / 53 | 88.9% / 55.2% | 8.4% |
| 448242 MI355X agg (whole server) | 32.1k, 34.0k | ~17k (plus decode) | 364 / 8 | 90.4% / 60.5% | 64.6% |

Reference [2H, blog]: GB300 DEP2 new-token prefill 71.2K -> 102.5K tok/s/GPU at 64K new on a 128K cached prefix after #59894,
#62359, #62240; AgentX prefill-only c256 DEP2 64.4K vs DEP4 53.3K tok/s/GPU.

### 3.4 Prefix hit: API vs engine counters disagree (open)
- API: B300 PD GPU 0.466 + CPU 0.510 = 0.976 > theoretical 0.969; MI355X 0.883 + 0.057 = 0.940 < 0.964 [FH].
- Engine prompt counters vs API input: B300 PD 5 workers x ~112k = ~0.56M/s vs 14.8M/s input = 3.8% computed; MI355X ~33k/s vs
  0.849M/s = 3.9% [INF on FH]. If the prompt counter is "computed tokens", the two sides recompute a similar fraction, which
  contradicts the API hit gap (2.4% vs 6.0% uncached). Neither reading is established. Measurement: per-request
  `num_cached_tokens` / computed-token counters from the MI355X R2 reproduction (SYNTHESIS E2/E3).
- B300 PD sum > theoretical: how `server_*_cache_hit_rate` is aggregated for PD rows (prefill-only vs both roles) is not read.
- GB300 vLLM PD row has no hit or KV-usage metrics in the API (0 / None) [FH]; its prefill log shows external hit ~55% [FH].

---

## 4. Decomposition of the gap (what each step contains)

Each factor below is a measured ratio between two published rows [FH]; what it contains is listed, but no row pair isolates a
single cause.

| Step | Ratio | Rows | Contains (confounded) |
|---|---:|---|---|
| A. MI355X colocated -> B300 colocated | 1.464 | 448242 -> 447890 | hardware (FP4 1.34x, links 1.68x, HBM parity), engine/kernels (FlashMLA mega attn, DeepGEMM MegaMoE, DeepSelect, torch.compile + FULL_AND_PIECEWISE graphs vs breakable graphs on ROCm), shape (DEP8 wide EP, 47.6 vs 150.8 GiB weights/GPU, 112 vs 61 GiB KV), scheduler (prefill-schedule-interval 96 + mbt 65,536 vs 11 + 16,384), router (consistent_hash both), no host offload on B300 vs UMBP on MI355X, interactivity 35.9 vs 15.8 |
| A'. MI355X colocated -> GB300 colocated | 2.019 | 448242 -> 447896 | as A plus Grace/C2C host (Engram host tables on Grace), DEP4 not DEP8, 128 GiB KV/GPU, max-num-seqs 320/rank |
| B. B300 colocated -> B300 PD | 1.323 | 447890 -> 448173 | PD itself (decode without prefill interference, TPOT 21.5 -> 11.2 ms), prefix tier (CPU offload hit 0.51), Dynamo KV router + session affinity, 18 vs 8 GPUs, newer image (a2bcb565 -> 96710245), p90i 35.9 -> 83.8 |
| C. B300 PD -> GB300 PD | 1.221 | 448173 -> 445572 | 40 vs 18 GPUs, DEP16 over NVL72 MNNVL vs DEP8 in one HGX node, KV over MNNVL cuda_ipc vs EFA, Grace, FP4 1.11x, older image (10-19), p90i 83.8 -> 75.5 |
| Product A x B x C | 2.366 | | = 1,011,514 / 427,451 |

Within-hardware facts that bound the "PD" and "scale" shares [FH rows]:
- PD on one SKU: B300 PD best / B300 colocated best = 1.323; GB300 PD best / GB300 colocated best = 1.172.
- Scale on one SKU: B300 PD 16 GPUs (2 nodes) 738,119 -> 18 GPUs 828,209 (+12%, also c2048 -> c3584 and p90i 120 -> 84);
  GB300 DEP32 decode (64 GPUs) 954,146 < DEP16 (40 GPUs) 1,011,514.
- So of the 2.37x (GB300 PD vs bar), ~2.02x already exists between single-server colocated points (A'); PD+NVL72 scale add ~1.17x
  [INF from FH ratios]. Of the 1.94x (B300 PD vs bar), 1.46x is colocated-vs-colocated and 1.32x is PD on the same SKU.

What we cannot split from public data (needs our runs, SYNTHESIS E3/E4/E6): how much of A is hardware vs ROCm kernels vs shape
(DP2+TP MoE vs DEP8); how much of B is decode-isolation vs prefix tier vs router.

---

## 5. NVIDIA recipe details that differ from the MI355X bar (B300 PD = the template)

From `notes/dynamo_perf/logs/448173/config.yaml` [FH] unless marked; GB300 differences from `logs/445572/config.yaml` [FH].

**Frontend / router (Dynamo 1.6.0.dev20261010):** `router-mode: kv` (KV-event-aware), `router-session-affinity-ttl-secs 14400`,
`DYN_ROUTER_TEMPERATURE=0`, `DYN_TOKENIZER=fastokens` with an 8 GB tokenizer cache, prefill `kv_events: true`, client sends
`X-Dynamo-Session-ID` from the correlation id (`AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true`,
`AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0`). GB300 adds `DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT=10000000` ("decode worker
selection by active request count with soft session affinity" [FH IX #4336 changelog]). MI355X bar: `vllm-router 0.1.14
--policy consistent_hash` over 2 DP ranks [FH reproduce_448242].

**Prefill workers (5 x 2 GPUs, B300):** TP1 DP2 + EP, `MultiConnector[NixlConnector(transfer_mode push, backends LIBFABRIC,
kv_buffer_device cuda) + SimpleCPUOffloadConnector(cpu_bytes_to_use 75,161,927,680 = 70 GiB, lazy_offload)]`; GB300 uses the
`push_as_nixl` module over UCX/MNNVL and 80 GiB (85,899,345,920 B) CPU offload; `kv-cache-dtype fp8`; attention backend
`FLASHMLA_MEGA_ATTN_DSV41` + MXFP4 indexer; `kernel-config {moe_backend: deep_gemm_mega_moe, sparse_indexer_topk_backend:
deep_select, enable_flashinfer_autotune}`; Engram `cpu_offload: true, embedding_across_dp: true`; `prefill-schedule-interval 2`;
`max-num-seqs 64`; `max-num-batched-tokens 16384` (GB300 32768); `long-prefill-token-threshold 8192`;
`prefix-cache-retention-interval 32768`; `VLLM_COMPILE` + `FULL_AND_PIECEWISE` with capture sizes to 2046 and
`decoder_replay_cudagraph_capture_sizes`; `gpu-memory-utilization 0.95`; `VLLM_USE_V2_MODEL_RUNNER=1`;
`VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1`; `numa-bind`; `enable-ep-weight-filter`. Speculative config identical to decode.

**Decode worker (1 x DEP8 B300 / DEP16 GB300):** TP1 DP8 + EP, NixlConnector `kv_consumer` push; `no-enable-prefix-caching`;
`enable-eplb` with `{communicator: symm_mem, num_redundant_experts: 16, window_size 1000, step_interval 3000, use_async}`;
`max-num-seqs 168` / `max-num-batched-tokens 1680` (GB300 253 / 2530); `FULL_DECODE_ONLY` graphs at every multiple of 6 up to
1008 (= 168 x 6); `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP=4` (GB300 6); per-rank NUMA binding list; KV memory 192 GiB/GPU.

**Colocated NVIDIA (B300 DEP8 / GB300 DEP4):** `prefill-schedule-interval 96`, `max-num-batched-tokens 65536`,
`long-prefill-token-threshold 8192`, `max-num-seqs 192` / `320`, explicit `kv-cache-memory-bytes` 112 / 128 GiB, capture sizes to
8190, Engram in HBM (B300) or host (GB300), vllm-router consistent_hash [FH: logs/447890, logs/447896 recipe.lock.yaml].

**Semantics of the less-known flags** [FH code, vLLM main 7a62ca0666]:
- `prefill_schedule_interval`: admit prefill only every N engine steps, other steps decode alone (`V/vllm/config/scheduler.py:239-246`).
- `prefix_cache_retention_interval`: token interval between retained sliding-window prefix-cache checkpoints; 0 keeps only
  semantic checkpoints (`V/vllm/config/cache.py:286-292`). Applies to DSv4.1's SWA groups; effect on hit not measured.
- `cache_aware_admission_window` exists (admit cached-prefix requests first within a window) (`scheduler.py:248-260`); not used by
  any of these recipes.
- `max_async_kv_loads_per_step` is the main-branch config for capping async KV loads per step (`scheduler.py:90-98`); the env name
  `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` used in both NVIDIA configs is not present in vLLM main (grep) - it belongs to the
  pinned images' code; equivalence not verified.
- `embedding_across_dp`: shard Engram embeddings across TP and all DP ranks (`V/vllm/config/engram.py:52-54`).
- `enable_ep_weight_filter`: read only local experts at load (`V/vllm/config/parallel.py:310-318`).
- NIXL push mode is on main (`V/vllm/distributed/kv_transfer/kv_connector/v1/nixl/push_scheduler.py`, `push_worker.py`).
- SimpleCPUOffloadConnector has ROCm code paths (`V/vllm/v1/simple_kv_offload/cuda_mem_ops.py:83-95`), but neither it nor
  NixlConnector overrides `only_restores_canonical_kv` (only UMBP and MultiConnector-of-all-canonical do:
  `V/.../kv_connector/v1/base.py:237`, `umbp/connector.py:110`, `multi_connector.py:274-280`), so on ROCm either turns off the
  AITER sparse-MLA fast path for DSv4.1 (SYNTHESIS C1).

**Other engines' published lessons on the same workload** [FH IX changelog `notes/ix/changelog_dsv41.yaml`]:
- SGLang GB300 PD: cap the decode batch "at the rank's equilibrium batch so the queue moves to TTFT instead of decode step time"
  (IX #4673).
- SGLang GB300 PD: a Mooncake external L3 prefix pool "whose layer-wise L3 reads block the forward pass and stall prefill after the
  client's turn burst" was replaced by a local host-DRAM HiCache tier on prefill (IX #4595); an earlier entry reports the external
  pool's higher hit let c1228 run 3 instead of 4 DEP2 prefill workers.
- vLLM GB300 PD: `max-num-seqs 126` chosen as "a MegaAttn wave boundary"; decode selection by active request count (IX #4336).

---

## 6. Open questions raised by this pass

- **Q-A** Why is the per-request non-in-flight time 4.2 s on MI355X c384 vs 11-13 s on its own c128/c256 and 10-17 s on NVIDIA rows
  (O3)? Measurement: per-request start/end and per-lane timelines from the aiperf export of an MI355X c384 reproduction (the
  CI run's aiperf artifacts were not downloaded; only `aiperf.log`).
- **Q-B** Why is the mean completed-request ISL+OSL 114.7k at MI355X c384 vs 125-146k elsewhere (O4)? Same measurement (per-turn
  index / trajectory progress per lane).
- **Q-C** API hit rates vs engine prompt counters (s3.4).
- **Q-D** MI355X colocated decode step: share spent on prefill chunks vs decode at c256/c384 (device-time split), and the
  decode-only step at 150-180 rows/rank for DP2 vs DEP8 (SYNTHESIS E3/E6).
- **Q-E** MI355X DEP8 colocated: author-local 182.7k (#5056) vs vllm#67377 26,714 (SYNTHESIS Q8) - needed before choosing the
  decode-node shape.
- **Q-F** openai-cluster NIC inventory (U1/G4): B300 moved the KV over EFA; our wire is unknown.

---

## 7. What is transferable to MI355X 8-GPU nodes + RDMA (openai smci355: 8 x MI355X per node, `Compute-Multinode` partition
MinNodes=2, QOS batch <= 4 nodes/user [FH ~/.claude/skills/machine-pool/clusters/openai.md]; NIC/RDMA inventory unknown)

Transferable in topology (same class as B300 PD):
- **T1 Shape**: P = N x DEP2 prefill workers on prefill node(s) + one DEP8 decode worker per 8-GPU node; KV over RDMA push.
  B300 2-node point (4P + 1D, c2048) is the direct analogue of 2 MI355X nodes; 3-node (5P over 2 nodes + 1D) the next step.
- **T2 Decode HBM via wide EP**: DEP8 gives 47 GiB weights/GPU on B300 vs 151 GiB on MI355X DP2 -> ~3x KV per GPU [FH]. On
  MI355X this needs a working DSv4.1 DP8+EP decode path (AITER MegaMoE over MoRI exists: `moe_backend aiter_mega_moe`,
  `V/vllm/config/kernel.py:146`, `amd/dspark.py:320`; perf conflict Q-E).
- **T3 Decode batch cap at equilibrium + full-decode graphs**: `max-num-seqs` per rank sized to the step-time knee
  (B300 168, GB300 253/126) with `FULL_DECODE_ONLY` capture at multiples of (1+drafts). ROCm uses breakable graphs
  (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`, FULL_AND_PIECEWISE aborted per IX #2962); whether FULL_DECODE_ONLY works for DSv4.1 on ROCm
  is not checked.
- **T4 Admission rate cap** for async KV loads (`max_async_kv_loads_per_step`, merged #61725; blog 460K -> 516K with two other
  fixes [2H]).
- **T5 Prefill-side prefix tier with session/KV-aware routing**: NVIDIA prefill gets ~51% of input from the CPU tier (70-80 GiB per
  2-GPU worker) [FH]. On ROCm: UMBP keeps the fast path but is not a PD connector; NIXL/Mooncake/MoRIIO + any offload loses it
  (C1) - cost to be measured (E4) before choosing.
- **T6 Router**: KV-event-aware routing + session affinity (Dynamo `router-mode kv`, TTL 14400) for prefill; least-active-requests
  for decode. Alternatives on our side: vllm-router policies; whether Dynamo's Rust frontend runs against vLLM ROCm workers is not
  checked.
- **T7 Prefill scheduler knobs**: `max-num-seqs 64`, `long-prefill-token-threshold 8192`, `max-num-batched-tokens 16-32k`,
  `prefill-schedule-interval 2`, `prefix-cache-retention-interval 32768`, `numa-bind`, `enable-ep-weight-filter`,
  `embedding_across_dp`, DEP2 (blog: DP-imbalance loss 5.9% DEP2 vs 13.6% DEP4 [2H]).
- **T8 Colocated knobs (if colocated stays the vehicle)**: NVIDIA colocated runs `prefill-schedule-interval 96` with
  `max-num-batched-tokens 65536` and capped `max-num-seqs`; MI355X bar uses 11 / 16384 / uncapped [FH]. Untested on MI355X.
- **T9 CPU-path fixes already merged upstream**: recursive-doubling CPU all-reduce #61767, prompt int32 #62702, SWA null-block skip
  #61724, chunked staged writes #62698 [FH merged per SYNTHESIS C6]; check that the ROCm image used contains them.
- **T10 Prefill kernel items from the blog**: layer-20 bounded replay #59894, Engram 2 MB pages #62359, DeepSelect prefill top-k
  #62240 [2H gains on GB300]; ROCm status per item not read (DeepSelect is SM100-only; ROCm has `persistent` top-k,
  `kernel.py:394`).
- **T11 Methods**: `vllm bench forward` (#61339), mirror/lag replay for DP imbalance, real-vs-uniform routing for EP balance,
  `DecodeBenchConnector` (with the ROCm fast-path caveat, SYNTHESIS C1).

Not transferable (or needs new ROCm work):
- **N1** 72-GPU NVLink domain (DEP16/DEP32 decode, KV via MNNVL cuda_ipc) - MI355X scale-up domain is 8; beyond 8 the EP
  all-to-all crosses RDMA.
- **N2** Redundant-expert EPLB: ROCm DSv4.1 runner has `n_redundant_experts = 0` (`amd/model.py:167-173`); `symm_mem` EPLB
  communicator raises off CUDA (`eplb_communicator.py:761-762`). Blog: EPLB balancedness 0.35 -> 0.94 target at DEP32 [2H].
- **N3** CUDA kernels: FlashMLA mega attention, DeepGEMM MegaMoE, DeepSelect, FlashInfer autotune; ROCm equivalents are AITER
  Gluon sparse MLA, AITER CK a8w4 / AITER MegaMoE, persistent/per-row top-k.
- **N4** Grace/NVLink-C2C host memory for Engram and KV offload.
- **N5** Raw FP4 peak 1.34-1.49x.

Levers list for the experiment plan (ordered by how directly a public NVIDIA measurement supports them; none measured on MI355X):
1. PD 1P-node(4xDEP2 or other P shape) + 1D-node(DEP8) at c~2048 (T1, T2) - B300 analogue 738k.
2. Decode DEP8 + EP to free HBM, decode batch cap at equilibrium, decode-only graphs (T2, T3).
3. Prefill prefix tier sized like NVIDIA (~35-40 GiB host DRAM per prefill GPU) + KV-aware/session routing (T5, T6), after E4.
4. Async KV admission cap (T4) and the merged CPU-path fixes (T9).
5. Prefill knobs (T7) and colocated knobs (T8) as cheap A/Bs on the colocated reproduction.
6. ROCm work items: redundant EPLB for DSv4.1 (N2), connector fast-path override safety (SYNTHESIS G1), FULL_DECODE_ONLY graphs.

---

## Sources

- InferenceX API rows / logs: `https://inferencex.semianalysis.com/api/v1/benchmarks?model=DeepSeek-V4.1-Flash&sequence=agentic-traces`,
  `/api/v1/server-log-files?id=`, `/api/v1/server-log?id=&file=&download=1` (fetched 2026-11-19).
- IX runs: https://github.com/SemiAnalysisAI/InferenceX/actions/runs/43861172833, /43769338888, /39207089292, /42108214901, /42241164555.
- IX PRs: #4231/#4585 (B300 Dynamo vLLM), #4336, #4266, #4595, #4673, #2962, #5036, #5056.
- vLLM blog 2026-10-23: https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep (copy in `notes/critic/`).
- Hardware: [AMD MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html),
  [AMD MI355X platform](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x/platform.html),
  [NVIDIA GB300 NVL72](https://www.nvidia.com/en-us/data-center/gb300-nvl72/),
  [HGX B300 spec summary (acecloud)](https://acecloud.ai/blog/nvidia-hgx-b300/), [Pantheon HGX B300](https://pantheon.run/learn/nvidia-hgx-b300-specs),
  [gpupoet MI355X (link rate)](https://gpupoet.com/gpu/learn/card/amd-instinct-mi355x).
- Harness: PyPI `agentx-harness==1.0.14`, `aiperf==0.13.0` (unzipped in `source/pypi_agentx/`).

---

## Verification (adversarial)

Verifier pass 2026-11-19. Every source re-opened independently: API re-fetched to
`notes/dynamo_perf/verify/bench_refetch.json` (`/api/v1/benchmarks?model=DeepSeek-V4.1-Flash&sequence=agentic-traces`,
227 rows), `notes/dynamo_perf/verify/files_448172.json` (`/api/v1/server-log-files?id=448172`); log windows re-parsed with an
independent script `scripts/verify/pa_logstats.py` (active span = gen tput > 100 tok/s, or prompt tput > 1000 for prefill;
middle 60%). Verdicts: CONFIRMED / CORRECTED / REFUTED / UNVERIFIABLE.

| # | Claim (short) | Verdict | Correct fact and evidence |
|---|---|---|---|
| V1 | tput/GPU = (conc/GPU) x (ISL+OSL) / (conc/qps); B300 PD/MI355X = 1.037 x 1.577 x 1.183 = 1.93; conc/GPU 192 vs 199 | CONFIRMED (with a caveat) | [FH re-fetch] 448242: 2 GPUs, c384, `mean_qps` 7.46387, `tput_per_gpu` 427,451, p90i 15.80, image `vllm-openai-rocm:nightly-60bd6dd2...`; 448173: 10P+8D, c3584 (199.1/GPU), `mean_qps` 109.95, 828,209, p90i 83.82, image `cu134-nightly-96710245...`, router `dynamo-router 1.6.0.dev20261010`. 1.037 x (51.45/32.60) x (135,614/114,651) = 1.934; reported ratio 1.9375. Caveat [INF]: the identity is definitional (it only re-expresses tput = qps x tokens/request); `total_tput_tps/mean_qps` gives 114,539 / 135,582 tok/request (vs 114,651 / 135,614 from mean input + mean output), so the "1.18x larger request" factor is 1.184 either way. conc is the configured lane count, not measured in-flight. |
| V2 | Mean TPOT 50.0 / 11.2 / 12.0 ms; e2e 47.2 / 19.7 / 19.8 s; TTFT 2.2 / 8.5 / 7.8 s | CONFIRMED | [FH re-fetch] `mean_tpot` 0.05005 / 0.01118 / 0.01203; `mean_e2el` 47.23 / 19.71 / 19.83; `mean_ttft` 2.199 / 8.508 / 7.784 (rows 448242 / 448173 / 445572; 445572 = GB300 24P+16D, c9888, p90i 75.47, `cu134-nightly-aarch64-dev-20261019`, 2026-10-20). |
| V3a | B300 PD decode DEP8 median 1,230 running (~154/rank, cap 168), 103.7k gen tok/s, ~84 tok/s/request | CONFIRMED (approx.) | [FH logs/448173 decode_w0, one aggregated line per second] my window 11-16 20:59:59-21:57:59: median gen 105.8k, median running 1,231, median waiting 54 / deferred 49 (doc: 41). Per request [INF]: 105.8k/1,231 = 86; mean-over-window 82.8k/825 = 100; API 1/mean TPOT = 89 tok/s. Range 84-100, centre ~89. |
| V3b | MI355X agg median 364 (~182/rank) at 7.27k gen, ~20 tok/s/request | CONFIRMED | [FH logs/448242, 10 s interval] window 14:13:49-14:50:09: median gen 7,265 (mean 7,209), median running 364; 7,265/364 = 20.0 = 1/TPOT (50.05 ms). No `max-num-seqs` in the recipe (`recipe.lock.yaml`), so the vLLM default applies (1,024 per engine on large-memory GPUs, `V/vllm/engine/arg_utils.py:3027-3029`, main; image default not read). |
| V3c | B300 colocated DEP8 ran 756 at 24.5k gen, ~32 tok/s/request | **REFUTED** | [FH logs/447890] the 1 s log lines are bimodal under `prefill-schedule-interval 96`: in the window 34% of lines have gen < 5k tok/s and 47% have prompt tput > 100k (prefill-only seconds); gen p25/p50/p75 = 2.9k / 24.5k / 84.5k. The median is not a rate. Mean gen 39.3k / mean running 756 = 52 tok/s/request; API 447890 `mean_tpot` 21.47 ms -> 46.6 tok/s; API `output_tput_tps` 37,527 / (1280 x in-flight 0.637) = 46 [INF]. Correct: ~46-52 tok/s/request. Consequence: the s3.1 statement "B300 colocated -> PD decode ~2.6x per-request rate" becomes **~1.9x** (TPOT 21.5 -> 11.2 ms; 46.6 -> 89 tok/s). |
| V4 | Decode HBM: B300 DEP8 46.94 GiB weights / 192.39 GiB KV; GB300 DEP16 28.95 / 202.09; MI355X DP2 150.82 weights, 180.01 consumed, 61.02 KV | CONFIRMED | [FH] logs/448173 decode: `Model loading took 46.94 GiB` (all 8 ranks), `Available KV cache memory: 192.39 GiB`, consumed 53.43-53.44 GiB, gpu-mem-util 0.95 of 267.69 GiB; 445572 decode: 28.95 / 202.09 GiB, consumed 48.45, util 0.95 of 276.62; 448242: 150.82 / 61.02 GiB, consumed 180.01, peak activation 18.15 + CUDA graph 11.5 GiB, **gpu-memory-utilization 0.9** (vLLM's own log line: `--kv-cache-memory=77921342976` (72.57 GiB) "to fully utilize gpu memory"). Extra fact: the MI355X bar leaves ~11.5 GiB/GPU of KV on the table at util 0.9 vs NVIDIA's 0.95. |
| V5 | Ratio chain 427,451 -> 625,966 (x1.464) -> 828,209 (x1.323) -> 1,011,514 (x1.221); GB300 DEP4 863,068 = 2.02x; GB300 PD/colocated 1.17x | CONFIRMED | [FH re-fetch] 625,966.0/427,451.4 = 1.4644; 828,209.3/625,966.0 = 1.3231; 1,011,513.5/828,209.3 = 1.2213; 863,067.9/427,451.4 = 2.0191; 1,011,513.5/863,067.9 = 1.1720. Each is the best published row of its (hardware, framework, disagg) class in the re-fetch (447890 c1280 p90i 35.86 2026-11-10 `cu134-nightly-a2bcb565`; 447896 c1024 p90i 26.06). For completeness the best GB300 Dynamo-SGLang PD is 1,071,944 (row 448095, c7480, p90i 105.6). |
| V6 | B300 2-node PD (4x DEP2 P on one node + DEP8 D on one node, 16 GPUs, c2048) 738,119 at p90i 120.5 = 1.73x; 1.84x MI355X c256 401,896 | CONFIRMED | [FH] row 448172: 8P+8D, 4 prefill workers, c2048, 738,118.6, p90i 120.48, run 43769338888 attempt 2; server-log file list shows `dsxe-sa-b300-prd0-gpu-08_prefill_w0..w3.out` and `dsxe-sa-b300-prd0-gpu-09_decode_w0.out` (one node each). 738,119/427,451 = 1.727; /401,896 (row 448241, c256, p90i 23.75) = 1.837. |
| V7 | B300 PD KV over EFA (NIXL push, LIBFABRIC, FI_PROVIDER=efa); prefill MultiConnector + SimpleCPUOffload **70 GiB lazy per 2-GPU worker**; decode DEP8, no prefix cache, EPLB 16 redundant via symm_mem, max-num-seqs 168, FULL_DECODE_ONLY to 1008, VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP=4 | **CORRECTED** (offload size) | [FH logs/448173/config.yaml:94,99,153,155,175-177,188,190] all transport/decode items confirmed (`FI_EFA_USE_DEVICE_RDMA=1` also set). The offload size is **per DP rank = per GPU, not per worker**: prefill log prints `SimpleCPUOffloadConnector: role=WORKER, per_rank=70.00 GB, world_size=1` and `Allocating 593133 offload blocks (70.00 GB)` for both `EngineCore_DP0` and `EngineCore_DP1`; vLLM main divides `cpu_bytes_to_use` by `parallel_config.world_size` (= TPxPP, excludes DP) (`V/.../simple_cpu_offload_connector.py:104-107`, `V/vllm/config/parallel.py:535`). So 140 GiB per DEP2 worker, 700 GiB over the 5 prefill workers; GB300 445572 is likewise 80 GiB **per rank** (`per_rank=80.00 GB`). Plus Engram tables 94.42 GiB pinned host memory per node "shared across DP replicas via /dev/shm" (prefill log). The s5/T5 text "70-80 GiB per 2-GPU worker" and lever 3 "~35-40 GiB host DRAM per prefill GPU" should read **70-80 GiB per prefill GPU**. |
| V8 | Per-GPU peaks B300 vs MI355X: FP4 13.5 vs 10.07 (1.34x), FP8 ~4.5 vs 5.03 (0.89x), HBM ~1.0x, scale-up 1.8 vs ~1.07 TB/s (1.68x); GB300 15 PF, NVL72 | CONFIRMED as [2H] | NVIDIA DGX B300 page: "FP4 144 PFLOPS / 108 PFLOPS* (dense); FP8 72 PFLOPS** (sparse; dense = 1/2)" -> 13.5 / 4.5 PF per GPU; NVIDIA docs disagree on HBM (2.1 vs 2.3 TB per node) and some third parties use 15 PF/GPU for B300. AMD: MXFP4 10.1 PF (10,066 TF), FP8 5.0 PF dense, 288 GB, 8 TB/s, 7 IF links x 153 GB/s, platform P2P 1,136 GB/s aggregate bidirectional. Unit caveat [INF]: 1.8 TB/s NVLink and 7 x 153 GB/s are both bidirectional figures, so 1.68x is like-for-like. Still second-hand; not measured. |
| V9 | API hits B300 PD 0.466 + 0.510 = 0.976 vs theo 0.969; MI355X 0.883 + 0.057 = 0.940 vs 0.964; engine counters ~3.8-3.9% computed on both; contradiction left open | CONFIRMED, with one fact added | [FH re-fetch] 448173 GPU 0.46571 + CPU 0.50960 = 0.9753, theo 0.96929; 448242 0.88290 + 0.05656 = 0.9395, theo 0.96353. Engine: B300 prefill w0 window mean 112.1k (median 88.2k) tok/s x 5 workers = 0.561M vs API input 14.80M -> 3.8% (assumes w1-w4 = w0; only w0 downloaded); MI355X mean 34.0k (median 32.1k) vs 0.848M -> 4.0% (3.8%). Added [FH code, vLLM main 7a62ca0666]: the Rust frontend's "Avg prompt tput" counts **locally computed tokens only** (`rust/src/llm/src/log_stats.rs:132-135,369`, same as Python `loggers.py:165-166`); the pinned images were not checked. If the images match main, both engines compute ~4% of input while the API reports 6.0% vs 2.4% uncached; still open. Windows (middle 60%) differ from the API's whole-run averages. |
| V10 | AgentX replays end-to-start delays; out-of-flight 4.2 s at MI355X c384 vs 10-17 s elsewhere; unexplained | CONFIRMED, citation CORRECTED | Arithmetic [INF on FH]: 51.4-47.2 = 4.2; c128 13.0; c256 11.0; NVIDIA rows 10.3-16.9. The run used `agentx-harness==1.0.14` (`logs/448173/benchmark.out`), which vendors its own aiperf build (`_build_info.py` COMMIT_SHA 18015a2d). The cited lines `agentic_replay.py:1550-1580` (`_dispatch_next_turn`, "honoring delay_ms", `scheduler.schedule_later(delay_s, ...)`) are in that vendored copy (`source/pypi_agentx/x/aiperf/...`); in the standalone `aiperf-0.13.0` wheel the same function is at line ~1371. `weka_parallel_convert.py:428-450` (`_end_to_start_delay_ms`, floor at 0) and `inferencex_agentx_mvp.py:36-38,57` (system idle cap 10 s, `forbid_inter_turn_delay_cap`, `require_cache_bust=FIRST_TURN_PREFIX`, trace idle cap 300 s) confirmed in the vendored copy. Mechanism of the c384 anomaly: not established. |
| V11 | Cache-bust marker per lane at token 0; c3584 / c9888 reuse 393 traces 9.1x / 25.2x; no cross-lane prefix sharing | CONFIRMED, wording CORRECTED | [FH] `enums.py:248-262` ("Prefix variants diverge at token 0"); aiperf.log lines "Sampled 3584 trajectory lanes from 393 distinct traces (avg 9.1 lanes per trace)" and "9888 ... (avg 25.2 ...)". Precise unit: the marker is per **session instance**, digest of (benchmark_id, recycle_pass, trajectory_index, trace_id) (`timing/strategies/cache_bust.py:3-13`, `timing/conversation_source.py:192-210`), and is shared by all members of one trajectory tree (main session + subagents share the base trace id, `cache_bust.py` `base_trace_id`). So: no prefix sharing across lanes or across recycles of a lane; sharing within a lane's own tree is allowed. |
| V12 | ROCm gaps: n_redundant_experts=0; symm_mem EPLB raises off CUDA; DeepSelect SM100-only; NIXL / SimpleCPUOffload do not override `only_restores_canonical_kv` -> AITER sparse-MLA fast path off | CONFIRMED | [FH vLLM 7a62ca0666] `amd/model.py:167-173` ("The current ROCm runner has no redundant EPLB copies", `n_redundant_experts = 0`); `eplb_communicator.py:761-762` (`raise ValueError("symm_mem EPLB requires CUDA")`); `config/kernel.py:391` ("deep_select": "SM100a/SM103a only"); `only_restores_canonical_kv` defined only in `base.py:237` (returns False, :244), `umbp/connector.py:110`, `multi_connector.py:274-280` (True only if all children are); consumed by `kv_connector/utils.py:103-114` -> `amd/rocm.py:383-385` (`_use_aiter_sparse_mla = _aiter_sparse_mla_enabled(kv_connector_may_load_foreign_kv(...))`) and `amd/rocm.py:597`. Nuance: "SM100-only" = SM100a/SM103a (B300 is SM103). |

Net corrections to the body of this file (not edited above; apply when it is next revised):
1. s3.1 row 447890 and its bullet: B300 colocated per-request decode rate is ~46-52 tok/s (not ~32), and the colocated -> PD
   per-request gain on B300 is ~1.9x (not ~2.6x). Do not use medians of the 1 s B300 colocated log lines as rates.
2. s5 prefill workers, T5, lever 3: SimpleCPUOffload is 70 GiB (B300) / 80 GiB (GB300) **per prefill GPU** (per DP rank),
   i.e. 140-160 GiB per DEP2 worker; plus ~94 GiB/node pinned Engram tables.
3. s0 item 6 / s3.2 could add: the MI355X bar runs gpu-memory-utilization 0.9; vLLM's own log says 72.57 GiB KV/GPU at full use
   (61.02 GiB used).
4. O3 citation: `agentic_replay.py:1550-1580` refers to the aiperf vendored in `agentx-harness 1.0.14` (the build that ran), not
   the standalone `aiperf-0.13.0` wheel.
5. O6: marker is per session instance (trace x recycle pass x lane), shared within a trajectory tree.
