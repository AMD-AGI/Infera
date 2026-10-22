# PD r4 decode vs mixed DCP4 c40 vs 22k baseline decode: config diff and step distributions (2026-10-22, logs only)

Sources (copies in `/var/tmp/yihou/analyst_ws/pddiff/`):
- PD r4 decode
  - `server_args` from `pd/runs/pdBeperf-r4/server-logs/decode-0.a1.log`, 19:54:17Z line
  - env/args from `pd/harness/dryrun.pdBeperf-r4-054.yihou.txt`, decode-0 line 27
  - decode lines (TP0) from the same log, A2 window 21:45:19Z + 3640 s
- Mixed full1-dcp4-c40
  - `server_args` + env header from `mixrun/logs/mix-dcp4-sim.v3-045.20261021T142655Z.log`
  - decode/prefill lines (TP0) in the profiling window 15:18:05Z + 3624 s
- 22k baseline e2g: `results/e2g/dryrun.e2g-102.yihou.txt` in the 22k packup

## 0. Headline (measured)

- **No decode server-arg difference between PD r4 and mixed explains a large step-time gap.**
  - At matched running count, PD decode steps are within +1.4..+5 ms of the mixed server's decode-only steps (§2).
  - PD TPOT (19.7 ms) is better than mixed (36.9 ms), because the PD decode never interleaves prefill.
- **The per-chip gap is mostly the denominator and the upstream path, not the decode step:**
  - Mixed: total 26,277 × 4 = 105.1K tok/s on 4 GPUs.
  - PD r4 A2: 18,321 × 8 = 146.6K tok/s on 8 GPUs (1.39× mixed in total, with 2× the GPUs).
  - The 4 PD prefill GPUs computed only 17.9M new tokens in the A2 hour (4.9K tok/s for all four); 510.8M tokens were prefix-cached (decode_stats.r4.txt `pf_new`/`pf_cached`).
  - PD decode running stays at about 18-19 whether the client runs C40 (18.75) or C52 (18.31, `decode_stats.r4.txt` r4-C52). At C52, transfers are in flight in 89.9 % of lines vs 71.3 % at C40.
  - So extra concurrency does not reach the decode batch. It queues in prefill/transfer/TTFT (PD TTFT mean 3.93 s vs mixed 1.30 s).

## 1. Config differences, ranked by likely impact on decode step time

| # | Item | PD r4 decode | Mixed DCP4 c40 | 22k e2g decode | Evidence on impact |
|---|---|---|---|---|---|
| 1 | `SGLANG_OPT_USE_TOPK_V2` (DSA indexer top-k kernel) | `false` (dryrun line 27); for GLM DSA on HIP an unset value also defaults to false (`arg_groups/model_hook.py:434-437`) | **`true`** (log env header) | `false` | Different kernels (`dsa_topk_backend.py:55-56` `should_use_topk_v2`): the legacy top-k path vs the v2 `topk_transform_paged_v2`. The r2 PD graph trace has `coop_topk` at 9.3 ms per verify at Σctx 1.6-1.8× T9. Mixed T9 eager ref top-k is 1.63 ms/step (`topk_main_kernel` 1.23). Even scaled by 1.8×, that leaves a gap of several ms/step. Eager vs graph and the scaling are not like-for-like; **unmeasured A/B**. In the 22k DPA4 decode, rows per DP are ≤ 48, so the fused gfx950 indexer served most steps and this kernel mattered less there (inference). |
| 2 | ROCm debug agent (`HSA_TOOLS_LIB=…librocm-debug-agent.so.2`, `HSA_ENABLE_DEBUG=1`, `ROCM_DEBUG_AGENT_OPTIONS`) | **on**, prefill and decode (dryrun lines 19, 27; pdrun `DEBUG_AGENT=1`, ledger P005) | off | **off** (e2g `audit.txt:36`, `census.txt:34`) | The 22k kit measured e2i R2 with the decode agent on at 21,921 vs 22,114 off (second-hand, `working_process.md` R052/R053), about -1 %. Prefill with the agent on is unmeasured. |
| 3 | Decode radix cache | `disable_radix_cache: True`, `disaggregation_decode_enable_radix_cache: False` | radix on (mixed) | same as PD | Not a step-time item. Every turn re-sends the whole prefix KV from prefill to decode (`decode_prefix_len` 0; the decode radix path is `disaggregation/decode.py:1477-1490`). This drives TTFT/transfer load: transfers in flight mean 4.8 requests, 71 % of decode lines. |
| 4 | MoE placement | `--ep-size 4` (EP4, `moe_a2a_backend none`) | EP4 | **`--ep-size 1`** (TP MoE) | PD = mixed; differs from 22k only. Unmeasured here. |
| 5 | Attention layout | TP4 + DCP4 (attn_tp 4, no DPA) | same | TP4 + DPA4 (attn_tp 1) | Structural; this is the DCP design. Custom AR: PD r4 on, mixed on, e2g **off** (`--disable-custom-all-reduce`). |
| 6 | `max_running_requests` | 64 | 80 | 64 (16 per DP) | Not binding: PD running max 47 in A2. |
| 7 | `index_share_for_mtp_iteration` | false | true (checkpoint default) | e2g c44c dryrun shows true on both legs | Measured no step effect at 34K (benchrun T11, mixrun R21/22: ≤ 0.6 ms). |
| 8 | Env only on PD (and 22k) | `AITER_USE_FLYDSL_MOE_SORTING=1`, `HIP_FORCE_DEV_KERNARG=1`, `HSA_NO_SCRATCH_RECLAIM=1`, `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0`, `SGLANG_DP_USE_GATHERV=0` | absent | same as PD except GATHERV=1, NCCL_DEBUG=INFO | Unmeasured. All were inherited from the 22k recipe. |
| 9 | Env only on mixed | — | `HSA_ENABLE_SDMA=0` (v3 node) | — | Copy engines only. |
| 10 | `mem_fraction_static` | 0.85 | 0.85 | 0.95 | Capacity only (usage 0.455 in A2). |
| 11 | `enable_fused_qk_norm_rope` | True (flag) | None (not passed) | True | Not checked whether None resolves to the fused kernel. Both logs print "aiter fused_qk_norm_rope kernel available". |

Identical in PD r4 and mixed (server_args diff of all keys):
- cuda graph decode `max_bs` 64 with the same bs list (… 16, 18, 20, 22, 24, 26, 28, 30, 32, 40, 44 …);
- EAGLE 5/1/6, page_size 64, triton/triton DSA, `dsa_topk_backend sgl-kernel`, `moe_runner_backend auto`, `moe_a2a_backend none`;
- `dcp_comm_backend a2a`, `dcp_replicate_q_proj True` (inactive: fp8 kv_b), `enable_aiter_allreduce_fusion True`, custom AR on, `disaggregation_decode_polling_interval 1`, overlap schedule on.

The only server-arg differences are: radix/hicache, disaggregation keys, `json_model_override_args`, `max_running_requests`, `enable_fused_qk_norm_rope`, `watchdog_timeout`.

## 2. Step-time distribution vs running count (TP0 decode lines; step = 1000 × running × accept / gen_tput, as benchrun's stepbins)

PD r4 A2 (1,486 lines; running mean 18.75; queue 0, prealloc 0 in every line; #transfer-req mean 4.8, > 0 in 71 % of lines):

| running | n | median ms | p10 | p90 |
|---|---|---|---|---|
| 8-11 | 145 | 41.5 | 34.4 | 50.6 |
| 12-15 | 197 | 51.7 | 44.3 | 59.9 |
| 16-19 | 262 | 61.4 | 53.9 | 66.0 |
| 20-23 | 301 | 68.5 | 63.2 | 74.2 |
| 24-27 | 230 | 74.1 | 68.8 | 80.4 |
| 28-31 | 126 | 78.9 | 73.9 | 83.4 |
| 32-35 | 53 | 85.7 | 81.2 | 92.4 |

Transfers in flight vs none, at matched running:
- 16-19: 60.8 (n 190) vs 62.1 (n 72)
- 20-23: 68.2 (n 206) vs 68.9 (n 95)
- So concurrent KV transfers do not slow the decode step (measured).

Mixed full1-dcp4-c40, same method:

| running | all lines n / median | "clean" lines (no Prefill line since previous decode line) n / median | PD median |
|---|---|---|---|
| 16-19 | 69 / 70.1 | 10 / **58.6** | 61.4 |
| 20-23 | 88 / 79.1 | 8 / **63.2** | 68.5 |
| 24-27 | 168 / 88.0 | 12 / **69.8** | 74.1 |
| 28-31 | 174 / 96.3 | 6 / **77.5** | 78.9 |

- PD decode-only steps are +1.4..+5.3 ms over mixed decode-only steps at the same running. The mixed clean bins are small (n 6-12), and the context distributions of the two runs at those instants are not matched.
- Mixed "all lines" are much slower, because prefill interleaves.
- For reference, mixed decodebench (fixed context, TOPK_V2 true, AR on): 150K N=16/24 = 49/61 ms; 131K N=16/24 = 48/60 ms.

## 3. What would decide it (cheapest first; no runs done)

1. `SGLANG_OPT_USE_TOPK_V2` true vs false: a mixed decodebench A/B at 150K/200K on a v3 node, then the PD decode with true.
   - Watch correctness: the 22k kit kept false; `dsa.topk.indexer.bug.analysis.md` discusses the v2 kernel's padded-row clamps (#30378) on DP attention. Our DCP4 decode has no DP attention.
2. Debug agent off (`DEBUG_AGENT=0`) on both legs: a config-only change; expected about 1 % on decode by the 22k kit's measurement.
3. Throughput beyond decode: prefill GPUs are mostly idle (4.9K new tok/s across 4) and decode running does not grow with concurrency. Fewer prefill GPUs per decode GPU would raise the per-chip number. A decode radix cache would cut per-turn full-context transfers. Both are structural; not measured.
