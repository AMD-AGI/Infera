# Decode-side DCP knobs worth A/B for PD throughput (ROCm, GLM-5.2, TP4/EP4 + DCP4 + EAGLE 5/1/6)

Code read on `ae529c96bf`. "Expected direction" is given only where the code determines it; everything
else is marked **open** together with the measurement that decides it. Nothing here has been measured.

| # | Knob | Default on this setup (code) | What the code does | Expected direction | Measure |
|---|---|---|---|---|---|
| K1 | `--dcp-comm-backend a2a` vs `ag_rs` | **a2a**: unset resolves to `a2a` on HIP (`arg_groups/overrides.py:1364-1387`); `fi_a2a` is CUDA-only | a2a packs output and fp32 LSE into **one** `all_to_all_single` per layer, in the output dtype (bf16), then combines locally (`layers/dcp/comm.py:474-537`). ag_rs does **two** collectives per layer: LSE all-gather, then `reduce_scatter` of a **fp32** `[H,B,D]` tensor (`comm.py:85-123`). | a2a ≤ ag_rs in bytes and in collective count, so keep a2a. Latency on XGMI/RCCL at small B is unmeasured. | decode ITL at fixed batch (C16/C40), a2a vs ag_rs, or a kineto trace of one decode step |
| K2 | `--dcp-replicate-q-proj` | resolves to True under a2a (`model_overrides/deepseek_v2.py:33-34`) | On gfx950 with aiter, GLM's bf16 `kv_b_proj` is quantized to fp8 e4m3fn (`deepseek_common/deepseek_weight_loader.py:724-731`). `_prepare_replicated_q_proj` skips any layer whose `w_kc` is not bf16/fp16 (`model_executor/model_runner.py:1139-1149`). | **Likely a no-op here.** Every layer should keep the per-layer Q all-gather. If so, A/B is pointless. | **Confirmed by pdBe-r1 and mixrun logs**: the target runner logs `prepared ... for 0 MLA layers`. The draft (NextN) runner logs `for 1 MLA layers`, because its arch is rewritten to `GlmMoeDsaForCausalLMNextN` (`configs/model_config.py:945`), which the fp8 kv_b gate (`architectures[0] == "GlmMoeDsaForCausalLM"`) skips, so its w_kc stays bf16. Toggling the flag only affects the single draft layer. |
| K3 | `SGLANG_DSA_DCP_OWNED_PREFILL_RATIO` | 3.0 (`environ.py:1867`) | Only consulted by `_plan_dcp_gathered_prefill`, which requires `is_extend_without_speculative()` (`dsa_backend.py:1420-1436`). A PD decode's first batch is PREBUILT with no forward (`decode_schedule_batch_mixin.py:23-35`); draft-extend and verify are excluded. | **No effect on the PD decode.** It matters only for mixed (non-PD) serving or a DCP prefill. Do not A/B it in PD. | (optional) count of `dsa_gathered_prefill` plans = 0 on decode |
| K4 | `--cuda-graph-max-bs-decode` | mixrun/known-good use 64 | Decode graphs are keyed by **request** batch size (`runner/decode_cuda_graph_runner.py:707-745`). A larger batch runs eager. In eager, recipe `dsa_dp_sync` builds a full `context_len`-wide page table when `seq_lens_cpu` is absent (`dsa_backend.py:~1104-1115`). The DCP decode is TP4 without DP-attention, so one batch holds **all** running requests (the 22k decode was DPA4, about 1/4 per rank). | Set it ≥ the expected running batch (≥ `--max-running-requests`). Below that, steps fall to eager. A larger value costs capture time and graph memory only. | decode log: share of non-graph steps; capture time and memory |
| K5 | `--max-running-requests` vs capacity | 64 (mixrun) | Mixed DCP4 measured `max_total_num_tokens` 9,267,456 logical (mixrun R1). In PD the decode radix cache is **off** by default (`pd_disaggregation_hook.py:121-127`), so decode KV holds only the running requests' contexts, with no prefix cache. | Capacity limits concurrency only when (running reqs × average context) approaches 9.27M, e.g. 64 × 145K. The mixed-mode DCP gain came from prefix-cache hits (#44322: 94-97% chip hit). That mechanism is absent on a PD decode; prefix hits live on the prefill and its HiCache. **Open:** whether PD+DCP4 beats the 22k non-DCP PD at all. | decode `token_usage` and `#running-req` at C40/C48; queueing (`#queue-req`) on decode |
| K6 | `--ep-size 4` vs 1 on decode | mixrun/#44322: EP4; 22k decode: EP1 with DPA4 | Not DCP-specific. | **open** | ITL at fixed batch |
| K7 | `--dcp-size 2` vs 4 (TP4) | 4 | DCP2: half the decode capacity gain and smaller a2a groups; each rank still computes all 64 heads ÷ 2. | **open** | ITL plus capacity |
| K8 | `--mem-fraction-static` (decode) | 0.85 (mixrun) vs 0.95 (22k decode) | Graph memory under DCP: the a2a buffers are `[W, B, H/W, D+2]` per call. | Higher means more KV; it is capped by graph capture OOM. | capture success, `max_total_num_tokens` |
| K9 | `SGLANG_OPT_USE_TOPK_V2` | mixrun true, 22k false | Not DCP-specific. | **open** | ITL |
| K10 | `--disable-custom-all-reduce` (decode) | on in the 22k recipe (garbled-MTP fix) | Correctness fix in the 22k image; whether it is still needed on v0.5.22 + branch is unverified. | Keep for correctness; A/B only after T4 passes. | GSM8K / garbled-output check, then ITL |

## Ordering suggestion

1. Correctness first (T4).
2. K2 (log read, free) and K4 (match graph bs to the running batch).
3. Baseline AgentX at the best concurrency.
4. K1, K6, K9, each one run at fixed concurrency.
5. K5, K7 only if decode-side queueing or KV limits are observed. The `llm-pd-bottleneck-finder` skill can
   localize prefill vs transfer vs decode first.

## Update after pdBeperf-r2 (A1): ranking at ~27 running in one TP4/DCP4 decode batch (code read, unmeasured)

**Observed** (pdrun, first-hand):
- 16,670 tok/s/GPU vs the 22k kit's A1 19,640;
- TPOT mean 23.7 ms vs 14.3 ms;
- decode running 23-32 in one TP4 batch, vs about 4-9 per DP rank in the 22k run.

| Rank | Knob | Code facts | Expected direction | Gate |
|---|---|---|---|---|
| 1 | **Decode custom all-reduce ON** (drop `--disable-custom-all-reduce`) | Attention is TP4 here, so every layer has o_proj (row-parallel, `deepseek_v2.py:2152`) plus MoE output all-reduces over 4 ranks, for all T = 6B rows. That is about 2 x 78 all-reduces of a `[T, 6144]` bf16 per verify step, plus the draft steps. The 22k decode had attn_tp 1. The other session's new baseline (`~/dev/git/infera.glm52.pd/CLAUDE.md`, 10-21) runs decode custom AR **ON** with `index_share_for_mtp_iteration=true` and idlefix #45483. | faster all-reduces. Size unmeasured. | **Correctness first.** The garble root cause (pack-up 2026-09-18) was custom AR + TP4 + IndexShare off. Check math, passkey and real-draft accept length on the DCP decode with custom AR on. |
| 2 | `index_share_for_mtp_iteration=true` (new baseline) | The draft reuses the target top-k, so the NextN indexer does not run on draft steps. Under ROCm DCP the merged/global top-k is the same on all ranks (replicated K). | small saving per cycle (about 5 indexer layer-steps) | Watch for the IndexShare MAF; the baseline carries idlefix #45483, which our branch does not have. |
| 3 | **DP2 + DCP2** (attn_tp 2) | It halves the per-GPU replicated indexer and `fused_qkv_a` vs DCP4, and splits the batch into 2 replicas (`perf_layouts.md` section 2). | less replicated work; adds DP imbalance and halves capacity per replica | Untested combination (no guard, no run). Smoke first. |
| 4 | Variant B (`SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1`) | about 2.8 ms/verify-step overhead at about 160 rows (benchrun) | wins if the replicated indexer is ≥ 3.7 ms per verify step | pdBeperf trace (`tools/trace_step_breakdown.yihou.py`) |
| 5 | `SGLANG_OPT_USE_TOPK_V2` true/false | Switches the decode paged fused top-k to the v2 JIT kernel (`dsa_topk_backend.py:55-56, 128-140`). The 22k decode used false, mixrun/pdBe true. | unknown. Top-k was 4.5% of the 22k busy time, at about 4x fewer rows per GPU. | A/B |
| 6 | `--ep-size 4` vs 1 (decode MoE) | not DCP-specific | unknown | A/B |
| 7 | decode tokenizer/detokenizer workers 4/4 (new baseline) | host-side output processing for a 27-32 request batch in one scheduler | unknown | cheap A/B |
| 8 | `--cuda-graph-max-bs-decode` / bs list | Spec capture list: 1-8, 10-32 step 2, then 40-64 step 4 (`cuda_graph_hook.py:516-523`). At 23-32 running, padding is ≤ 1 request; 33-39 pads to 40. | small unless running exceeds 32 | decode parser `graph%` and running histogram |
| - | `--dcp-replicate-q-proj` | no-op on target layers (K2) | none | - |

**Tooling:** `tools/trace_step_breakdown.yihou.py`
- It attributes each kernel to its `step[VERIFY|DRAFT|DECODE|EXTEND ...]` span via launch correlation.
  CUDA-graph kernels share the graph launch's correlation.
- It reports ms/step per category (indexer_logits, indexer_topk, attention, dcp_merge, comm, moe,
  moe_route, gemm, kv_write, norm/rope/act, copy, other), plus the uncategorized and top kernels.
- Validated on a local sglang ROCm trace: the CPU-span attribution matches the GPU-annotation
  cross-check (23.57 vs 23.58 ms/step).
- Kernel-name regexes for GLM/DSA are untested. Check the "uncategorized" list on the first pdBeperf
  trace and refine with `--cat`.

## RCCL at TP4/DCP4 decode (25% of the verify step in the r2 trace): levers from code

The r2 verify (bs 24) had 314 `ncclDevKernel_Generic` per verify, about 4 per layer:
- the DCP Q all-gather (`forward_mla_rocm.py:772-777`);
- the DCP LSE a2a (`:1018-1031`);
- the o_proj all-reduce;
- the MoE output all-reduce.

1. **Custom all-reduce ON.** This affects two things:
   - (a) The o_proj and MoE all-reduces use aiter custom AR instead of RCCL.
   - (b) It is a precondition for the aiter **AR + RMSNorm fusion**.
     - `fused_allreduce_rmsnorm` returns None when `ca_comm` is None or disabled
       (`distributed/parallel_state.py:819-829`).
     - `--disable-custom-all-reduce` therefore also disables the `--enable-aiter-allreduce-fusion` path.
     - That fusion applies only when DP attention is off (`layers/layer_boundary/residual/add_norm.py:170-182`),
       so the 22k DPA decode never had it and our TP4 decode can.
   - Expected: fewer and faster collectives plus one fewer norm kernel per boundary. Size unmeasured.
   - **Gate:** the 09-18 garble (custom AR + TP4 + IndexShare off). Run the correctness checks first.
2. **Remove the per-layer Q all-gather via replicated q-proj** (`--dcp-replicate-q-proj`, default True
   under a2a).
   - It is skipped for the 78 target layers because GLM's `kv_b_proj` is converted to fp8 on gfx950
     (`deepseek_weight_loader.py:724-731`), and `_prepare_replicated_q_proj` requires bf16 `w_kc`
     (`model_runner.py:1139-1149`).
   - Options:
     - (a) Keep `kv_b` bf16 under DCP (skip the gfx950 fp8 conversion when `dcp_size > 1`): q-rep becomes
       active, the all-gather disappears, but the absorb bmm runs bf16. Trade-off unmeasured.
     - (b) Teach q-rep fp8 `w_kc`: all-gather `w_kc` plus each rank's per-tensor `w_scale` and use
       per-head-group scales in the absorb bmm. Code change in the bmm; moderate.
   - Each removes 78 all-gathers of about `[T, 16 heads, 576]` bf16 per verify step (about 2.7 MB per rank
     at T = 144).
3. **Keep `--dcp-comm-backend a2a`.** It already merges output and LSE in one collective; ag_rs would add
   one more per layer, plus a fp32 reduce-scatter.
4. **DP2 + DCP2.** Attention collectives (Q all-gather, a2a, o_proj AR) run over 2 ranks on half the rows,
   but there are 2 replicas and MoE still spans 4. Untested combination.
5. **Fewer collectives per layer** (no existing flag found): fusing the o_proj all-reduce with the DCP a2a
   (both run over the same ranks after attention) would need new code; not proposed.
