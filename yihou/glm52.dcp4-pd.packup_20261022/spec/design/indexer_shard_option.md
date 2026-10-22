# Option: stop replicating the DSA decode indexer under DCP (lever if the trace says the indexer dominates)

Status: **prepared, not implemented.** This is a code read of sglang #42618 (head `a335b6eb`, diff at
`/var/tmp/yihou/research_glm52_dcp/pr42618.diff`) against our branch `906b0915a9`. Nothing below is
measured, except the 22k kineto numbers quoted with their source.

## 0. First-hand evidence on the indexer's share (22k decode, DP4/DCP1)

Kit ledger R054 (`glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/analysis/working_process.md:1002`):
kineto of e2j R1, 8 steps on all 4 DP ranks, DP0 batch 8-9, about 64 ms/step, GPU busy union 72.7%.

"top kernels nccl 17 %, MoE fp4 gemm1 15.6 %, _sparse_mla_fused 13.6 %, **indexer paged-MQA 9 %**, MoE
gemm2 7.7 %, **topk 4.5 %**"

- So in the 22k layout the indexer (logits plus top-k) is **about 13.5% of the top-kernel time**. Each
  rank there scores only its own DP batch.
- The kit's cost model (R042) states the indexer is proportional to `6 x sum(ctx)` per rank.
- **No DCP4 trace exists.** Under DP1/DCP4 every GPU scores the whole batch, about 4x the 22k per-rank
  indexer work at equal total batch (`perf_layouts.md` section 2). Its share of the DCP4 step is
  **unknown**; the pdBeperf kineto trace decides it.

## 1. What #42618 changes (decode path; prefill noted only where it touches us)

| Area | #42618 | Our branch (#43735 + #44322) |
|---|---|---|
| Index-K layout | **Striped** like the latent: written only on the owner rank (`slot % W == rank`, row `slot // W`) via `dcp_localize_write_loc` in all three store sites (aiter fused prepare, fused k store, `_store_index_k_cache`). The pool is per-rank size. | **Replicated**: every rank holds the whole sequence, addressed by widened slot (`memory_pool.py:5344-5352`). It costs about 17% of the DCP4 KV budget (doc). |
| Decode/verify/draft-extend indexer | Each rank runs `aiter_paged_mqa_logits` over **its shard** (local lens and local block table, `dcp_local_index_block_table`). Local top-k, then `dcp_exchange_topk`: pack (fp32 score, int32 global pos) as `[rows, topk]`, **all-gather over DCP**, merge to the global top-k (Triton `dcp_topk_pack/merge`, `kernels/ops/attention/dcp_kernels.py` +202). Exact: a global top-k token is in its owner's local top-k. | Every rank scores the whole context for all rows (paged path, `dsa_indexer.py:2231-2237`). No collective. |
| Attention | Owned top-k slots, split-K triton with LSE, LSE merge (custom all-gather ≤ 1 MiB, else RCCL a2a) | Same idea (#44322 `_forward_triton_dcp` + a2a merge) |
| MTP IndexShare | **Off** under DCP (each rank holds only part of the top-k) | Works (replicated). The 22k recipe already runs `index_share_for_mtp_iteration: false`, so nothing is lost for us. |
| Fused top-k / tilelang fused rope+store | Disabled under DCP | Fused top-k page-table path still exists on decode (`dsa_backend.py` `_get_fused_topk_page_table`) |
| Prefill (not run on a PD decode) | Index-K all-gather or owned-prefill top-k exchange | Row split on the replicated K (#44322) |
| Per-forward state | Module global `_STEP` (`layers/dcp/dsa.py`), flagged by the reviewer (CHANGES_REQUESTED 2026-10-13) | n/a |
| Measured (PR body, MI355X GLM-5.2-MXFP4, mixed, **with HiCache**) | c40 29,786 tok/s/GPU, c48 33,835; KV pool 10.89M | #44322: c40 27,087, c48 13,252 (no HiCache); #45205 (+HiCache) c40 29,339, c48 31,935. Different runs, configs and HiCache, so this is **not** a clean A/B of the indexer layout. |

## 2. Two ways to bring the lever onto our branch

### Variant A: full #42618 index-K sharding
Needs:
- the striped write sites;
- per-rank pool sizing;
- the decode indexer exchange;
- the kernels;
- **plus our PD relayout for index K, which has to change.**

Today `StateType.DSA` copies 64-slot pages one-to-one (proved by T3). A striped decode page holds 64
**non-contiguous** positions (`256L + r + 4j`). The `IndexKeyCache` page layout is `[64 x 128 B keys | 64
x 4 B scales]` (`index_key_cache.py:39-46`), so a token is two non-adjacent pieces. That requires:
- a new index-K pack kernel that gathers owned tokens' key and scale bytes into the destination page
  layout;
- a token-granular plan for state, not the page path;
- a rewritten T3;
- a host-pool change for HiCache (#45205 assumes replicated index K; not on our branch anyway);
- either porting #42618's prefill index-K gather/owned paths, or gating the shard to `disaggregation_mode
  == decode`, since mixed serving would otherwise read a striped K in the replicated-K prefill split.

Interaction with the recipes: `dsa_indexer rows` trims padded q rows inside the aiter paged branch, and
#42618's DCP branch sits **before** that code. A port must apply the same trim, or padded rows get
scored and gathered (wasted work; correctness depends on `-1` handling of padded lens).
`dsa_page_table_rows` is unaffected because the top-k output stays per token.

Effort: about 3-4 days of code plus GPU validation. Risk: high. It spans write, pool, PD transfer and
kernels; the PR is unmerged with requested changes; there is no CI.

### Variant B (recommended if needed): split the scoring, keep the replicated index K
Changes:
- Each DCP rank scores only **its share of the context pages**: request page `j` belongs to rank
  `j % W`. Because the K is replicated, any split is legal.
- Then a local top-k, an all-gather of (score, global position), and a merge, using #42618's
  `dcp_exchange_topk` (it has a pure-torch fallback; the Triton pack/merge kernels are optional).

Why it is the smaller change:
- **No change** to index-K layout, writes, pool size, PD relayout (T3 stays valid), HiCache, or prefill.
- It touches only the HIP DCP decode/verify/draft-extend branch of `_get_topk_paged`, plus the fused
  top-k gate (fused top-k must be off on this path, because the merge returns positions).
- MTP IndexShare keeps working: the merged global top-k is identical on every rank.
- It keeps the 17% replicated index-K memory, so it gets no capacity gain. Capacity is not binding in PD
  at C40 (`perf_layouts.md`).

Correctness:
- **Merge exactness:** any global top-k token is in its owning rank's local top-k, so the set is exact up
  to tie order.
- **Local lens and positions:** must be derived per query row, including the verify rows' causal lengths
  (`seqlens_expanded`).
- **Global position mapping:** `pos = (page_list[c // 64] * W + r) * 64 + c % 64` relative to the
  request's page order. It needs a small kernel or a vectorized torch expression.
- **Graph capture:** capture-safe (all_gather and topk with static shapes), but rows × W × topk must be
  padded to the captured batch.

Interaction with the recipes: apply the `dsa_indexer rows` trim before the split. Nothing else changes.

Effort: about 1-1.5 days of code plus unit tests, then GPU validation (passkey 8K/34K/96K plus GSM8K vs
replicated, then pdBeperf A/B behind an env gate, e.g. `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1`).
Risk: medium. It is self-contained and env-gated, so it is easy to A/B and to revert.

## 3. Expected compute and communication per decode GPU per target-verify step (formulas; unmeasured)

Notation: `R = 6B` verify rows (the draft steps add `B` rows each), `C` = context length (22k mean about
257K, derived from e2gA1), `L_idx` = 21 full indexer layers, `W` = 4.

| | Replicated (now) | Variant A or B |
|---|---|---|
| index-K bytes read for logits | `R x C x 132 B x L_idx` | `R x C/W x 132 B x L_idx` |
| logits FLOPs (32 index heads x 128) | `R x C x 32 x 128 x 2 x L_idx` | `/W` |
| local top-k input | `R x C` per layer | `R x C/W` per layer |
| extra: merge top-k | - | `R x (W x 2048)` per layer |
| extra: all-gather | - | send `R x 2048 x 8 B` per layer, receive `(W-1)x` that, so `L_idx x R x 2048 x 8 x W` bytes moved per step |
| index-K memory | `+W x` replicated share (~17% of budget at DCP4, doc) | A: per-rank; B: unchanged |

Illustration (derived, not measured): B = 40 (R = 240) gives an all-gather receive of about
`21 x 240 x 2048 x 8 x 3 ≈ 248 MB` per step per GPU for the target verify, plus the draft steps. The
achieved all-gather bandwidth on XGMI is unmeasured, so its time is unknown. Measure it before
implementing: one `all_gather_into_tensor` of a `[240, 2048, 2]` int32 tensor over the DCP group,
timed in the pdBeperf container.

The trade-off is about `(W-1)/W` of the indexer logits and top-k work, against one all-gather plus
merge per indexer layer. Which wins depends on C and B, and on the indexer's share of the DCP4 step,
which is not measured yet.

## 4. Decision rule

1. Take a kineto trace of one pdBeperf decode step (DCP4, CUDA graph, C40). If indexer (paged MQA logits
   plus top-k) is ≥ about 25% of busy GPU time, the lever is worth building. That threshold is arbitrary,
   picked as about 2x the 22k share.
2. Microbench the all-gather above, on the same node.
3. Implement Variant B behind an env gate, run the T4 correctness checks, then A/B at C40/C48.
4. Consider Variant A only if B wins and the 17% index-K memory then matters, i.e. KV becomes binding at
   high concurrency.

## 5. Implementation status (Variant B, speculative, default OFF)

Branch commits on top of `906b0915a9`:
- `a78ff55a8b`: `layers/attention/dsa/dsa_dcp_split_indexer.py`. Pure functions:
  - `owned_page_table` / `owned_lens` (page `j` -> rank `j % W`);
  - `local_to_global_pos`, `pack_candidates`, `merge_candidates` (exact merge via `torch.topk` over
    `W x topk`);
  - `positions_to_slots` (the fused paged top-k contract: `real_page_table[row, p // P] * P + p % P`);
  - `split_decode_topk`.
  - Tests: `test/registered/unit/test_dsa_dcp_split_indexer.py`, 5 tests on CPU covering random scores
    with partial pages and short rows (0..129 tokens), ties, MTP verify rows (6 per request), the
    `owned_lens` partition, and slots. All pass.
- `a04c3f8e0c`: env `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER` (default False) plus the hook in
  `Indexer._get_topk_paged`.
  - The hook is inside the aiter branch, after the recipe's padded-row trim; it restores the `-1`
    padding rows.
  - Active only when all of these hold:
    - HIP with aiter preshuffle;
    - `metadata.dcp_size > 1`;
    - decode, target verify, or draft extend v2;
    - no init/local tokens;
    - 1-D lens;
    - page-table rows ≥ query rows.
  - It returns slots when `SGLANG_DSA_FUSE_TOPK` is on and the metadata is not force-unfused,
    otherwise positions. These are the same two spaces `topk_transform` returns.
- Microbench: `tools/dcp_allgather_bench.yihou.py` (torchrun, 4 ranks; eager and CUDA-graph all-gather
  of `[2, rows, 2048]` int32 plus merge top-k, rows 40..400; prints the per-step cost for 21 layers).

**Unverified on GPU** (CPU tests use a reference scorer and top-k):
- (a) `aiter_paged_mqa_logits` with a strided (every W-th page) block table, per-rank lens and a
  smaller `max_seq_len`;
- (b) `fast_topk_v2` on the padded local logits;
- (c) capture and replay of the path inside the decode CUDA graphs;
- (d) equality with the replicated top-k on real data.

GPU validation order:
1. The bench.
2. Server with the env on, eager: math, passkey 8K/34K/96K, GSM8K vs env off.
3. Then CUDA graph.
4. Then a pdBeperf A/B at C40. Ship only if the trace and the A/B say so.

## 6. Microbench result (benchrun, v3-060 MI355X, W=4, torch.distributed RCCL, first-hand by benchrun)

| rows | all-gather graph (us) | all-gather eager (us) | merge topk (us) | x21 layers (ms) |
|---|---|---|---|---|
| 40 | 31 | 36 | 84 | 2.4 |
| 80 | 44 | 50-54 | 57-63 | 2.1-2.3 |
| 120 | 56 | 61 | 59 | 2.4 |
| 160 | 70 | 75 | 62 | 2.8 |
| 240 | 94 | 98 | 69 | 3.4 |
| 320 | 120 | 124 | 84 | 4.3 |
| 400 | 145 | 150 | 95 | 5.0 |

Logs: `/shared_nfs/yihou_glm52/dcp_pd_20261021/bench/logs/ag.*.log`. Two runs agree within about 2%.

Derived (arithmetic on the bench, not an end-to-end measurement):
- Overhead added by Variant B per decode cycle at B=40:
  - target verify (rows = 6B = 240, 21 indexer layers): ≈ 3.4 ms;
  - 5 draft steps (rows = B = 40, 1 NextN indexer layer each): ≈ 5 x 0.115 ≈ 0.6 ms;
  - total ≈ **4 ms per cycle**.
- **Break-even:** Variant B removes about 3/4 of the replicated indexer (paged-MQA logits plus top-k). It
  pays only if that replicated indexer time is above about 5.3 ms per cycle on the DCP4 decode (4 ms /
  0.75). Read this straight off the pdBeperf kineto trace.
- The merge `torch.topk` over `W x 2048` columns costs about as much as the all-gather for rows ≤ 160.
  If B is close to paying, the next step is a fused merge kernel (#42618 has `dcp_topk_merge` in
  `kernels/ops/attention/dcp_kernels.py`). Its gain is unmeasured.

## 7. Break-even at the mixrun DCP4 C40 operating point (derived; nothing end-to-end measured)

**Inputs**
- mixrun mixed DCP4 C40: running mean 27, so about 162 verify rows; TPOT mean 36.9 ms (mixrun,
  first-hand).
- benchrun: all-gather plus merge over 21 layers ≈ 2.8 ms at 160 rows.
- Per draft step (1 NextN indexer layer, about 27 rows): ≈ 0.12 ms, so 5 steps ≈ 0.6 ms.

**Overhead added by Variant B:** about 2.8 ms per verify step, about 3.4 ms per decode cycle.

**Break-even:**
- Variant B removes about 3/4 of the replicated indexer.
- It wins if the replicated indexer (paged-MQA logits plus top-k) on the DCP4 decode is above
  **2.8 / 0.75 ≈ 3.7 ms per verify step**, or about 4.5 ms per cycle including the draft indexer.

**Rough expectation from the 22k trace (R054)**
Assumptions (unverified):
- the trace percentages are shares of GPU-busy time;
- indexer time scales linearly with rows at the same context mix.

Calculation:
- Indexer (9% + 4.5%) x 72.7% busy x 64 ms ≈ 6.3 ms per step at 8-9 requests per rank.
- Scaled to 27 requests on one DCP4 GPU: about 20 ms per step.
- That is well above 3.7 ms, so B would likely pay. **Low confidence:** this is two extrapolations on a
  different layout.

**What the pdBeperf kineto trace must show:** per verify step on one decode GPU, the summed time of the
indexer kernels (aiter paged-MQA logits plus the top-k kernel, over the 21 indexer layers) is
**≥ 3.7 ms**. Report it next to the step's total GPU time. If it is below about 3.7 ms, do not deploy B.

## 8. The gfx950 fused DSA indexer and Variant B (code read)

**What it is** (`layers/attention/dsa/dsa_indexer.py:440-500` gate, `:703-859` `_gfx950_fused_decode`;
`gfx950_fused_indexer.py:237-327` `run`; kernels in `kernels/jit/csrc/dsa_gfx950/`). Four launches:
1. `dual_gemv_kernel`: q_b, and k plus weights_proj, in one GEMV.
2. `qk_rope_hadamard_quant_kernel`: k_norm, rope, Hadamard, q quant, index-K store, head gate.
3. `logits_hist_m`: paged MQA logits plus histogram.
4. `k_scatter`: top-k 2048 plus page transform, emitting physical slots.

It is called in `forward_cuda` **before** the standard path (`dsa_indexer.py:2158-2172`). It returns
`None`, falling through to the standard path (where the Variant B hook lives), unless all of these hold:
- the mode is decode or target verify (**not** draft-extend-v2);
- not inside a piecewise/breakable prefill graph (decode CUDA graphs are fine);
- the metadata is not `force_unfused_topk`;
- no prefill CP;
- bf16 hidden states;
- no LoRA;
- pool `index_page_size == 64`;
- **1 ≤ rows ≤ MAX_ROWS = 48** (`gfx950_fused_indexer.py:33`, "bs 8 x num_draft_tokens 6");
- per-row seqlens with `rows == seqlens rows`;
- a contiguous int32 64-page table.

**Under DCP it is replicated like the standard path:**
- it has no DCP awareness;
- every rank scores the whole context of all its rows over the replicated index K;
- it writes index K at the widened `out_cache_loc`.
So at ≤ 48 rows it is the same replicated work, only fused.

**Consequence for Variant B (the hook is only in the standard path):**

| Step | Rows at ~27 running | Path taken (by code) | Variant B active? |
|---|---|---|---|
| target verify | 6 x 27 = 162 > 48 | fused declines ("row count is outside [1, 48]"), so standard path | **yes** |
| draft decode (NextN indexer) | 27 ≤ 48 | fused | no (replicated, 1 layer) |
| draft extend v2 | n/a | standard | yes |
| target verify at ≤ 8 running | ≤ 48 | fused | no |

So at the PD operating point (23-32 running), Variant B should already apply to the verify indexer even
with the fused indexer enabled. benchrun's observation (hook only in draft extend) is consistent with a
run at ≤ 8 running requests. **Unverified:** check the decode log for the one-time line
`gfx950 fused DSA indexer declined this call: row count is outside [1, 48]` and the running-request count
during the A/B.

**Options**
1. **Keep fused on; Variant B covers rows > 48 (the current code).**
   - No change.
   - At high concurrency the fused path does not serve verify anyway, so `--no-enable-dsa-fused-indexer`
     changes nothing there for verify.
   - It does change draft decode and small batches.
2. **Split inside the fused path** (for batches ≤ 8 or the draft indexer).
   - `run()` would need restructuring: keep nodes 1-2 (projection and store); run node 3
     (`logits_hist_m`) on this rank's page subset with per-rank lens.
   - Node 4 (`k_scatter`) emits slots without scores, so the merge needs a score-returning local top-k:
     standard `topk_func` plus `pack_candidates`, then the merge and `positions_to_slots`.
   - Kernel ABI support for a strided page table and per-rank lens is unverified.
   - Effort is about a day. Payoff is small: these rows are few (draft decode, 1 layer; small batches).
3. **`--no-enable-dsa-fused-indexer` plus split everywhere** (benchrun's A/B).
   - It loses the fused projection/quant/store and logits/top-k for decode, draft decode and small verify.
   - **The cost of disabling the fused path is unmeasured.** The A/B must compare four arms: fused on/off
     x split on/off, at the same concurrency, with the running count recorded.

**Recommendation (code-based):** measure at the PD operating point (≥ 20 running).
- If verify rows exceed 48, compare Variant B on vs off **with the fused indexer left on**. That isolates
  the split.
- Treat "fused off" as a separate variable.

**Trace script:** `tools/trace_step_breakdown.yihou.py` now separates
- `idx_fused_proj` / `idx_fused_logits` / `idx_fused_topk` (`dual_gemv_kernel`,
  `qk_rope_hadamard_quant_kernel`, `logits_hist_m`, `k_scatter`), from
- `idx_unfused_logits` / `idx_unfused_topk` / `idx_unfused_prep` (aiter paged-MQA, top-k / transform,
  index-K quant/store).

## 9. pdBeperf-r2 decode trace (pdrun, 18:35Z, DCP4 decode, CUDA graph, sim 3.61, C40 load, running 24-27)

Source: `pdrun/traces_r2/1792607756.117898-TP-0-EP-0.trace.json.gz` (TP0). Breakdown:
`pdrun/traces_r2/breakdown.TP0.v2.analyst.txt` (`trace_step_breakdown.yihou.py`, with the graph-launch
and GPU-span attribution added).

**Trace coverage**
- Of 60 graph launches (20 cycles x verify, draft, draft-extend), only **1 target-verify launch** has its
  kernels in the trace (2,369 kernels). The other 19 verify launches carry none, so the profiler did not
  capture them.
- 40 draft / draft-extend launches are captured.
- So the verify numbers below are **one sample** (`step[VERIFY bs=24]`, 144 rows). The 13% whole-trace
  "busy" is an artifact of the missing kernels.

**Target verify, bs 24 (144 rows): 70.6 ms of kernels, GPU-busy over the whole span**

| category | ms | % |
|---|---|---|
| comm (RCCL `ncclDevKernel_Generic`, 314 kernels: about 4 per layer) | 17.8 | 25.2 |
| indexer logits (`_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`, 21 calls) | 15.6 | 22.1 |
| MoE gemms | 10.3 | 14.6 |
| indexer top-k (`coop_topk_kernel`) | 9.3 | 13.1 |
| other gemms | 8.0 | 11.4 |
| sparse MLA attention | 3.4 | 4.9 |
| MoE routing / norm / kv write / DCP merge / misc | about 6 | about 8 |

- Verify used the **unfused** indexer (144 > 48 rows), as predicted in section 8.
- Draft and draft-extend launches: 3.41 ms each on average, about 6.8 ms per cycle. They are a mix of
  fused (`logits_hist_m`, `k_scatter`) and unfused indexer kernels.
- Cycle ≈ 70.6 + 6.8 ≈ 77 ms of kernels, against the step spans' period of about 80 ms and the
  decode_stats 66-75 ms/step.

**Against the Variant B rule:**
- The replicated verify indexer is **about 25 ms per verify step** (logits 15.6 + top-k 9.3 + prep 0.4),
  against the 3.7 ms break-even (section 7).
- Derived: B removes about 3/4 (≈ 19 ms) for an overhead of about 2.5 ms at 144 rows (benchrun), so
  roughly **16 ms per verify, about 20% of the cycle**. This is not measured end to end.
- **Variant B is indicated. A/B it with the fused indexer left on.**

**Second lever visible:** RCCL is 25% of the verify step. That supports custom all-reduce ON (perf_knobs
rank 1), gated on correctness. The trace does not split the RCCL kernels by collective
(all-reduce / all-gather / all-to-all); one generic kernel name covers all of them.

## 10. Why Variant B is slower in benchrun T7 (code read; mechanism not measured)

T7 setup (`ledger.benchrun.md:123-143`): **about 34K-token contexts**, fully cached, N = 8/16/24/32, fused
indexer ON. Split is +6 ms/step at N = 16-32, about constant in N, and equal at N = 8 (fused path,
≤ 48 rows).

**1. The split logits kernel does scale with the owned context, not the table width.**
- aiter `deepgemm_fp8_paged_mqa_logits` (aiter `2bb549102`, `aiter/ops/triton/attention/pa_mqa_logits.py:470-590`)
  launches `rows x next_n x SplitKV` programs.
- In the gluon kernel (`gluon/pa_mqa_logits.py`, from `:594`), each program loads `context_length` for
  its row and loops over `cdiv(split_context_length, KVBlockSize)` blocks of that row only.
- Our wrapper passes `next_n = 1` with per-row owned lens (`owned_lens`) and a per-row owned page table.
- So scoring work and index-K reads are about 1/W per row.
- Not the cause: the max table width and `max_seq_len` only size the (uninitialised) logits buffer;
  there is no per-row padding to `max_len`; the recipe's padded-row trim keeps only real rows.

**2. The split adds a roughly fixed cost per indexer layer that does not depend on context.** In
`dsa_dcp_split_indexer.py` + `Indexer._dcp_split_decode_topk`, every one of the 21 layers repeats:
- the `owned_page_table` strided copy;
- `owned_lens` (about 10 elementwise ops);
- the unfused local top-k;
- `pack_candidates` (about 12 ops);
- one RCCL all-gather (benchrun: 56-120 us at 120-320 rows);
- the merge reshapes plus `torch.topk` over `W x 2048` (benchrun: 59-84 us);
- `positions_to_slots` (about 8 ops).

That is roughly 40 extra kernels per layer, about 840 per verify step. All-gather plus merge alone are
about 2.5-3 ms per step (benchrun microbench); the small ops add more. **The total is about what T7
measured (+6 ms).** Inferred from code; the per-op GPU time is unmeasured.

**3. Why no saving shows up at 34K.**
- The saving is about (W-1)/W of the replicated indexer, and that scales with context length.
- At 34K it is evidently smaller than the fixed cost. The PD trace (section 9) had a 25 ms verify indexer
  at bs 24 on AgentX contexts.
- benchrun's own AgentX bins (T5/T6, 25-32 running, all lines, small n) show split **faster**: 108.8/114.8
  vs 123.9/119.7 ms. That is consistent with a context-length crossover, but not proof.
- **The T7 result is not representative of the PD AgentX operating point.** An A/B at AgentX contexts
  (or a trace of one N=32 step at 34K and one at about 100K+) decides it.

**Fixes the code makes obvious** (lower the fixed cost; none changes the result):
- (a) Compute `owned_page_table` / `owned_lens` / the position base **once per forward** and reuse them in
  all 21 layers. They depend only on the metadata, not the layer. This removes about 11 ops x 20 layers.
- (b) Fuse pack + position mapping and merge + slot mapping into two Triton kernels. #42618's
  `dcp_topk_pack` / `dcp_topk_merge` in `kernels/ops/attention/dcp_kernels.py` are a template.
- (c) The all-gather (about 1.5-2.5 ms per step) stays.

So B's fixed cost can drop to about 3 ms per step, but not below. These are worth doing only if the
AgentX-context A/B shows B helps there.
