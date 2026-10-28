> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Notes — integration method, traps, open questions (what / why / how / context)

## 1. Integration method (the reusable part)
- What: `overlay/sitecustomize.py` (chains the distribution's sitecustomize, then `infera_kernels.install()` when any
  `INFERA_KERNEL_{SPARSE,INDEXER,MOE}=1`) + `overlay/infera_kernels/__init__.py`: a meta-path finder wraps the loaders of four sglang
  modules and patches them right after import, before any `from ... import` binds the names:
  - `kernels/ops/attention/dsa/triton_sparse_mla.triton_sparse_mla_fwd` -> KF sparse MLA
  - `srt/layers/attention/dsa/dsa_indexer.aiter_paged_mqa_logits` -> KF logits
  - `srt/layers/attention/dsa/dsa_topk_backend.DSATopKBackend.topk_transform` -> KF top-k (legacy sgl-kernel PAGED path)
  - `srt/layers/moe/moe_runner/aiter.AiterRunnerCore.run` -> `glm52_moe_mxfp4`
- Gates (else the original runs): TARGET_VERIFY only for sparse + indexer (recorded by wrappers on
  `DeepseekSparseAttnBackend.forward_{extend,decode}` and `Indexer._get_topk_paged`); 64 heads, d_v 512, no topk_length;
  indexer rows <= 128 with row-expanded tables, `SGLANG_OPT_USE_TOPK_V2=false`; MoE EP1 (no expert_mask), per_1x32,
  257 experts, topk 9, M 97..455, no MORI output arena, no fp8 activations, dispatch policy 2.
- Safety: any exception on the KF path disables that kernel for the process and falls back (`[infera-kernels] ... RAISED`);
  the first eligible eager call (sglang's pre-capture warm-up) also runs the original and logs `CHECK` (SNR / set
  equality). `FORCE_ORIG=True` routes everything to the original (used by `kbench.registry.py` for in-process A/B).
- Harness hook-in: `config.registry.kf.sh` = `config.registry.dcpx.sh` + (KF_ON=1) `DECODE_MOUNTS_ADD` (overlay at
  `REDACTED_LOCAL_PATH`) + `DECODE_EXTRA_ENV` (`PYTHONPATH=REDACTED_LOCAL_PATH INFERA_KERNEL_*=1`). One variable, dryrun-verified.
- Scope limit: as written it does not engage on EP4 MoE, on the DCP sparse/indexer paths (`_forward_triton_dcp`,
  `_dcp_split_decode_topk`), or with TOPK_V2=true.

## 2. Trap: audit FAIL "speculative_algorithm='EAGLE' (want None)" on the prefill (kfa1)
- Why: the env files `export H=<shared harness>`; chain_cap2 `source`s the env and then used `$H/pipeline.v3`, i.e. the
  SHARED run.v3 + audit_live.v3, which does not know prefill MTP (the v3x audit does). The private audit copy has the fix.
- How: the private chains reset `H=$KH` right after sourcing the env. kfa1 was continued with `chain_cont.registry.sh`
  after a manual `run.v3 audit` (private copy) passed on the same live stack (`audit.manual.txt`). kfb1/kfb2/kfa2 passed
  the audit inside the pipeline.

## 3. Trap: GPU memory access fault with the indexer patch (compute-cluster, v0.5.21, 2026-10-23 04:13Z)
- What: indexer-only server faulted right after `indexer_topk rows=82 -> KF` during a prefill-time batch. The KF top-k
  indexes `page_table_1` by row; outside TARGET_VERIFY the table is not row-expanded. An earlier all-KF server hung
  (3 GPUs at 100 %, py-spy: all schedulers waiting in `graph_variants.select`) — same cause suspected, not proven.
- How: TARGET_VERIFY gate + `page_table_1.shape[0] == rows` (notes 1).

## 4. Trap: KFD gate on cluster cluster nodes
- `rocm-smi --showpids` always lists cluster's `gpuagent` (cluster-amd-exporter, 0 VRAM) -> chain_cap2's "KFD processes
  already on the node" STOP. Private chain counts only KFD processes holding VRAM. Never touch cluster-amd-* containers.

## 5. Trap: two servers initialised together on one node (compute-cluster)
- The second server's `ncclCommInitRank` failed ("unhandled cuda error") when both initialised together, with and
  without `--ipc host`; started after the first was up it worked. Cause not established. (Same symptom noted in
  historical-e2e-workspace decode_only.registry.sh, which burns PIDs for slot b.)

## 6. 0.5.22 vs v0.5.21 differences that mattered
- `q8kv8_topk_length_from_indices` moved to `sglang.kernels.ops.attention.metadata` (candidate + task copy edited).
- `AiterRunnerInput.output` (MORI arena) and `AiterMoeQuantInfo.fp8_activations` exist -> MoE gate checks them.
- aiter 2bb549102 vs e7d2453: `fused_moe.py` differs by ~200 lines (EP aux sort, swiglu_limit plumbing); the hooked
  functions keep their names/signatures; kernel A/B + in-model CHECK (51-52 dB) passed.

## 7. Shapes (the user's question for DPA): per-rank decode batch is small
- A2 windows: mean per-rank bs 5.38 (base) / 4.60 (KF); bs <= 8 in 88 % / 93 % of samples (`results/decode_bs_distribution.txt`).
  The KF campaigns tuned bs 8-14 (workset batch model N(9, 5/3)). Sparse MLA and MoE still engage at these shapes,
  the indexer engages only at 60..120 rows.
- Open: why TARGET_VERIFY calls with <= 48 rows never reach the KF indexer (gate log shows only `-> orig` for them);
  per-kernel in-model time split (no profile taken); why TTFT p50 rises ~1-1.8 s with KF.

## 8. Correctness
Simulated acceptance hides model-quality regressions. Evidence so far is kernel-level only: indexer top-k sets identical
to production (workset + in-model warm-up), sparse MLA 32 dB vs fp32 (production 26.5-36.6), MoE 46-52 dB vs production.
A real-acceptance run (accept length + GSM8K) with KF on is still to do.

## 9. DCP4 decode (ppdc2 of `glm52.ppdc-pp4-dcp4.packup_20261023`): where the load lands (code read 93034eba91 + its samplers)
- Decode is DP1 (TP4 EP4 dcp4): every rank sees ALL running requests. Running requests per decode sample
  (`results/rounds/ppdc*/metrics.A*.tsv.gz`, nonzero samples): ppdc1 C56 mean 32 (p10 19, p50 34, p90 43);
  ppdc2 C64 mean 39 (p50 40, p90 52); ppdc2 C72 mean 45 (p50 47, p90 60). TARGET_VERIFY rows = 6 x that: ~200-290 typical.
- Sparse MLA: `_forward_triton_dcp` -> `triton_sparse_mla_decode.triton_sparse_mla_decode_splitk` (returns lse, per-row
  `owned_counts` = the slots of the global top-2048 this rank owns, ~512 on average), q heads all-gathered to 64
  (`num_dcp_q_heads = 16 x 4`). KF tuned `triton_sparse_mla_fwd` for 48-84 rows x 2048 slots -> different kernel, different
  shape; the overlay does not hook it.
- Indexer: `_dcp_split_decode_topk` -> `aiter_paged_mqa_logits` over this rank's pages (rows ~200-290, context L/4) +
  `topk_backend.topk_func` (local, not `topk_transform`) + triton pack/merge across ranks. KF logits cap 128 rows; KF top-k is
  the page-transform variant -> neither engages.
- MoE: EP4 (expert_mask) -> outside the KF MoE (EP1). User: do not pursue EP4; EP8 later when a machine is found.
- Net: as written the overlay is a no-op on the DCP decode. Matching DCP needs new KF targets (DCP split-K sparse decode at
  ~200-290 rows x ~512 owned slots x 64 heads + lse; local indexer logits at ~200-290 rows over L/4).

## 10. DCP A/B measured (same-node ppdc1, C56, 3600 s, `results/summary.dcp.tsv`)
kda1 (compute-node-05, KF off) 27,062 / 27,051 tok/s/chip, TPOT 19.44 / 19.26 ms; kdb1 (compute-node-06, KF on) 27,077 / 27,033,
TPOT 19.38 / 19.26 ms -> no difference, as the engagement log predicts (`results/kf_engagement.kdb1-dcp.txt`: no
triton_sparse_mla_fwd decode call, MoE all `-> orig`, logits KF only at rows 72/96). Same-node ppdc1 matches the
cross-node 27,426 of the handoff. New KF campaigns for the DCP shapes: `tasks/dcp_sparse`, `tasks/dcp_indexer`
(REDACTED_LOCAL_PATH started 2026-10-23 13:00Z on compute-cluster compute-node-10).

## 11. DCP KF kernels in e2e (2026-10-24, compute-node-05, same-node ppdc1, C56) — slower; indexer is the cause
- KF kernels: dcp_sparse iter5 (6af13db) + dcp_indexer iter4 (09f754e), overlay switches INFERA_KERNEL_DCP_{SPARSE,INDEXER}
  (config `KF_DCP=1`, `KF_DCP_SPARSE/INDEXER`). Kernel level in the v0522 image (`results/dcpk.v0522.compute-node-05.log`):
  sparse 0.0635 -> 0.0385 ms weighted, indexer 0.332 -> 0.141 ms, both exact; all captured row counts checked
  (`results/dcp_*_allrows.v0521img.txt`: never slower than prod on synthetic data).
- e2e (TPOT mean / p90 ms, score): kda1 base 19.44 / 21.58, 27,062 (A1) | kdc1 both KF 21.33 / 26.68, 25,578 (A1),
  21.13 / 26.76, 25,683 (A2) | kde1 indexer-only KF 21.51 / 26.74 (A1, from profile_export.jsonl; the stack was stopped
  before aiperf wrote its JSON — the same recomputation gives 19.43 for kda1). -> the KF DCP indexer makes the decode
  slower in-model although it is 2.0-2.7x faster on the driver's synthetic inputs; sparse is neutral-to-positive.
- Cause NOT established. Measurements that would answer it: per-kernel in-model timing of the indexer path (profiler on
  a live DCP stack), and the KF selector's fallback counters (`_SEL_DEBUG` in kf_dcp_indexer) on real logits; also
  whether the 6-sibling-row page-table fusion in `_dcp_logits` falls back to single-row passes on real tables.
- Consequence: DCP KF tasks need real captured inputs (logits/tables/lens/q from a live stack) instead of synthetic ones.

## 12. DCP: sparse-only round and the indexer on REAL captured inputs (2026-10-24)
- kdg1 (compute-node-07, DCP KF sparse only): 27,306 / 27,400 tok/s/chip, TPOT 18.92 / 18.82 ms vs kda1 base (compute-node-05)
  19.44 / 19.26 -> -2.5 %, inside the ~3 % node-to-node spread (different nodes).
- kdf1 (compute-node-02): eager decode (TPOT 56 ms, not a perf number) with the overlay's capture hook: index-K pool of layer 10
  + 24 TARGET_VERIFY calls on DCP rank 0 (`$R/kdf1/dcpdump`, 1.6 GB). `results/dcp_indexer_realdata.kdf1.txt`
  (`scripts/kernel/dcp_indexer_realdata.registry.py`, v0522 image, prod pipeline vs KF iter4, other ranks' packs recomputed):
  * 20/24 calls: KF 1.36-2.35x faster.  4/24 calls: KF 0.20-0.28x (1.8-2.5 ms vs 0.5 ms): exactly the calls where
    the KF selector's candidate count exceeded its cap (max 8,509-9,169 > 8,192) and 1-6 rows took the slow exact
    fallback. Synthetic driver data never reached the cap. -> this is the in-model regression.
  * KF is NOT exact on real data: 1-3 rows per call differ from the production slot set in 11/24 calls (prod
    reproduces the reference in every row). Cause not established.
  * Sibling page tables were identical in every call (the 6-row fusion of `_dcp_logits` always applied).
  * Real contexts: max ~690k, mean 72-200k tokens; rows 96-366.
- Next for DCP (lower priority per user): use these dumps as the KF dataset (real logits distribution), make the
  selector's worst case bounded, and gate on exactness against production.
