# GLM-5.2 decode "Memory access fault" with index_share_for_mtp_iteration=true — root cause and fix (2026-10-20)

Image `infera-sglang:v0521-yihou-1010-mc4340` (sglang `9f5d8712` + local diff), same-node 1P1D P4D4 on spur MI355X,
EAGLE 5/1/6 on decode, triton DSA, TP4/DP4 + DP attention, AgentX C56.

## Symptom
Decode worker: `Memory access fault by GPU node-N ... Reason: Unknown` -> `Fatal Python error: Aborted`, 23-100 min into
load, only with index_share=true (g4n1, g4n2, g4n2b, g4n4, dbge1a, dbge1b, dbge4a). dmesg (CAP_SYSLOG container):
UTCL2 client TCP (shader), PERMISSION fault, 2 MiB-aligned page.

## Faulting kernel (ROCm debug agent, 2 runs, same pc)
`sgl::dsa_topk::coop_topk_kernel<2048,12,4096,1024>(CoopParams<2048>)` (AOT sgl_kernel common_ops), pc +700 = the
`s_waitcnt` after `global_load_dwordx4` in the round-0 histogram row scan. Registers (disassembly of the image's code
object): row 0, row_starts null, fused page-table transform; row length 60314 / 99462 / 19238 while the score buffer
ends < ~46 KB after its start -> a length far beyond the score row.

## Faulting call (call-site guard in sgl_kernel/top_k.py, synchronous on eager calls)
`Indexer._forward_cuda_k_only` (dense-skip path) -> `metadata.topk_transform(dummy_logits)`; dummy logits are
`index_topk` (2048) wide; lengths up to 104740. Batch dump at every BAD call (32 samples, 2 nodes):
`ForwardMode.IDLE`, bs 1-2, seq_lens = seq_lens_cpu = [1], spec_info EagleDraftInput, DP padding
(global_num_tokens [1|2]x4) — but the indexer's attention metadata held cache_seqlens / seqlens_expanded of an earlier
real batch (e.g. [104740]), page_table_1 (1, 1048651).

## Mechanism (code, image tree; unchanged on upstream main 16eb87c06f)
1. `EAGLEWorkerV2.seed_dsa_topk_from_draft_extend = index_share and seed width` (eagle_worker_v2.py:294). A PD decode
   batch without a DSA seed sets `EagleDraftInput.cuda_graph_compatible = False` (:1033, :1228); the bit is min-reduced
   over DP (`dp_attn._spec_input_cuda_graph_compatible`, "forces every DP rank eager"). Never False with index_share off.
2. An IDLE DP rank then drafts eagerly: `draft()` skips `draft_attn_backend.init_forward_metadata` for idle (:651-659);
   `_draft_forward_idle` (:869-905) runs each step under `ForwardContext(attn_backend=draft_attn_backend.attn_backends[i])`.
3. `ModelRunner._forward_raw` keeps that outer context (model_runner.py:1891). `EagerRunner._execute_idle`
   (eager_runner.py:452-463) re-initializes `model_runner.attn_backend`, which for the draft runner is the
   draft-extend backend (eagle_worker_v2.py:389) — not `attn_backends[i]`.
4. The DSA indexer reads `get_attn_backend()` = `attn_backends[i]` -> stale metadata. `_should_skip_logits_computation`
   sees IDLE with seq_lens_cpu max 1 <= 2048 -> k-only path ranks 2048-wide dummy rows with stale ~1e5 lengths ->
   coop_topk_kernel over-reads: MAF if the next page is unmapped, silent garbage otherwise (dbge4b: 4 BAD, no MAF).

## Fix (eager_runner.yihou-idlefix.py, additive)
In `EagerRunner._execute_idle` (bs > 0), after `model_runner.attn_backend.init_forward_metadata(forward_batch)`, also
initialize the active forward-context backend when it differs (`has_forward_context()` / `get_attn_backend()`).

## Evidence for the fix
| run | fix | guard | result |
|---|---|---|---|
| dbge5a / dbge5b | no | SKGUARD mask (clamp) | 12 / 20 BAD in ~15-20 min (24 by job end on 250), 0 MAF (clamped) |
| dbge6a | yes | SKGUARD count (no clamp) | full A1 (88 min of bench), 0 BAD, 0 MAF; DP0 pd_seedless 1360, draft_eager 3653 / 83297 iters |
| idxf1a | yes | none | A1+A2 complete (~170 min of bench, 14:59-17:49Z): 0 MAF; A2 score 21814 (idx false same node: 22045) |

Workspace: /shared_nfs/yihou_glm52/idxshare_dbg_20261020 (dumps, disassembly, guards, envs). Ledger: poll.md (spur).
