# vLLM-side changes behind Dynamo+vLLM DSv4.1-Flash PD (NVIDIA) and their status on MI355X / ROCm

Date: 2026-11-19. Research only (no GPU). Dimension: what vLLM itself (not Dynamo) changed or configured to make the
NVIDIA PD rows fast, and whether each item exists / works / is in our image on the ROCm DSv4.1 path.

Tags: **[FH]** first-hand (code file:line, PR page read via `gh`, API row/log read by me); **[2H]** claim made by a PR
author or the vLLM blog for their own setup (NVIDIA hardware unless stated), not re-measured; **[INF]** my arithmetic or
inference. Paths: `V` = `source/vllm` (main 7a62ca0666, 2026-11-19); `IX` = `source/InferenceX` (fetched to origin/main
6ed3774, 2026-11-19 01:49 PST); `RCP` = `V/recipes/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml`; `N` = our image's vLLM commit
`60bd6dd2edd1` (main, 2026-11-16 05:35Z, `vllm/vllm-openai-rocm:nightly-60bd6dd2...`, used by the 427,451 MI355X bar).
"In N" = GitHub compare API `compare/<merge_sha>...N` returned `ahead`/`identical` (merge commit is an ancestor of N) [FH].
Raw pulls: `notes/dynamo/` (PR JSON in `notes/dynamo/prs/`, reproduce JSON and log heads for rows 445572 / 448173).

Target rows (all AgentX, DSpark-5, synthetic AL 3.51 injected by the IX srt driver [FH: IX `infx/launch/drivers/srt/submit.py:208`,
`infx/golden_al_distribution/README.md:185-186`]):

| row | HW | layout | GPUs | conc | tok/s/GPU | p90i | date | vLLM in image |
|---|---|---|---:|---:|---:|---:|---|---|
| 445572 | GB300 | 12P DEP2 + D DEP16 | 40 | 9888 | 1,011,514 | 75.5 | 10-20 | `cu134-nightly-aarch64-dev-20261019` = `v0.32.0rc1.dev281+g49cb2bbf7` (main 10-16) + dev patches [FH: decode log line 401] |
| 448173 | B300 | 5P DEP2 + D DEP8 | 18 | 3584 | 828,209 | 83.8 | 11-17 | `cu134-nightly-96710245` = `v0.34.0rc1.dev142+g96710245e` (main 11-10) [FH: decode log line 441] |
| 448242 | MI355X | colocated TP1xDP2 + UMBP | 2 | 384 | 427,451 | 15.8 | 11-18 | N = 60bd6dd2 (main 11-16) |

---

## 0. Bottom line

1. **Almost every vLLM PR the blog credits is merged and already in our image N** (all 13 merged PRs the blog links - #61725, #61767,
   #62702, #61724, #62698, #59894, #62359, #62240, #62009, #61395, #61868, #61869, #61339 - are ancestors of 60bd6dd2; #61121 is open,
   #59908/#60267 closed) [FH: compare API]. The gap to NVIDIA is therefore not "missing vLLM main commits" in general. It is the subset of
   those changes that are **CUDA-only by construction** (DeepGEMM MegaMoE + symm_mem EPLB #62009, DeepSelect top-k #62240,
   cuMem FABRIC Engram #62359, specialized decode #61868/#61869, FlashMLA MegaAttn backend) plus **recipe-level choices we do
   not run** (PD itself, wide DEP decode with EPLB, decode-only graphs at 6-token granularity, no prefix cache on decode,
   async-KV admission cap, random DP dummy inputs).
2. **On the ROCm DSv4.1 path, EPLB is rejected outright with AITER MegaMoE** ("AITER MegaMoE does not support EPLB")
   [FH: V/vllm/models/deepseek_v4/amd/model.py:639-640, called from :784; V/vllm/models/deepseek_v41/amd/model.py:167-173
   "The current ROCm runner has no redundant EPLB copies"]. The enabling PR #61176 is open since 10-12 with no hardware run
   [FH: gh]. On NVIDIA, EPLB gave +12.0% (DEP16) / +25.0% (DEP32, 72 req/rank) decode throughput [2H: #62009, GB300].
3. **The platform-agnostic CPU-side PD fixes are in N**: #61767 recursive-doubling CPU all-reduce (unconditional),
   #61724 SWA null-head skip, #62702 int32 prompt buffers, #62698 chunked staged writes, #61725 `--max-async-kv-loads-per-step`
   [FH: compare API; V/vllm/config/scheduler.py:90; V/vllm/v1/core/sched/scheduler.py:1076]. They only matter once we run
   a DP decode instance with a KV connector; none has been measured on ROCm.
4. **The NVIDIA recipes' `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` env var is a dev-image knob; upstream vLLM ignores it.** In the
   B300 828,209 row the decode log prints `Unknown vLLM environment variable detected: VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP`
   [FH: row 448173 decode_w0 log line 360], so that row ran **without** the admission cap; the GB300 1,011,514 row's dev image
   did not warn (cap likely active) [FH: row 445572 decode log lines 349-351]. The same B300 row also ran with **EPLB disabled
   for the DSpark draft** (#61395 merged 11-13, after its image) [FH: log line 699] and with **Engram cuMem FABRIC unavailable,
   fallen back to /dev/shm** on prefill [FH: prefill_w0 log line 430]. So 828k on B300 was reached without three of the blog's
   levers. Upstream flag for us: `--max-async-kv-loads-per-step N`.
5. **The NVIDIA PD transfer path (NIXL push + packed MLA rows, #61121) is NVIDIA-tested only and #61121 is still an OPEN draft**;
   upstream `transfer_mode: push` (#64230) is in N, `push_as_nixl` module is dev-image only [FH: gh; RCP:2346-2347]. On ROCm the
   NIXL path has no ROCm-specific code beyond one Mamba sync [FH: V/.../nixl/base_worker.py:3570] and any non-UMBP connector
   turns off the AITER sparse-MLA fast path (SYNTHESIS C1). PD transfer on ROCm is the MoRIIO/Mooncake question of the other
   dimension files, not something this list unlocks.
6. **DBO / microbatching is not used by any NVIDIA DSv4.1 PD recipe** [FH: no `enable-dbo`/ubatch flag in IX
   `dsv41flash/vllm/{b300,gb300}-fp4/agentx/disagg-variants.yaml`], and on ROCm `use_ubatching` turns SWA bounded replay off
   [FH: V/vllm/models/deepseek_v41/amd/model.py:1135-1142]. Not a lever.

---

## 1. Item-by-item: what changed, effect, platform, ROCm status, in N

Columns: **Plat** = NV-only (CUDA kernel / CUDA driver feature) vs agnostic (core / scheduler / Python); **ROCm DSv4.1** =
whether the ROCm model path uses it or has an equivalent; **In N** = merged commit is an ancestor of 60bd6dd2.

### 1.1 PD / KV-transfer side (blog "CPU overhead" table + recipes)

| PR / item | What it changes (code) | Measured effect (source) | Plat | ROCm DSv4.1 status | In N |
|---|---|---|---|---|---|
| #61121 NIXL: transfer DSv4.1 packed MLA rows whole (OPEN, draft, upd 10-29) | gate in `nixl/base_worker.py` from `not _is_csa_linear` to `use_mla and not _has_mamba`; stacked on #61074 | 1P1D TP4+TP4 GB200, 100k prompt: NIXL descriptors 11,118 -> ~430, bytes 179.2 -> 103.6 MB per 100k transfer, xfer 4.56 -> 2.83 ms, TTFT p50 unchanged (1078 -> 1057 ms c1) [2H: #61121 body]; blog row "NIXL push + packed rows": 141K -> 440K tok/s/GPU, p90 ~40 -> 159, GB300 8P1D c3072 [2H: blog] | NIXL (UCX/LIBFABRIC); logic agnostic | NIXL has no ROCm-specific DSv4.1 handling [FH: grep nixl/*.py]; main still has the `_is_csa_linear` gate [FH: V/.../nixl/base_worker.py:427, 779-816]; no ROCm run found | **No** (open) |
| #61074 NIXL: register packed MLA rows as one region | per-row region registration | prerequisite of #61121 [2H] | agnostic | not reached by DSv4.1 without #61121 [2H: #61121 body] | Yes (merged 10-15) |
| #64230 NixlConnector `transfer_mode: push` | selects `NixlPushConnector` (WRITE) | (selection only) | agnostic | untested on ROCm | Yes (merged 10-28) |
| #66795 NIXL push: stop WRITE submission stalls, early block release | push worker robustness | - | agnostic | - | **No** (merged 11-19) |
| `push_as_nixl` module (`kv_connector_module_path`) | dev-image connector used by GB300/B300 recipes | - | - | not in vLLM main [FH: grep V; RCP:2346-2347 "come from the dev image ...dev-20261012-59e75f4"] | No |
| #61725 `--max-async-kv-loads-per-step` (MERGED 11-10) | caps new async KV loads started per `schedule()`; waits in `async_kv_waiting` [FH: V/vllm/config/scheduler.py:90-99; V/vllm/v1/core/sched/scheduler.py:1076, 2803-2804; V/vllm/engine/arg_utils.py:613] | one `schedule()` with 140K-token requests: 174.3 ms at in-flight cap 64 vs 5.8 ms with limit 2; slowest-rank schedule 28.7/48.2 ms -> 14.6 (limit 4) / 9.0 (limit 2) [2H: #61725 body]; blog: 460K -> 516K tok/s/GPU, p90 109 -> 153, GB300 7P1D c4800, together with two other CPU fixes [2H] | agnostic | applies to any async-load connector (Mooncake/MoRIIO/NIXL) on a DP decode; unmeasured on ROCm | **Yes** (flag name; env var `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` is NOT read upstream, see s2.1) |
| #61767 recursive-doubling CPU all-reduce for DP metadata (MERGED 10-15) | `cpu_all_reduce_()` in `V/vllm/distributed/utils.py:59-100`, used by V2 worker DP padding sync (`V/vllm/v1/worker/gpu/dp_utils.py:80`) and engine DP state sync; unconditional | DEP32: DP sync 10.8 -> 1.5 ms/step, step 16.10 -> 13.15 ms [2H: blog] | agnostic (gloo) | used whenever DP>1 with async scheduling (NCCL DP sync is disabled then, `V/vllm/config/vllm.py:1797-1803`; seen in NV logs [FH: 448173 decode line 375]) | Yes |
| #59908 int32 prompt arrays to workers | - | - | - | superseded | CLOSED |
| #62702 prompt token ids as one int32 buffer end to end (MERGED 11-05) | `TokenBuffer` int32 array, Rust frontend protocol, MRV2 staging | 128K prompt: step critical path pickle+unpickle+staging 6.86 -> 0.17 ms; input thread 3.99 -> 0.24 ms; EngineCore 6.45 -> 1.68 MB/request [2H: #62702 body] | agnostic | applies to any DP rank admitting long prompts (our CI already uses `VLLM_USE_RUST_FRONTEND=1`) | Yes |
| #61724 skip SWA null head of block tables (MERGED 10-15) | `null_head_len()` binary search; `V/vllm/v1/core/{kv_cache_utils,block_pool,single_type_kv_cache_manager}.py` | blog: with #62702/#62698, 855K prompt CPU per admission 38.8 -> 6.0 ms [2H] | agnostic | DSv4.1 has 43-layer SWA groups -> applies; note our `--block-size 128` makes SWA tables 4x shorter than the 32-block NV case [INF] | Yes |
| #62698 MRV2 staged writes parallel over 1024-element chunks (MERGED 10-25) | `V/vllm/v1/worker/gpu/buffer_utils.py` | 855K staged write 443 -> 3.7 us; per-admission rank skew ~200 -> ~10 us in DEP decode [2H: #62698 body] | Triton kernel, agnostic | MRV2 is the only runner on N (V1 removed) [FH: N `vllm/platforms/interface.py:1568`] | Yes |
| `kv_lease_duration: 120`, `enforce_handshake_compat: false`, `kv_load_failure_policy: fail` | NIXL push config in recipes [FH: IX b300 disagg yaml:93,146] | - | NIXL | MoRIIO/Mooncake have their own equivalents (not mapped) | - |

### 1.2 Prefill-side model changes (blog "Optimizations hidden behind the trace")

| PR / item | What it changes | Measured effect | Plat | ROCm DSv4.1 status | In N |
|---|---|---|---|---|---|
| #59894 trim the last KV-source layer (layer 20) in decoder replay (MERGED 10-09) | layer 20 writes KV for all rows (`write_kv` -> `attention.forward_kv`, V/vllm/models/deepseek_v41/attention.py:805), then runs only on the 128-row replay batch | +19.9% prefill at 32K new (10.1% at 4K, 23.1% at 64K) [2H: blog]; PR: +3.6/+4.3/+5.5% total tok/s at 8k/32k/128k in, DEP4 B200; AgentX c128 output +2.4% [2H: #59894 body] | agnostic model code | **ported**: AMD model replays from `cut = max(kv_source_layer_ids)` with `self.layers[cut].write_kv` [FH: V/vllm/models/deepseek_v41/amd/model.py:570-610, 714-752 (main); `def write_kv` at :551 in N]; ROCm decoder replay enabled by #59039 (10-13), prefix-cache SWA replay on gfx950 by #64094 (10-31, +13.5% c128 / +12% c256 on "best MI355X config" [2H]). Off with PCP, SP, ubatching, PP split, Engram at/after 20, drafter window >128 [FH: amd/model.py:1119-1160] | Yes |
| decoder-replay CUDA graphs (`decoder_replay_cudagraph_capture_sizes`, `nvidia/decoder_replay_cudagraph.py`, #63155) | NVIDIA prefill captures the replay batch in graphs under `VLLM_COMPILE` + `FULL_AND_PIECEWISE` [FH: IX gb300 disagg yaml:124] | not separately reported | NV path | ROCm: no `decoder_replay_cudagraph.py` under `amd/` [FH: ls]; DSv4.1 "does not support torch.compile", ROCm needs `VLLM_USE_BREAKABLE_CUDAGRAPH=1` [FH: RCP:227-231]; ROCm prefill steps run replay "in eager prefill steps" (same log text on NV) [FH: amd/model.py:745; 448173 prefill log line 437]. Whether ROCm prefill steps are graph-captured at all: not established | - |
| #62359 Engram host tables as host-NUMA cuMem with 2 MB pages (MERGED 10-29) | `nvidia/ops/engram_host_table.py`; FABRIC handles | +2.8% prefill at 32K [2H: blog] | **NV-only** (CUDA FABRIC) | ROCm counterpart #65595 (`VLLM_ROCM_ENGRAM_HIP_VMM=1`, opt-in, dmabuf) merged **11-18**: placement 38 s -> 5 s, "Lookups show no reliable gain" (3045 vs 2946 us), MI350X/MI355X [2H: #65595]; driver leak caveat ROCm/rocm-systems#13573. ROCm private tables already use THP (`use_thp` "Always on for private tables on ROCm") [FH: V/vllm/config/engram.py:66-70]. NB: in the B300 828k row FABRIC was unavailable and it fell back to /dev/shm [FH: 448173 prefill_w0 line 430] | #62359 yes (CUDA only); **#65595 No** |
| #62240 DeepSelect top-k for the prefill indexer (MERGED 10-28) | `V/vllm/model_executor/layers/sparse_attn_indexer.py:899-1006` with `kernel-config sparse_indexer_topk_backend=deep_select` | +15.6% prefill at 32K, +15.2% at 64K [2H: blog]; PR: DEP2 GB200 8x4096 new on 128K prefix 429.8 -> 401.5 ms/step, indexer 64.5 -> 30.9 ms [2H: #62240] | **NV-only** (`torch.ops.deep_select` CUDA ext, SM100) | ROCm indexer is `forward_hip` -> `rocm_aiter_sparse_attn_indexer` with AITER `indexer_top_k_prefill` or `_C.top_k_per_row_prefill` [FH: V/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:153-162, 1624-1640; sparse_attn_indexer.py:1428-1435]. ROCm prefill indexer top-k cost at 128K+ not measured | Yes (unused on ROCm) |
| Combined prefill (3 fixes above) | - | new-token prefill +35.6-43.9% over 4K-64K new on 128K cached; 64K: 71.2K -> 102.5K tok/s/GPU; 8K prefill 87.9 ms (no ctx) vs 189.4 ms (512K cached), 2x GB300 [2H: blog] | - | only #59894 ported | - |
| `embedding_across_dp` + Engram peer group (`peer_group_size`) | prefill: Engram rows split across DP (and across prefill engines) [FH: V/vllm/config/engram.py:52-83] | not reported separately | peer group: **NV-only** ("Requires ... CUDA FABRIC handles") [FH: engram.py:72-77]; embedding_across_dp in-node: agnostic | ROCm in-node peer reads exist when GPUs are 1-hop xGMI (`is_fully_connected`) and `cpu_offload=false`, TP1 [FH: common/engram.py:1092-1112; platforms/rocm.py:1030-1042]; #66500 (11-10, in N) fixed ROCm DP8/EP8 startup with this config, GSM8K 0.970 [2H]; InferenceX #5056 DEP8 Engram-in-HBM 182.7k vs host 175.6k at c256 [2H, local] | Yes (#66500) |

### 1.3 Decode-side (Wide-EP, EPLB, graphs, specialized decode)

| PR / item | What it changes | Measured effect | Plat | ROCm DSv4.1 status | In N |
|---|---|---|---|---|---|
| MegaMoE (DeepGEMM `deep_gemm_mega_moe`) = the NV "Wide-EP" MoE | fused EP dispatch + expert GEMMs + combine through a symmetric buffer [FH: V/vllm/model_executor/layers/fused_moe/deep_gemm_mega_moe.py:3, 192, 434]; recipes do **not** use DeepEP or an `all2all-backend` [FH: IX recipes] | blog: target MegaMoE 13.4 -> 6.0 ms DEP4 -> DEP32 (64 req/GPU, 128K, GB200), rest ~10 ms; decode MoE = 62% of target step at DEP4 [2H] | **NV-only** kernel | ROCm counterpart AITER MegaMoE over MoRI EPv1 shmem (#66644, merged 11-10, in N): requires gfx950, TP=1, PP=1, `--enable-expert-parallel`, `--all2all-backend mori` [FH: V/vllm/models/deepseek_v4/amd/model.py:611-641]. Open #68385 (A4W4 + MXFP4 combine + dispatch/GEMM overlap, needs aiter #8186): +22.2% output tok/s DEP8 1k/8k c256 [2H]. Earlier local work: MegaMoE ~2x slower/layer at 2 ranks [2H: IX #5098]. DEP8 numbers conflict 182.7k (#5056) vs 26,714 (#67377) [2H] | #66644 yes; #68385 no |
| EPLB symm_mem backend + fused replica mapping in MegaMoE + host sync removed from decode loop (#62009, MERGED 10-29) | `V/vllm/distributed/eplb/eplb_communicator.py:752-790` (`symm_mem` "requires CUDA" at :761-762), MegaMoE staging kernels | with #61395: +12.0% DEP16, +25.0% DEP32 (72 req/rank), +26.2% DEP32 (128 req/rank); MegaMoE 14.7 -> 7.6 ms/step; peer copy ~800 GB/s vs ~88 GB/s NIXL [2H: #62009]; blog: balancedness 0.35 -> 0.94 target, 0.21 -> 0.89 draft; rearrange every 3000 steps on 1000-step window [2H] | **NV-only** (symm_mem, DeepGEMM fuse) | **EPLB rejected** with AITER MegaMoE [FH: deepseek_v4/amd/model.py:639-640]; ROCm DSv4.1 has `n_redundant_experts = 0` hard-coded [FH: deepseek_v41/amd/model.py:167-173]; enabling PRs #61176 (OPEN, no HW run; plan: DSv4-Pro 8x MI355X balancedness 0.71 -> 0.97 expected) and #46206 (OPEN) [FH: gh]. EPLB communicators `torch_gloo`/`pynccl`/`nixl` exist with a ROCm branch for NIXL [FH: eplb_communicator.py:455-458, 863] | Yes (not usable on ROCm) |
| #61395 EPLB for heterogeneous DSpark draft (128 experts top-3) (MERGED 11-13) | `V/vllm/config/speculative.py`, `eplb_state.py`, `amd/dspark.py` (1-line) | part of the +12-26% above; draft balancedness 0.21 -> 0.89 [2H: blog] | agnostic control plane, needs a backend that supports EPLB | moot on ROCm until MegaMoE EPLB exists | Yes |
| FULL_DECODE_ONLY graphs on decode at 6-token granularity | decode `compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[6,12,...,6*max_num_seqs]}`, `max-num-batched-tokens = 10 x max-num-seqs` [FH: IX b300 disagg yaml:184-242; RCP runbook] | not reported separately | agnostic | ROCm sparse-MLA/SWA builders report UNIFORM_BATCH (ALWAYS only with adaptive verification) [FH: V/vllm/models/deepseek_v41/amd/rocm.py:335-360; sparse_mla.py:174, 258]; FULL_DECODE_ONLY is used on gfx942 with breakable graphs off [FH: RCP:262-289]; on gfx950 with DP+EP MegaMoE: not tested | - |
| `--no-enable-prefix-caching` on decode | decode never reuses prefixes (KV arrives from P) [FH: IX b300 yaml:167] | - | agnostic | trades the "decode pulls only the uncached suffix" saving (Mooncake path, SYNTHESIS 4.1) for simpler allocation; NV chose it; effect unmeasured | - |
| `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` | idle DP ranks run dummy decode batches with random inputs [FH: V/vllm/v1/worker/gpu_worker.py:1643-1656; envs.py:1639] | not reported | agnostic | applies to any DP>1 EP decode/prefill on ROCm (dummy tokens otherwise identical; effect on EP routing balance not measured) [INF] | Yes |
| `--enable-ep-weight-filter` | each EP rank reads only its expert shard at load [FH: V/vllm/config/parallel.py:310-317] | load-time only; B300 log: "EP weight filter disabled: EPLB expert layout ... does not match (384, 400)" [FH: 448173 decode line 714] | agnostic | startup time lever for DEP on ROCm | Yes |
| #61868 / #61869 specialized decode (+ DSpark on it) (MERGED 10-21 / 11-11) | `V/vllm/models/deepseek_v41/nvidia/specialized_decode/*` (CuTe DSL kernels, ~12.5k lines) | 826 vs 575 P90 tok/s/user at c1, 4x B300 [2H: blog, IX #4332] | **NV-only** | ROCm analog: monokernel (TP only, rejects DP/EP/EPLB [FH: amd/mono_decode.py:316-317]); low-concurrency lever, not throughput | Yes (unused) |
| `vllm bench forward` (#61339, MERGED 10-14) | exact-shape forward timing through `execute_model` across DP/EP ranks, one-shot CUDA graphs [FH: V/vllm/benchmarks/forward/main.py:3-9, 126-136] | method | profiler hooks are CUDA/NVTX only [FH: forward/profile.py:152-173]; timing path platform-neutral in code | ROCm run with DSv4.1 (no torch.compile, breakable graphs) not attempted by anyone found | Yes |
| `DecodeBenchConnector` | engine-level decode-only measurement | method | agnostic | on ROCm it switches DSv4.1 to the Triton sparse path (no `only_restores_canonical_kv` override) [FH: SYNTHESIS C1] | Yes |

### 1.4 Scheduler / engine settings common to both NV roles

| Setting (NV value) | Code / meaning | ROCm/our CI value | Note |
|---|---|---|---|
| async scheduling (default on) | `V/vllm/config/scheduler.py:270`; DP sync moves to CPU gloo [FH: vllm.py:1797-1803] | same default | `async_scheduling_depth` (>2 in-flight steps) exists on main [FH: scheduler.py:275] but **not in N** [FH: N scheduler.py grep] |
| `--prefill-schedule-interval 2` (both roles) | admit prefill every N steps [FH: scheduler.py:239-245] | 11 in the MI355X bar [FH: SYNTHESIS R1] | PD prefill-only engines: irrelevant on D (no prefill); on P it gates chunks; value for PD on ROCm untested |
| `--long-prefill-token-threshold 8192`, `--max-num-seqs 64`, `--max-num-batched-tokens 16384` (B300) / 32768 (GB300) on P | chunk cap per long request | bar: 16384 batched, no threshold | - |
| `--prefix-cache-retention-interval 32768` on P | periodic SWA prefix-cache checkpoints every 32768 tokens in addition to semantic ones [FH: V/vllm/config/cache.py:286-292] | default 0 | agnostic; affects SWA-group prefix hit on agentic multi-turn; untested on ROCm |
| `cache_aware_admission_window` | admit cache-resident requests ahead of cold ones [FH: scheduler.py:247-262] | not set | not set by NV either; in N |
| `--kv-cache-dtype fp8`, `attention backend FLASHMLA_MEGA_ATTN_DSV41`, `indexer_kv_dtype mxfp4`, `indexer_sparse_logits` | NV-only attention backend | ROCm: `ROCM_FLASHMLA_SPARSE_DSV41` only [FH: amd/model.py:176-191]; bar uses fp8_e4m3 | - |
| P host KV: `SimpleCPUOffloadConnector` lazy, 70 GB/rank (B300) [FH: 448173 prefill log 746-758], 64 GiB (GB300 yaml) / 170 GiB per worker (RCP runbook) | in a `MultiConnector` with NIXL | bar: UMBP 449 GB per 2 GPUs | SimpleCPUOffload does not override `only_restores_canonical_kv` -> on ROCm would lose the AITER fast path (SYNTHESIS C1) [FH-derived] |
| `--numa-bind`, `numa-bind-nodes` | process NUMA binding | not in bar | agnostic |
| `--enable-cumem-allocator` (GB300 true, B300 false: "EFA cannot register VMM memory") [FH: RCP:2372] | - | - | ROCm analog unknown for ionic/MoRI |

---

## 2. Observations from the NVIDIA CI logs (first-hand)

2.1 B300 828,209 (row 448173, image main 96710245, 2026-11-10; run 43769338888):
- `Unknown vLLM environment variable detected: VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` on decode [FH: decode_w0 line 360]. The image contains
  #61725 (compare: ahead) whose control is the CLI flag; the env var is not read anywhere in upstream main [FH: grep V]. The
  admission cap was therefore not applied in this row [FH-derived].
- `EPLB is disabled for the DSpark draft model (deepseek_v41). EPLB remains enabled for the target model.` [FH: line 699]; #61395 is not in
  96710245 (compare: behind).
- `Initialized EPLB communicator: SymmMemEplbCommunicator`; async EPLB rearranged 40/40 layers in 1.35-1.81 s [FH: lines 799, 2091-4562].
- Decode: `Available KV cache memory: 192.39 GiB`, `GPU KV cache size: 188,403,438 tokens` per DP rank [FH: lines 983-1021] -> 1,096 B per token-slot
  [INF]; prefill: 88.54 GiB / 47,892,886 tokens -> 1,985 B [INF from FH: prefill log]. Why the two roles differ by 1.8x: not established.
- Prefill: Engram `Host-NUMA cuMem FABRIC handles are unavailable ... share tables through /dev/shm` [FH: prefill_w0 line 430] -> #62359 not effective here.
- `Chunked prefill is enabled with max_num_batched_tokens=1680` and a warning that `max_num_scheduled_tokens is set to 1680 based on the
  speculative decoding settings` on decode [FH: lines 373, 378].
- Async scheduling on (`Disabling NCCL for DP synchronization when using async scheduling`) [FH: line 375].

2.2 GB300 1,011,514 (row 445572, `v0.32.0rc1.dev281+g49cb2bbf7`, main 2026-10-16 + dev patches; run 39207089292 attempt 12):
- No unknown-env warning for `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` (the dev image reads it) [FH: decode log lines 349-351].
- 49cb2bbf7 contains #61767, #61724, #59894, #61074, #61339 but predates #62359, #62240, #62009, #61395, #62702, #62698, #61725 (all merged 10-25..11-13)
  [FH: compare API; merge dates]. Which of those the dev patches carried: not determinable from the log head.
- Decode `Available KV cache memory: 202.09 GiB`, `GPU KV cache size: 188,904,744 tokens` [FH: lines 706-722] -> 1,149 B/slot [INF].

Implication [INF]: the 1.01M GB300 point predates most of the blog's prefill and EPLB PRs as merged upstream; the blog's per-lever
gains were measured on later builds and on different setups, so they are not additive contributions to 1.01M.

---

## 3. Levers transferable to MI355X vLLM (ranked by how directly they map; none measured on ROCm)

| # | Lever | Type | Needs | Experiment hook (SYNTHESIS s5) |
|---|---|---|---|---|
| L1 | PD topology of the NV rows: many small DP2 prefill engines + one wide DP decode, decode with `--no-enable-prefix-caching`, concurrency sized to decode seats (max-num-seqs x DP) | recipe | a working DSv4.1 ROCm PD connector (MoRIIO+#68513 or Mooncake) | E7/E8 |
| L2 | `--max-async-kv-loads-per-step` (NV used 2-6 by point via env) on the decode instance, plus `max_num_inflight_kv_loads` | flag, in N | DP decode + async-load connector | E8 sweep {none, 2, 4, 6} |
| L3 | Decode graphs `FULL_DECODE_ONLY` with capture sizes in steps of (1+num_spec)=6 up to 6 x max-num-seqs and `max-num-batched-tokens = 10 x max-num-seqs` | flag | check FULL_DECODE_ONLY works on gfx950 with AITER MegaMoE + breakable graphs off/on | E6 |
| L4 | `VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1` on any DP>1 role | env, in N | none | A/B on R1 (DP2) and DEP decode |
| L5 | Wide DEP decode with AITER MegaMoE (DEP4/DEP8 in one node) + open #68385 (A4W4, MXFP4 combine, overlap; aiter #8186) | kernel/PR | cherry-pick #68385 + aiter build; resolve #5056 vs #67377 conflict | E6 (decode step vs batch) |
| L6 | EPLB for AITER MegaMoE (#61176) + draft EPLB (#61395 already in N) | PR (open) | cherry-pick #61176, run balancedness log (`log_balancedness`) first to size the gap (NV: 0.35 real vs ~0.96 uniform at DEP32) | E6: real-vs-uniform routing at DEP8 |
| L7 | Engram GPU-resident and DP-sharded on decode (`{"cpu_offload":false,"embedding_across_dp":true}`; ROCm peer reads over xGMI; #66500 in N) | flag | DEP decode only (TP1) | E5/E6 |
| L8 | Prefill engine = DP2 (small DEP) rather than larger DP; blog DEP2 vs DEP4 per-GPU 64.4K vs 53.3K, imbalance loss 5.9% vs 13.6% [2H] | recipe | measure DP2+EP2 MegaMoE vs DP2 + non-EP MoE on ROCm (MegaMoE "~2x slower/layer at 2 ranks" [2H #5098]) | E6 prefill-only |
| L9 | `--prefix-cache-retention-interval 32768` on the prefix-holding role | flag, in N | none | A/B on R2 (colocated) and on P |
| L10 | Prefill indexer top-k at long context: ROCm uses AITER `indexer_top_k_prefill`; NV gained 15.6% at 32K from a faster top-k | kernel | profile ROCm prefill at 64K new / 128K+ cached first | E6 (M4) |
| L11 | `#65595` HIP-VMM Engram host tables (`VLLM_ROCM_ENGRAM_HIP_VMM=1`) | env, **not in N** (11-18) | newer nightly; author reports no lookup gain, faster placement; driver leak caveat | startup only |
| L12 | Methods: `vllm bench forward` (#61339), mirror-vs-lag replay for DP imbalance, real-vs-uniform routing for EP balance, GPU-only vs serving step gap | method | check bench forward runs on ROCm DSv4.1 | E6 |
| L13 | CPU-side admission fixes (#61724, #62702, #62698, #61767) | in N | nothing to do; verify on ROCm that DP decode step skew is small (per-rank `schedule()` time) | E8 diagnostics |
| L14 | Rust frontend + Dynamo-style KV/session-affine routing and decode least-loaded selection | router | vllm-router equivalent policies (not vLLM-engine) | E8 |

Not transferable (NV-only, no ROCm counterpart in tree): FlashMLA MegaAttn backend, DeepSelect, cuMem FABRIC / Engram cross-engine peer
group, symm_mem EPLB + MegaMoE-fused replica mapping, specialized decode (#61868/#61869), NVLink-domain DEP16/32 across trays.

---

## 4. Open questions / gaps

- G1: Whether ROCm prefill steps for DSv4.1 are graph-captured (NV prefill uses VLLM_COMPILE + FULL_AND_PIECEWISE + decoder-replay
  graphs; ROCm cannot torch.compile this model). Needs a startup log of R1.
- G2: Whether `FULL_DECODE_ONLY` capture works on gfx950 with DP+EP AITER MegaMoE and DSpark (UNIFORM_BATCH builders).
- G3: Whether AITER MegaMoE with #61176 is numerically correct and what balancedness ROCm DEP8 shows with real AgentX routing.
- G4: Why NV decode reports 1,096-1,149 B per token-slot and prefill 1,985 B (same model, fp8 KV); relevant to our 2,468 vs 1,630 gap (SYNTHESIS Q7).
- G5: Which dev patches were in the GB300 1.01M image beyond main 49cb2bbf7 (push_as_nixl, admission env; #61121?). The
  IX PR that produced row 445572 and its image build were not read.
- G6: Mooncake/MoRIIO analogues of NIXL push's packed-row registration (descriptor count per block) - not read; belongs to the
  connector dimension.
- G7: `async_scheduling_depth` (main only, not in N) - PR and effect not read.

---

## Verification (adversarial)

Verifier pass on 2026-11-19. I re-opened every source myself: `gh pr view`, `gh api compare/<merge_sha>...60bd6dd2` and `...49cb2bbf7`, `gh api contents?ref=60bd6dd2`, source/vllm main 7a62ca0666, the InferenceX origin/main 6ed3774 recipes, and the notes/dynamo log heads for rows 448173 and 445572.

| # | Claim (short) | Verdict | Correct fact / evidence |
|---|---|---|---|
| V1 | All 13 blog PRs are in 60bd6dd2; #61121 is an open draft | **Confirmed** [FH] | For all 13 merge SHAs, compare `<merge>...60bd6dd2` returns `ahead`. #61121 is `OPEN`, `isDraft=true`, with no merge commit. 60bd6dd2 was committed 2026-11-16T05:35Z. |
| V2 | ROCm MegaMoE rejects EPLB; n_redundant_experts=0; #61176/#46206 are open with **no hardware run** | **Corrected** [FH] | The code claims hold in main 7a62ca0: `deepseek_v4/amd/model.py:639-640` raises `NotImplementedError("AITER MegaMoE does not support EPLB.")`, and `deepseek_v41/amd/model.py:167-173` sets `n_redundant_experts = 0`. Both PRs are OPEN. **"No hardware run" is wrong for #61176.** The author's comment of 2026-10-12 reports 8xMI355X with PR e2b95d2 on nightly fe4df3df0, DP8/EP8 `mori_high_throughput` + `aiter_mega_moe`, random 8192/1024, c512, 2048 prompts. Results: EPLB off 4354 output tok/s vs EPLB on 4256 (**-2.2%**). Balancedness was about 0.55 without rearrange and about 0.69 with rearrange every 200 steps, not about 0.97. GSM8K (400 questions) was 0.970 off vs 0.9475 after 11 rearranges (11 questions flipped off-right/on-wrong, 2 the reverse). #46206 has no hardware run (its body says so). |
| V3 | #62009: +12.0% DEP16, +25.0% DEP32 (72 req/rank), +26.2% DEP32 (128 req/rank); MegaMoE 14.7 -> 7.6 ms | **Confirmed, with scope** [2H] | Numbers match the PR body (merged 2026-10-29). Scope from the body: GB300 NVL72, **decode only**, DSpark 5 drafts, prompts with ISL about 85k (prefix-cached), one redundant expert per rank, throughput measured 120-240 s into steady state after the first rearrangement, n=3. Image `cu134-nightly-aarch64-dev-20261019` = main 49cb2bbf71 plus the #62009 and #61395 files copied in. Decode step went 31.70 -> 24.47 ms. An earlier GB200 run on an older build gained only +2.0% / +3.7% / +7.7% (body). |
| V4 | B300 row 448173 logs: unknown env `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` (decode 360), "EPLB is disabled for the DSpark draft model" (decode 699), prefill FABRIC fallback (prefill 430) | **Confirmed, plus additions** [FH] | All three lines are present at the cited line numbers. Additions: (a) the decode log shows the same FABRIC -> `/dev/shm` fallback (`b300_decode_head.txt:528`), and the B300 decode runs Engram with `cpu_offload:true` (`b300v_448173_config.yaml:168`; decode command at line 348), not GPU-resident. GB300 row 445572 decode uses `cpu_offload:false`. (b) The B300 image base 96710245e (2026-11-10) already **contains #61725**: compare `dc74b8cb...96710245e` is ahead 18 / behind 0. The B300 decode args contain no `--max-async-kv-loads-per-step` (grep count 0 in config and log), so **the 828,209 row ran with no per-step async-KV admission cap**. The recipe value `"4"` (`b300-fp4/agentx/disagg-variants.yaml:144`) was ignored. [FH config+log; effect on throughput not measured] |
| V5 | GB300 1,011,514 row = v0.32.0rc1.dev281+g49cb2bbf7; base predates #62359, #62240, #62009, #61395, #62702, #62698, #61725 | **Corrected** [FH] | The version string is at `gb300_decode_head.txt:344,401`. 49cb2bbf7 = 2026-10-16T05:21Z. Compare `<merge>...49cb2bbf7` = `behind` for all seven listed PRs, and also for #61868 and #61869 (also not in the base). #61767, #61724, #59894 and #61339 are in the base. **But the base is not what ran.** Row 445572 used image `vllm/vllm-openai:cu134-nightly-aarch64-dev-20261019` (`reproduce_445572.json`). The #62009 body names this exact tag as "main 49cb2bbf71 with this PR and #61395 copied over the installed vllm package". The decode log agrees: the EPLB config has `communicator='symm_mem'` (#62009 code), and line 578 reads "EPLB is enabled for MoE part of drafter model" (#61395 behaviour). So pre-rebase #62009 and #61395 code **was in** the 1.01M run. The env `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP=6` (`gb300v_445572_config.yaml:150`) produced no "Unknown vLLM environment variable" warning (the only such warnings are lines 349-351, for other vars). That is consistent with a dev patch reading it [INF]. Whether #62359/#62240/#62702/#62698 were patched in is not established. |
| V6 | Upstream uses flag `--max-async-kv-loads-per-step` and no env; schedule() 28.7/48.2 -> 14.6 ms at limit 4; blog 460K -> 516K | **Confirmed** [FH code, 2H numbers] | `source/vllm/vllm/config/scheduler.py:90` (same field at ref 60bd6dd2); `vllm/v1/core/sched/scheduler.py:1074-1077`. Grep for `MAX_ASYNC_KV_ADMITS` in vllm/ returns nothing, and envs.py at 96710245e has 0 hits. #61725 body line 13: 28.7 / 48.2 ms (mean / p99) at cap 64 -> 14.6 / 14.6 ms at limit 4. Blog table (`notes/critic/...:167`): 460K -> 516K tok/s/GPU and P90 109 -> 153 tok/s/user, GB300 7P1D c4800, "with two other CPU fixes". |
| V7 | Layer-20 replay trim ported to AMD DSv4.1; #64094 +13.5% c128 on MI355X | **Confirmed, with nuance** [FH code, 2H numbers] | `deepseek_v41/amd/model.py:718-752` (main): `cut = max(config.kv_source_layer_ids)`, and `self.layers[cut].write_kv` feeds `DecoderReplayLayers`. The same code is at ref 60bd6dd2 (lines 694-702). Replay applies to **eager prefill steps only** (log string, line 745). #64094 body: the decoder-replay path on ROCm was enabled by #59039, and #64094 only re-enables the **prefix-cache** SWA replay path on gfx950 (other ROCm GPUs stay forced off). #64094 claims +13.5% at c128 and +12% at c256 on "our best MI355X config", and 128k prefill 17,667 -> 27,136 tok/s single-request. Merged 2026-10-31 (138dfa54) and in 60bd6dd2 (compare = ahead). |
| V8 | DeepSelect prefill top-k is CUDA-only; ROCm uses AITER `indexer_top_k_prefill` / `_C.top_k_per_row_prefill` via forward_hip | **Confirmed** [FH] | `sparse_attn_indexer.py:1428-1435` dispatches CUDA/XPU to `forward_cuda` and ROCm to `forward_hip`. `forward_hip` (1498+) calls `torch.ops.vllm.rocm_aiter_sparse_attn_indexer`. `rocm_aiter_mla_sparse.py:153-162,1631,1639` calls `rocm_aiter_ops.indexer_top_k_prefill` / `torch.ops._C.top_k_per_row_prefill`, with 0 occurrences of `deep_select`. DeepSelect lives in `vllm._deepselect_C` (`indexer_topk.py:38-43`). |
| V9 | ROCm counterpart of #62359 is #65595 (`VLLM_ROCM_ENGRAM_HIP_VMM=1`), merged 2026-11-18, not in 60bd6dd2; no reliable lookup gain; 38 s -> 5 s | **Confirmed** [FH status, 2H numbers] | #65595 merged 2026-11-18T03:13Z (bd455e1f), and compare to 60bd6dd2 = `behind`. Contents API for `amd/ops/engram_host_table.py?ref=60bd6dd2` returns 404. Body: placement 4x47 GiB 5 s vs 38 s on 4x MI350X, and mawong-amd measured about 1.7 s vs 5.6 s / 32 GiB on MI355X. Lookups 3045 vs 2946 µs, "no reliable gain". The amdgpu driver host-memory leak with `hipMemSetAccess` (ROCm/rocm-systems#13573) is a caveat. |
| V10 | NV PD decode recipes: FULL_DECODE_ONLY in steps of 6 up to 6x MNS; MBT = 10x MNS; no prefix caching; RANDOMIZE_DP_DUMMY_INPUTS=1; EPLB 16-32 redundant; no DBO/DeepEP/all2all | **Confirmed** [FH] | `git show origin/main:` of both `disagg-variants.yaml` files (6ed3774). Every decode variant follows 6xMNS capture and 10xMNS MBT: B300 MNS 18/40/72/132/168, GB300 76/18/126. `no-enable-prefix-caching: true` (B300:167, GB300:179). `VLLM_RANDOMIZE_DP_DUMMY_INPUTS: "1"` (B300:80,132; GB300:92,144). `num_redundant_experts` 16 (B300:166, GB300:178) and 32 (GB300:209,227). A grep for dbo/ubatch/deepep/all2all finds nothing. Additions: prefill uses `VLLM_COMPILE` + `FULL_AND_PIECEWISE`, MNS 64, MBT 16384 (B300) / 32768 (GB300). The recipes still set the env `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` (2-6), which current main ignores (see V4). |
| V11 | ROCm microbatching turns SWA bounded replay off; Engram peer reads need cpu_offload=false, TP1, 1-hop xGMI; #66500 (in our image) fixed DP8/EP8 startup | **Confirmed, with precision** [FH code, 2H GSM8K] | `_decoder_replay_supported` (main 1119-1156) returns False when `parallel_config.use_ubatching` is set (same at 60bd6dd2:1112; a second ubatching gate is at 60bd6dd2:334). `common/engram.py:1092-1112`: peer path when `not cpu_offload and dp_size>1 and TP==1`. Node-local groups additionally require `current_platform.is_fully_connected` = all pairs 1-hop XGMI (`rocm.py:1029-1045`). The `embedding_across_dp` requirement is stated only in the comment for cross-node groups. #66500 merged 2026-11-10 (3bf570ec), is in 60bd6dd2 (compare = ahead), and reports DP8/EP8 GSM8K 0.9704 strict-match with acceptance length 3.93. |
| V12 | Bytes per KV token-slot: decode 1,096 (B300) / 1,149 (GB300) vs prefill 1,985 (B300); cause not established | **Confirmed** [FH logs, INF arithmetic] | 192.39 GiB / 188,403,438 = 1096.5 B (`b300_decode_head.txt:994,983`). 202.09 GiB / 188,904,744 = 1148.7 B (`gb300_decode_head.txt:706,709`). 88.54 GiB / 47,894,993 = 1984.9 B (B300 prefill). All three logs print "Using DeepSeek's nvfp4_ds_mla KV cache format" with `--kv-cache-dtype fp8`, so the KV format string does not differ between roles. The cause remains open. |

Net changes to the body above:
- **V2.** #61176 has a ROCm hardware A/B: -2.2% throughput and lower GSM8K after rearranges, at balancedness 0.55-0.69. L6 should treat ROCm EPLB as a measured regression in that one setup, not as untested.
- **V5.** The 1.01M GB300 run **included** pre-rebase #62009 (symm_mem EPLB) and #61395 (draft EPLB). "Base predates" is true; "the run lacked them" would be false. G5 is partly answered.
- **V4.** The 828k B300 run had **no** async-KV admission cap: the env was ignored, and the flag existed in the image but was not passed. The B300 decode also used host-offloaded Engram on `/dev/shm`. GB300 decode used GPU-resident Engram. L7 should note that NVIDIA itself splits on this setting.
