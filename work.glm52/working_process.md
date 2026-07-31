# GLM-5.2-MXFP4 sglang bring-up — working process

Mission: 3 phases (mix / PD mooncake / PD mori), each correct output + conc=64.
Nodes: chi2878=10.2.122.3, chi2879=10.2.122.10 (8x MI355X gfx950, NIC enp193s0f1np1).
Reference program (对拍 target): /mnt/vast/jiejing/crusoe_glm_52/ (jiejing already solved this stack).
Conc=64 profile decided with user: **1k/1k** (short, fast validation). Order: Phase1 -> Phase3 -> Phase2.

## Round index

- [phase1_mix](phase1_mix/) — single-node mix on chi2879. Recipe = jiejing launch_sglang_infera.sh
  (self-rewritten, image=rocm/infera:sglang-v0.1.0-rc6 per user). STATUS: cold-starting.

## phase1_mix round 1 (2026-07-27 05:53)
- HYPOTHESIS: jiejing's mix recipe reproduces coherent GLM-5.2 on rc6 image; conc=64 (1k/1k) passes.
- CHANGE: container glm52-mix-p1 on chi2879, TP8, max-running-requests 64, cuda-graph-max-bs 64,
  mem-fraction 0.85, image rc6 (user switched from v0.1.1). Direct sglang.launch_server, no etcd.
- CMD: work.glm52/phase1_mix/launch.sh (on chi2879 via /mnt/vast/c_huggingface/p1_launch.sh).
- t+36s: TP0-3 init clean, torch distributed begin, tokenizer loaded (TokenizersBackend note = benign).
  No weight-shape IndexError (rc6 = 0.5.15, knows head_dim=192). Entering weight-load/JIT/cudagraph.
- t~4min: "server is fired up and ready to roll!" (fast — NFS weights warm). cudagraph capture 55s,
  max_total_num_tokens=3.55M, mem-fraction 0.85 → ~267GB/card. /health=200.
- CORRECTNESS (temp=0 probe, work.glm52/phase1_mix/probe.py): **4/4 correct** — France→Paris,
  China→Beijing, 2+2→4, largest planet→Jupiter. Coherent (reasoning-style CoT, GLM-5.2 is a reasoning model).
- CONC=64 (1k/1k, 256 prompts): **256/256 success, 0 fail**. dur 113s, total tput 4621 tok/s,
  concurrency 56.5, median TTFT 2.25s, median TPOT 22.1ms (stable decode), P99 TPOT 23.4ms.
- RESULT: **PASS**. Phase 1 (single-node mix) done. Recipe = jiejing's, image rc6, no code fix needed.

## phase3_mori round 1 (2026-07-27 06:05)
- HYPOTHESIS: jiejing's mori PD recipe (MORI_IB_GID_INDEX=-1 + all 8 ionic + NATS/etcd/router +
  libionic mount) reproduces coherent GLM-5.2 1P1D on rc6; conc=64 passes.
- SETUP: prefill=chi2878(10.2.122.3) memfrac0.85, decode=chi2879(10.2.122.10) memfrac0.70, TP8 each.
  NATS both nodes, etcd+infera.server router(:8100 round-robin) on chi2878. Scripts: work.glm52/phase3_mori/.
- PRECHECK: VRAM drained to 0.3GB/card both nodes (killed jiejing leftover glm52-mori-decode +
  mix container). rail_test chi2878<->chi2879 ionic_0 gid1 = 337.98 Gb/s (fabric healthy).
- t+40s: both legs TP0-7 init clean, torch distributed begin, 8 active ionic NICs enumerated in
  both containers (libionic mount OK). NO MoRI GID ENODATA crash. Weight-load/JIT stage.
- RESULT: pending (cold start).

### phase3 round 1 diagnosis (2026-07-27 06:10) — 2 router-flag bugs, fixed
- BUG A: router launched with `--kv-events off` (an ENGINE flag) -> infera.server rejects it
  ("unrecognized arguments: --kv-events off"), router Exited(2). FIX: drop it (router discovers via etcd).
- BUG B: router launched with `--request-transport nats` -> NATS request path needs JetStream, but
  `nats:2.10` broker started without `-js` => NoRespondersError, router can't dispatch. Prefill
  prefilled the req (#inflight-req:1) but decode never tapped. 对拍 jiejing launch_pd_router.sh:
  jiejing router uses **`--request-transport http --router-policy kv-aware`** (engines use nats, router uses http).
  FIX: relaunch router with http transport + kv-aware.
- NOTE: decode PD-warmup emitted '!!!!' garbage (jiejing's NSA-PD signature) but that's the dummy
  4-tok warmup; REAL requests are coherent. Prefill warmup already 200-OK (KV transfer works over MoRI).
- Evidence the transfer works: prefill log "Disaggregation warmup request completed with status 200".
- After router fix: single probe through :8100 -> coherent ("...capital of France is Paris..."). MoRI
  RDMA KV transfer + NSA decode both correct on chi2878/chi2879 (all NICs have routable GID at idx1,
  so infera's default MC_GID_INDEX=1 is right here; the -1 auto only needed on chi2866's alternating NICs).
- CORRECTNESS (temp=0 probe via router :8100): **4/4 correct** — France→Paris, China→Beijing, 2+2→4, planet→Jupiter.
- CONC=64 (1k/1k, 256 prompts, via router :8100 -> MoRI RDMA PD): **256/256 success, 0 fail**.
  dur 101s, total tput 5167 tok/s, concurrency 63.1, median TTFT 535ms, median TPOT 20.9ms
  (stable decode — PD isolates decode from prefill), P99 TPOT 21ms.
- RESULT: **PASS**. Phase 3 (2-node PD mori) done. prefill=chi2878, decode=chi2879, MoRI over 8 ionic NICs.
  Root fixes were both router-flag mistakes (drop --kv-events off; use --request-transport http not nats).
  GLM-5.2 NSA cross-node PD KV transfer over MoRI RDMA works correctly on this node pair.

## phase2_mooncake round 1 (2026-07-27 ~06:15) MODE=tcp
- HYPOTHESIS: jiejing's mooncake RDMA is a kernel/driver dead-end (kernel 6.8.0 no P2PDMA/
  DMABUF_MOVE_NOTIFY, peermem hard-faults on real xfer) — proven exhaustively by jiejing on same
  kernel family. So go straight to jiejing's proven MC_FORCE_TCP=1 correct-output path for the
  conc=64 functional pass. Then a separate quick RDMA attempt to independently confirm the wall.
- SETUP: same as phase3 but --disaggregation-transfer-backend mooncake + MC_FORCE_TCP=1 +
  MC_DISABLE_HIP_TRANSPORT=1 MC_GID_INDEX=1 MC_IB_GID_INDEX=1 MOONCAKE_DISABLE_HIP_DMABUF=1.
  Router = http+kv-aware (phase3 lesson). Scripts: work.glm52/phase2_mooncake/.
- RESULT: pending (cold start).

### phase2 mooncake TCP result (2026-07-27 06:30)
- CORRECTNESS (temp=0 probe via router :8100): **4/4 correct** — output coherent over mooncake TCP.
- CONC=64 (1k/1k, 256 prompts): **FAIL — only 50/256 succeeded**. dur 115s, agg 886 tok/s,
  concurrency 28 (couldn't reach 64), median TTFT 51s (!). Decode TPOT fine (13ms, gen 156 tok/s).
- FAILURE MODE (not a crash — both legs health=200): prefill log floods with
  `KVTransferError: Decode instance could be dead, remote mooncake session 10.2.122.10:16269 is
  not alive`. mooncake TCP KV-transfer sessions drop under concurrent load; TTFT 51s = TCP handoff
  can't keep up. Matches jiejing "TCP too slow, RDMA is the only viable transport".
- ROOT: mooncake cross-node RDMA is the intended path but a kernel/driver dead-end here (jiejing:
  kernel 6.8.0 no CONFIG_PCI_P2PDMA/DMABUF_MOVE_NOTIFY -> peermem GPU-direct hard-faults on real
  transfer; hip-priority picks intra-node HIP-IPC cross-node). TCP is the only correct-output path
  but can't sustain conc=64.
- STATUS: mooncake PD = correct output YES, conc=64 throughput NO (TCP-bound). Bringing evidence to user.

## phase1b_mtp round 1 (2026-07-27, user pivot) — single-node MTP spec-dec
- GOAL (user): single-node open MTP (EAGLE spec-dec) and test. Phase 2 mooncake paused.
- RECIPE (对拍 jiejing launch_sglang_mtp_caseA.sh): patched deepseek_nextn.py mounted (eh_proj
  bf16 fix — MTP layer 78 fully bf16, sglang's prefix-guard missed submodule exclude), EAGLE
  5-draft (num-steps 5, eagle-topk 1, num-draft-tokens 6), mem-frac 0.80, image rc6. On chi2879.
  I set max-running-requests 64 + cuda-graph-max-bs 64 (jiejing used 24 for single-stream debug).
- WATCH: jiejing hit 'size of tensor a (3072) vs b (6144)' nextn shape crash BEFORE the eh_proj
  fix; with the patch it worked (accept_len ~2.6, decode ~135-159 tok/s, 1.7-2x speedup).
- RESULT: pending (cold start).

### phase1b_mtp round 1 diagnosis (2026-07-27 06:43) — jiejing patch breaks rc6
- CRASH: draft worker init ValueError "'DeepseekV3ForCausalLMNextN' is not a registered model".
- 对拍/source: model_config.py:_config_draft_model maps GlmMoeDsaForCausalLM -> DeepseekV3ForCausalLMNextN
  for the draft (intentional; GLM MTP uses the DeepSeek nextn path). That class IS an EntryClass in
  rc6 stock deepseek_nextn.py (line 264). So why "not registered"?
- ROOT CAUSE: I mounted jiejing's patched deepseek_nextn.py (from her v0.1.1 work) over rc6's stock.
  jiejing's file is structurally divergent (423 lines vs rc6's 365, different imports/API) -> fails to
  import cleanly on rc6 -> EntryClass never registers -> class unresolvable.
- KEY: rc6 STOCK deepseek_nextn.py ALREADY has jiejing's eh_proj fix natively (lines 298-308:
  "For quark, if the MTP layer is listed in exclude_layers, set quant_config to None" -> eh_proj bf16).
  jiejing's manual patch is upstreamed in rc6 => unnecessary AND harmful here.
- FIX: relaunch MTP on rc6 WITHOUT the patch mount. (My rewrite, work.glm52/phase1b_mtp/launch.sh.)
- RESULT: pending.

### phase1b_mtp round 2 (2026-07-27 06:53) — the REAL shape bug + minimal rc6 fix
- Without jiejing patch: hit jiejing's documented crash "size of tensor a (3072) must match b (6144)"
  in deepseek_nextn.py load_weights (eh_proj). So rc6 does have a real bug (not just the import issue).
- ROOT CAUSE (evidence): GLM-5.2 quark exclude list has SUBMODULE entry 'model.layers.78.eh_proj'
  and NO bare 'model.layers.78'. rc6 stock deepseek_nextn.py:305 checks bare
  `ckpt_prefix=f"model.layers.{78}"` -> should_ignore_layer=False -> nextn_quant_config stays
  non-None -> eh_proj built MXFP4-packed (3072) while ckpt weights bf16 (6144) -> shape crash.
- FIX (minimal 1-line, my own patch in /mnt/vast/c_huggingface/glm52_nextn_patch/, NOT jiejing's
  incompatible 423-line v0.1.1 file): ckpt_prefix -> f"model.layers.{78}.eh_proj" so the quark
  submodule exclude matches -> nextn_quant_config=None -> eh_proj bf16. Same idea as jiejing line 363
  but surgically applied to rc6 stock. py_compile OK. Saved to work.glm52/phase1b_mtp/deepseek_nextn.rc6patch.py.
- RESULT: pending (relaunch with patch mounted).

### phase1b_mtp round 3 (2026-07-27 06:58) — nextn loads, but decode hits CUDA JIT kernel
- After the 1-line nextn fix: draft model LOADS ("Load weight end type=DeepseekV3ForCausalLMNextN
  quant=quark"), cudagraph captured, server "ready to roll". A single short completion = coherent.
- BUT the probe (repeated calls) HANGS: decode-time MTP verification calls
  sglang.jit_kernel.fused_metadata_copy (CUDA kernel, #include <cuda_runtime.h>) -> won't compile
  on gfx950 -> "Multi-backend fused metadata copy kernel failed", retries every request -> hang.
- SOURCE (dsa_backend.py:2410): the fused-copy block is gated by
  `if envs.SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA.get()` (default True). num_steps>3 -> multi
  variant, <=3 -> single variant; BOTH are CUDA fused_metadata_copy kernels. The env's `else`
  branch runs plain per-backend replay (no CUDA JIT).
- FIX: add env SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0 to route around the CUDA fused kernel.
- RESULT: pending (relaunch).

### phase1b_mtp round 4 (2026-07-27 07:14) — MTP WORKS
- With SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0: server ready, probe 4/4 correct (no hang).
- SPEC-DEC ACTIVE (proof): decode batch shows accept len 3.52-4.83 (of 6 draft), accept rate
  0.51-0.77, single-stream gen throughput 219 tok/s (vs ~80 no-MTP baseline = ~2.7x). Coherent.
- RESULT: **single-node MTP WORKS on rc6.** Full recipe to enable GLM-5.2 MTP on rc6:
  (1) 1-line patch deepseek_nextn.py ckpt_prefix -> ".eh_proj" (quark submodule exclude match);
  (2) env SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0 (skip gfx950-incompatible CUDA fused_metadata_copy);
  (3) EAGLE 5-draft (num-steps 5, eagle-topk 1, num-draft-tokens 6), plus base DSA ROCm envs.
  Patch + launch saved in work.glm52/phase1b_mtp/.

## phase3b_pdmtp round 1 (2026-07-27 08:08) — PD-disaggregated MTP (decode-leg spec-dec)
- GOAL (user): PD分离的MTP. spec-dec on DECODE leg only (prefill MTP is pointless — prefill doesn't
  gen tokens; jiejing: prefill spec-dec throttles pipeline). Transport = mori RDMA (Phase3 proven).
- TOPOLOGY: prefill=chi2832(10.2.122.79, NO MTP), decode=chi2879(10.2.122.10, MTP=1). [chi2878 taken
  by someone's pd_uni DSv4; user said use chi2832 — killed llying's kimi_pd_debug container to free it.]
  rail chi2832<->chi2879 = 335 Gb/s, both GID idx1. router http+kv-aware on chi2832.
- DECODE MTP config (jiejing KV-pool tuning to avoid PD-conc OOM): EAGLE steps 3 (not 5),
  num-draft-tokens 4, num-reserved-decode-tokens 256, decode mem-frac 0.80. + my nextn 1-line patch
  + SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0. Scripts: work.glm52/phase3b_mtp/.
- RESULT: pending (cold start; chi2832 fresh, pulled rc6).

### phase3b_pdmtp result (2026-07-27 08:17) — PD-MTP WORKS, conc=64 PASS
- Cold start: chi2832 fresh node, cold NFS weights -> prefill ~6min (slow first load); decode ~3min.
  Both registered (prefill@chi2832, decode@chi2879). My nextn patch + precompute env work in PD too.
- CORRECTNESS (temp=0 probe via router :8100): **4/4 correct** — full path router->prefill->MoRI RDMA
  ->decode+EAGLE. Coherent.
- SPEC-DEC ACTIVE on decode leg: accept len 2.75-2.88 (steps=3), accept rate 0.58-0.62, decode gen
  92 tok/s.
- CONC=64 (1k/1k, 256 prompts): **256/256 success, 0 fail** — no high-conc crash (jiejing's KV-pool
  tuning steps=3/reserved=256/memfrac0.80 held). dur 70s, total tput **7444 tok/s** (vs 5167 no-MTP
  Phase3 = +44%), median TPOT **12.1ms** (vs 20.9 no-MTP = 1.7x faster decode), median TTFT 412ms,
  concurrency 59.5.
- RESULT: **PASS. PD-disaggregated MTP works, conc=64 passes, ~1.7x decode speedup from spec-dec.**
  Topology prefill=chi2832(no MTP) + decode=chi2879(MTP), mori RDMA. Scripts work.glm52/phase3b_mtp/.

## phase2b_mooncake_unified round 1 (2026-07-27, user retry w/ PR#19 image)
- GOAL (user): re-attempt GLM-5.2 mooncake PD on infera/engine-sglang:pd-unified (PR#19 fix:
  MC_DISABLE_HIP_TRANSPORT=1 default -> cross-node RDMA not HIP-IPC; dma-buf compiled in, runtime
  switch). No MTP. Method learned from sglang_unified_pd_test.packup (DSv4 conc=128 zero-fail on this image).
- KEY vs my failed Phase 2: pd-unified image + NO MC_FORCE_TCP (real RDMA); dmabuf OFF (bare
  ibv_reg_mr+peermem, the proven cross-node path); sglang_router mini-LB (not infera.server);
  all-8-ionic; RDMAV_FORK_SAFE=1; decode on :30001. GLM-5.2 DSA recipe kept (tilelang/fp8_e4m3).
- IMAGE dist: registry pull denied (local build); NFS docker save too slow (packup warned) ->
  streamed docker save|ssh chi2879 docker load directly (~5min). Both nodes now have it.
- SETUP: prefill=chi2878(10.2.122.3) gmu0.85, decode=chi2879(10.2.122.10) gmu0.80. libionic inject
  OK (8 active ports both). Scripts: work.glm52/phase2b_mooncake_unified/.
- RESULT: pending (cold start).

### phase2b_mooncake_unified result (2026-07-27 09:39) — RDMA WORKS, conc=64 PASS
- Image infera/engine-sglang:pd-unified (PR#19). NO MC_FORCE_TCP — real RDMA. dmabuf OFF.
- Transport confirmed RDMA (not TCP/HIP): prefill log "rdma_context.cpp:75 HIP dmabuf disabled via
  MOONCAKE_DISABLE_HIP_DMABUF" on all 8 NICs. Zero "not alive"/KVTransferError/hipIpc on both legs.
- CORRECTNESS (temp=0 via sglang_router :8002): 4/4 correct.
- CONC=64 (1k/1k, 256 prompts): PASS. Total generated tokens 262144 = 256x1024 (all 256 completed),
  concurrency 63.17, 0 transfer errors. total tput 5147 tok/s, median TTFT 543ms, TPOT 20.9ms.
  ~= mori RDMA (5167). vs my old Phase2 mooncake TCP (50/256, TTFT 51s) — RDMA fully fixes it.
- ROOT: PR#19 defaults MC_DISABLE_HIP_TRANSPORT=1 -> mooncake uses cross-node RDMA instead of
  intra-node HIP-IPC (the hipIpcOpenMemHandle wall). dmabuf compiled-in but OFF by default (ionic
  no-ODP 2x pin); bare ibv_reg_mr+peermem path works on chi2878/2879 (kernel has P2PDMA + ib_peer_mem).
- METHOD from sglang_unified_pd_test.packup (DSv4 conc=128 zero-fail on this image); I swapped DSv4
  recipe -> GLM-5.2 DSA recipe, kept mooncake env + sglang_router mini-LB. Scripts: work.glm52/phase2b_mooncake_unified/.

### phase2b_mooncake_unified + MTP (2026-07-27 09:55) — PASS
- MTP on decode leg (chi2879), mooncake RDMA PD on pd-unified image. prefill (chi2878) no MTP.
- KEY: pd-unified deepseek_nextn.py is 423 lines (NOT rc6's 365) — has the SAME bare-prefix eh_proj
  bug at line 363. Made a pd-unified-SPECIFIC 1-line patch (glm52_nextn_patch_unified/). Do NOT reuse
  the rc6 patch (different structure). Draft head loads clean (type=DeepseekV3ForCausalLMNextN quark).
- KEY2: pd-unified does NOT need SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0 (env doesn't exist here).
  Its dsa_backend.py try/excepts the CUDA fused_metadata_copy (steps>3) AND uses a plain loop for
  steps<=3 -> the gfx950-incompatible kernel is never a hard failure. With steps=3 it's not even hit.
  So the ONLY pd-unified MTP fix needed = the nextn 1-line patch.
- CORRECTNESS (temp=0 via router): 4/4. Spec-dec active: accept len 2.65-2.90, rate 0.55-0.63.
- CONC=64 (1k/1k, 256): **256/256 success**, dur 98.9s, total 5302 tok/s, concurrency 60.7,
  median TTFT 1898ms, TPOT 19.0ms. vs no-MTP mooncake PD (5147/20.9ms) — MTP +3% tput, TPOT 20.9->19.0.
- RESULT: PASS. mooncake RDMA PD + MTP works on pd-unified. EAGLE steps=3/reserved=256/memfrac0.80.
