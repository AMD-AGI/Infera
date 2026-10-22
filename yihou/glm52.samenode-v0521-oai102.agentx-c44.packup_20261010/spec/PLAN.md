# PLAN v2 (2026-10-10) — debug GLM-5.2 same-node 1P1D P4D4 on sglang v0.5.21 to a passing run

Supersedes PLAN v1 (kept below the line for history). Directive: mission.md "CURRENT DIRECTIVE".
Research of record: workflow `wf_63fbfafe-63a` (5 angles + synthesis + adversarial critic),
JSON at `/shared_nfs/yihou_glm52/agentx44_20261009/logs/research.{wf_63fbfafe,plan+critique}.yihou.json`.
Runtime workspace: `/shared_nfs/yihou_glm52/v0521dbg_20261010/` (sub-workspace for this phase).

## Hypotheses (ranked; evidence first-hand unless marked)

| id | statement | evidence | toggle |
|---|---|---|---|
| **H1** | **FAULT 2 (c44d prefill)**: HiCache host pool at `--hicache-size 450` is `hipHostRegister`ed in 2 chunks (default `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=256`); kernels address the whole pool from ONE translated base (`runtime.cuh:62-76 get_device_accessible_ptr` → `hipHostGetDevicePointer(tensor.data_ptr())`), so any host slot ≥ chunk 0 is unmapped on the device → MAF. Boundary = floor(256 GiB / (64 tok × 44,928 B)) × 64 = **6,118,144 tokens/rank** (pool 10,016,064). Latent in v0519 too (same code, `environ.py:747`); proven v0519 runs used ratio 1.5 ≈ 154 GB (single registration) or never filled the pool. | leader read `pool_host/common.py:140-180`, `runtime.cuh:62-76`, v0519 `memory_pool_host.py:256`; critic checked `mla.py:398` granularity | `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512` (419 GiB pool → 1 registration); or `--hicache-ratio 1.5` |
| **H2** | **FAULT 1 (c44c decode)**: `index_share_for_mtp_iteration=true` → missing/invalid PD DSA seed sets `cuda_graph_compatible=False`, min-reduced over DP → sustained eager DP-padded draft/verify; `pad_batch` does not pad `dsa_topk_indices` → OOB. | `eagle_disaggregation.py:63-108`, `dp_attn.py:54-70,494`, `eagle_info.py:195-201` (research); c44c ~5,900 eager NCCL collectives/decode rank vs 0 in c44d/c44e (critic, first-hand); v0519 history ON faulted 5/5 at 32-86 min, OFF 13 h 33 m clean | `{"index_share_for_mtp_iteration":false}` (cost ≈0.3 %) |
| H3 | v0521-only gfx950 fused DSA decode indexer (4 JIT kernels; unchecked `pt[p]` at `paged_mqa_logits.cuh:150`) faults, esp. on padded/eager rows | research, code read; but decode ran ~20 min clean with it ON + index_share false (c44d/c44e) | `--no-enable-dsa-fused-indexer` + `SGLANG_DISABLE_AITER_FUSED_FP8_DSA_INDEXER=1` |
| H8 | decode mem 0.95 headroom (≈11 GB) | capacity-only in Stage 2 | 0.85 → 0.92 → 0.95 |
| H4-H10 | residual suspects (fused bmm+rope+cache write, packed top-k v2 — already off via TOPK_V2=false, WAR barrier off on HIP, mooncake HIP copy, HiCache JIT/indexer-pool rewrite, MoE sorting policy, deferred decode KV release, prefill non-final-chunk wait-event patch, R003 config) | research | see research JSON; only if faults persist |
| H11 | upstream #37648 class residual (GLM-5.x + EAGLE long-context MAF) | second-hand | replicate final config |

## Experiments (each splits several hypotheses; diagnostics always on)

Diagnostics in every run: rocm-debug-agent if it loads (smoke first), per-rank cumulative host-slot proxy (`Prefill batch` #new-token), eager-fallback counter from decode `NCCL INFO` lines, census, watchdog every 10 min.

| step | config (Δ vs c44d) | what pass/fail decides |
|---|---|---|
| **E0** micro-repro (no model, minutes) | allocate >256 GiB pinned host tensor, register in 2 chunks exactly as `_cuda_host_register`, run a HIP/JIT copy that reads slot ≥ chunk-0 boundary via the base device pointer; then with CHUNK_GB=512 | H1 mechanism confirmed/refuted in isolation, no model needed |
| **E1 relaxed baseline** (user step 1) | `--hicache-ratio 1.5` (≈154 GB, single reg), index_share **false**, decode mem **0.85**, fused indexer **OFF**, CONC **40**, AgentX full 3600 s | v0.5.21 can complete a full run at all (removes H1, H2, H3, H8 together) |
| **E2 fixed target** (in parallel on a 2nd node, or right after E1) | target: hicache-size **450 + CHUNK_GB=512**, index_share false, decode mem **0.95**, fused indexer **ON** (v0521 default), CONC **44**, full 3600 s | if pass → target reached except index_share (H1 fix sufficient, H3/H8 not triggered). If fail → leg + kernel name pick the next relaxation; step through E1→E2 one group at a time |
| E3 (if E2 fails) | bisect E1→E2 by groups: {450+512}, {0.95, C44}, {fused indexer ON} | localizes the trigger |
| E4 optional | index_share **true** + fused indexer OFF, ≥90 min or 2 replicates | whether index_share can come back (H2 vs H3 interaction) — only with ≥2 clean replicates |

Pass = AgentX `passed`, 3600 s window, 0 MAF on all 8 ranks, request errors under threshold.

## Allocation

- 136-138: taken (yaoc 8P4D on 136+138 from ~02:58; limou vLLM on 137). 135 busy. 139-141 have no
  usable dockerd (user). → oai: job **205476** (any node, 24 h, `sleep infinity` hold) PENDING
  QOSGrpNodeLimit; job **202546** (102 pin) PENDING, est. 12:13 UTC. With 2 nodes: E1 ∥ E2.
- Image: build on the allocated node from clone `fb8ae9c9` (base pull + build ≈ 15-20 min);
  identical images already exist on 136/138 (mooncake .so `f22b1364` = proven).

---

# PLAN v1 (2026-10-09) — same-node 1P1D P4D4 on 102, v0.5.21 base, AgentX full CONC=44

Runtime root `R=/shared_nfs/yihou_glm52/agentx44_20261009`. Node `crsuse2-m2m-102`,
job 200276 (ends 11:17 UTC) then 202546. Shim: `PATH=$KV/scripts/shimbin:$PATH`,
`SPUR_SSH_JOBS=crsuse2-m2m-102:<job>`, `KV=/shared_nfs/yihou_glm52/kvcap_20261008`.

Phases P0-P6 completed through P5 attempts c44a-c44e; see debug/working_process.md R001-R006.

## Update 05:40 UTC — nodes and arms

| node | job | arm | config (all: legacy IPC both legs, debug agent, index_share false) |
|---|---|---|---|
| crsuse2-m2m-227 | 205515 (burst) | **E1** relaxed | ratio 1.5, mem 0.85, fused indexer OFF, CONC 40 — after v0519 sanity |
| crsuse2-m2m-067 | 205548 (burst) | **E2** target | 450 + CHUNK_GB=512, mem 0.95, fused indexer ON, CONC 44 — after v0519 sanity |
| crsuse2-m2m-102 | 202546 (oai) | **E2b** | E2 but fused indexer OFF (splits H3 from H1/H8) — Stage-1-proven node, warm caches |
Root causes established so far: H1 (HiCache chunked host registration, R008/R010) and the same-node
"RCCL race" = ROCr IPC abstract-socket PID collision (R012/R013, fix: legacy IPC on both legs).

## Update 05:48 UTC — burst preempted; arms re-placed

Burst jobs 205515 (227) / 205548 (067) were PREEMPTED at 05:43 (amd-burst-qos). Not resubmitted
(3600 s benches on preemptible nodes). Arms now: **E2b on 102** (oai 202546, running; decode-only
port override after R015) and **E1 on 136** (plain ssh, passed v0519 sanity with legacy-IPC fix,
NODE_RW_ROOT). E2 (fused indexer ON) follows on whichever node frees first.
All rounds: legacy IPC both legs, fixed ports < 30000 (R015), debug agent on stderr.

## Update 07:20 UTC — optional arm G: sglang PR #42553 (HiCache host pool via hipHostMalloc/GTT)

Assessment: debug/working_process.md R021. Implemented as an opt-in overlay, not enabled in any
running round: `harness/config.yihou.gtt42553.sh` (GTT_BASE_CONFIG = any E-config or mc4340
overlay) → mount `mounts-v0521/src/pool_host_common.yihou-pr42553.py` (v0521 common.py
781e2481 + PR hunk @ 8c27ef48) and prefill env `HSA_USERPTR_FOR_PAGED_MEM=0`. Dryrun diff vs e1b
= exactly the mount (both legs; inert on decode) + the env (prefill). Guard refuses
`--hicache-size` whose 4-rank KV+indexer need > 95 % of GTT 1477 GB (450, 300 refused; 280 ok).
Use: after E1 passes, A/B **E1 vs E1+G** (same node, same CONC) — tests the PR's stall/throughput
claim on our nodes (numa_balancing=0 here, so the gain may be small). Also a candidate size-reduced
target (hicache ≤ 280 + G) if 450 + CHUNK512 proves unstable. NOT usable with hicache 450.
Pending before first use: GPU-backed PR unit test + one 150 GB hipHostMalloc/DSA write-back smoke
(needs a free GPU; both nodes busy now).

## Update 09:35 UTC — E1 PASSED; E2b running

- **E1 PASS** (e1c, 102, R025): 4174 ok / 2 err, 168,389 tok/s, MAF 0. Chain: mc4340 image, default
  IPC, private PID + DECODE_PID_OFFSET=30000 (R020/R022; spur-authz denies --pid host).
- **e2b4** (102, launched 09:26) = E1 + {hicache 450 + CHUNK_GB=512, decode mem 0.95, CONC 44};
  fused indexer still OFF. Pass → only {fused indexer ON} left (E2). Fail → bisect the three groups.
- Then: E2 (fused ON) → target. Optional: arm G (PR #42553) A/B; real-acceptance AgentX run.

## Update 13:20 UTC — TARGET PASSED (e2c, R027)

e2c = full mission target on v0.5.21 passed AgentX full C44 (4179 ok / 2 err, 164,194 tok/s, MAF 0).
Results table: RESULTS.md. Stack left up on 102 awaiting user decision (replicate / packup / extra
arms: index_share true, real acceptance, arm G / stop).
