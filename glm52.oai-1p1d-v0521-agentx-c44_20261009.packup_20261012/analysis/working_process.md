# Working process — debug rounds

## R001 — c44a prefill wedged after first /health (2026-10-09 06:49-07:15 UTC)

**Goal.** Prefill keeps serving requests after the first health generate, as r013 (v0519) did.

**Symptom.** Prefill `/health` 200 once at 06:49:24, then 16+ min of "Health check failed … no
response from detokenizer", `curl /health` timing out; no new `Prefill batch` lines after
06:49:23 (`#inflight-req: 1`); GPUs 0-3 0 % use; prefill caches unchanged; no MAF/TB/OOM.
r013's prefill, same phase: one 1-token health batch → 200, repeatedly.

**Evidence (first-hand).** `py-spy dump` of `scheduler_DP0_TP0`: main thread in
`zmq send ← sock_send ← emit_kv_metrics (kv_events_publisher.py:90) ← report_prefill_stats
(metrics_reporter.py:872) ← process_batch_result_disagg_prefill (prefill.py:1137)`.
`send_metrics_from_scheduler` is a `zmq.PUSH` that *connects* to `metrics_ipc_name`
(`ipc_channels.py:76-77`); nothing in sglang or infera binds a PULL there. Only ranks with KV
events enabled call it (prefill: yes; decode: `--kv-events off`).

**Difference vs. reference.** Stage-1 r012/r013 bind-mounted a modified
`kv_events_publisher.py` (`NOBLOCK` + drop-with-warning). The v0521 harness dropped all five
Stage-1 mounts (correctly for the four expert8 files, which would downgrade v0521 sources;
the fifth was the fix for exactly this). v0519 and v0521 have the same blocking call.

**Hypothesis.** Blocking PUSH send with no peer stalls the prefill scheduler.

**Test (failing first).** `harness/mounts-v0521/test_noblock.yihou.py`, CPU container on the
built image, 100 sends to an unbound PUSH (SNDHWM 4): stock v0521 file → `BLOCKED(>10s)`;
patched file → `returned` + one warning. Confirmed.

**Fix (single change).** `harness/mounts-v0521/src/kv_events_publisher.yihou-noblock.py` = v0521
file + `NOBLOCK`/`zmq.Again` drop with one warning (+ logger, + slot field). Mounted on both
legs via `mounts.yihou.v0521-noblock.tsv`. KV events kept ON (router kv-aware routing).
Round c44b. Verify: prefill keeps logging new batches after first health 200; warning ≥1.

**Side finding.** decode a1 died in the known same-node RCCL race (06:52, p2p.cc:256); retry a2
came up normally — unrelated.

## R001 result — fix verified at bring-up (c44b)

c44b prefill healthy 07:34:39 and served consecutive health generates (new 1-token batches,
two 200s, 0 "Health check failed"), unlike c44a. Decode a1 up first try 07:42:02 (no RCCL
race). Router 07:42:08 with exactly {39101, 39357}. Census both legs: 4× Found 0 HCAs, HIP
installed 4, HIP FAIL 0, rdma 0, EINVAL 0, hipIpc failed 0, OOM/TB 0/0. Smoke via router OK
(garbled text = simulated acceptance). Decode max_total_num_tokens 2,762,176 @0.95 (v0519
Stage 2: 2,773,376, −0.4 %); avail 10.88 GB. Sustained-load proof of the fix = the benchmark.

## R002 — agentx_env refuses: "server_info dp=1, command=4" (07:44)

Cause (first-hand): sglang v0.5.21 deprecates `--dp-size N --enable-dp-attention` and resolves
it to `attn_dp_size=N, dp_size=1` (`server_args.py:404-411`); live `/get_server_info` on both
legs: `dp_size 1, attn_dp_size 4, enable_dp_attention True`. Topology is as intended; the
validator (and run.c44 audit) compared the old field.
Fix: `bench-harness/tools/agentx_env.py` compares `attn_dp_size` when the command has
`--enable-dp-attention` and the server reports that field; older servers unchanged. Re-run →
`agentx_env: PASS`. Benchmark AgentX full C44 started ~07:46 (OUT_DIR runs/c44b/agentx-c44-r2).

## R003 — c44b prefill DP3 aborted in triton compile at 08:02:25 (warmup 293/488)

**Symptom.** Warmup stalled at returned=293/488 (08:02→08:12, errors 0). Prefill: no batch
after 08:02:25 on any rank; ranks 0-2 spinning (GPU 100 %) in collectives / KV send; DP3 PID
`<defunct>`, GPU3 0 %. Decode idle in collectives.
**Evidence.** prefill-0.log:5793 `scheduler_DP3_TP3: …llvm/ADT/Sequence.h:275 iota_range …
Assertion 'Begin <= End' failed` → `Fatal Python error: Aborted`. Crashing stack: triton
`make_amdgcn` ← `compile` ← autotuner ← aiter `batched_gemm_a8w8_a_per_token_group_prequant_
w_per_batched_tensor_quant` :103 ← sglang `rocm_absorb_q_bmm` (forward_mla_rocm.py:228) ←
`forward_absorb_rocm_prepare` :715, eager EXTEND. DP3's last logged batch: 2 seqs, 2174 new
tokens, 210,176 cached. (R001 fix holds under load: NOBLOCK warning ×4, no metrics stall.)
**Differential vs. v0519 (known-good, C40-full 1 h on this kernel path).** Identical: triton
build (3.7.0+amd.rocm7.2.0.git89002410), aiter kernel + wrapper (byte-identical), gfx950
config JSONs (identical), MHA/MLA dispatch `set_dsa_prefill_impl` and `handle_attention_dsa`
(identical). DIFFERENT: v0521 call sites (`:228`, `:298`) pass `config=_absorb_bmm_config(…)`
— a hand-built tile config for m ≤ 256 (adds waves_per_eu, matrix_instr_nonkdim 16,
cache_modifier .cg); v0519 passed no config (aiter default via `get_gemm_config`).
**Hypothesis.** The hand-built config for some m ≤ 256 drives triton's LLVM into the
assertion; aiter's default config does not.
**Test.** `build/repro-absorb-bmm.yihou/` — compile+run the kernel at GLM-5.2 dims
(H 64, K 192, N 512) over M ∈ {1…256, 2174}, both `v0521` (hand config) and `default`
(config None), one process per case, cold triton cache, GPU 7. (running 08:15→)
**Candidate fix (prepared, NOT deployed until the test discriminates).**
`mounts-v0521/src/forward_mla_rocm.yihou-noabsorbcfg.py` — `_absorb_bmm_config` returns None
→ both call sites behave as v0519.

### R003 update 08:38 — hypothesis weakened, repro blocked

- DP3's last logged batch before the abort was 2 seqs / **2174** new tokens → for M=2174
  `_absorb_bmm_config` returns None, i.e. the SAME call as v0519. Full-file diff of
  forward_mla_rocm.py: the only change on this kernel's paths is the `config=` argument
  (q_nope construction identical). So if the crashing call was M=2174, the hand-config
  hypothesis is wrong. OPEN alternatives: (a) crashing call was another M ≤ 256; (b) the
  assertion is not a pure function of (shape, config) — e.g. concurrent compilation (an
  inductor compile-worker thread was alive in the fault dump); (c) v0519 never met that key.
- Repro attempt 1 (GPU 7, no warm AITER cache): stuck; container unkillable ("did not receive
  an exit event"). Attempt 2 (GPU 3, warm cache, strided q_nope layout like the model):
  py-spy shows it blocked in aiter import → `rocminfo` subprocess. Host `rocminfo` also hangs;
  `rocm-smi` (sysfs) works; VRAM frozen at 146/273/269/9/147/147/147/2 GiB since ~08:22,
  no KFD PIDs. = driver still tearing down c44b (cf. 14 min drain c44a→c44b). GPU work
  blocked until it completes.
- The prepared candidate fix (hand config → None) is NOT justified by current evidence;
  not deployed.

### R003 resolved 08:41 — root cause reproduced; fix deployed in c44c

Driver teardown finished ~08:39 (all GPUs 283 MiB). Repro sweep 2 (GPU 3, warm AITER cache,
strided q_nope as in the model, cold triton cache, one process per case):

| M | default (config None = v0519 call) | v0521 hand config |
|---|---|---|
| 2174 / 1916 / 1351 | OK | OK (cfg None, >256) |
| 2, 16, 64, 128 | OK | OK |
| **256** | OK | **rc 134: `Sequence.h:275 iota_range … Begin <= End` — the c44b assertion** |

Hand config at M=256 = BLOCK_M 64, BLOCK_N 128, num_warps 4, num_stages 2, **waves_per_eu 4**
(the only tested point > 2; likely trigger — not isolated further). So the c44b crash call was
an M ≤ 256 extend, not the logged 2174-token batch (alternative (a) above).
**Fix:** `forward_mla_rocm.yihou-noabsorbcfg.py` (`_absorb_bmm_config` → None; both call sites
= v0519 call, which passes at every tested M). Mounted on both legs with the R001 file.
Round c44c launched 08:42. Upstream follow-up candidate: sglang `_absorb_bmm_config` emits a
config triton/LLVM on gfx950 aborts on (report with the one-case repro `one.yihou.py 256 v0521`).

## R004 — c44c decode GPU memory access fault 6 min into profiling (09:21:20)

**Before it:** R001+R003 fixes verified live in both containers; warmup 488/488, 0 errors
(09:15:34); profiling healthy 6 min (decode gen 201-324 tok/s/rank, accept 3.51-3.66).
**Symptom (watchdog first, leader confirmed).** decode-0.a3.log:222056 `Memory access fault by
GPU node-9 … address 0x772e5ca00000. Reason: Unknown.` → `Fatal Python error: Aborted`.
node-9 = HIP device 7 (KFD node = HIP+2 on this node class, per the 20260919 packup) =
physical GPU7 = decode DP3; GPU7 VRAM dropped to 147 GiB. DP3's last batch: token usage 0.69,
prealloc 0.64 (not exhausted).
**Host stack when the async fault surfaced:** `fused_dsa_draft_extend_metadata`
(kernels/ops/attention/dsa_metadata.py:690) ← dsa_backend `_apply_cuda_graph_metadata`
:1873 ← `init_forward_metadata_out_graph` :981 ← `eagle_draft_extend_cuda_graph_runner.execute`
:659 ← eagle_worker_v2 `_draft_extend_for_decode` :1145. (Async: where the host was, not
necessarily the faulting kernel.)
**Prior evidence.** `glm52.dsa-indexer-gpu-fault.packup_20260919`: same signature (decode, MAF
"Reason: Unknown", under sustained AgentX). P8D8 sweep packup: index_share ON faulted
~1h26m, OFF 13h33m fault-free. The triton/idxshare C40 packup (source of index_share=true)
itself states "IndexShare stability was NOT tested".
**Decision (one variable).** `index_share_for_mtp_iteration=false` (config.sh's measured
default; the stable choice per the user's "stable optimizations"). Everything else as c44c.
Round c44d launched ~09:31. Not proven causal — c44d is the test: full 3600 s with no MAF
would be consistent; a MAF would refute it.
**Allocation.** `scontrol update TimeLimit=1-03:00:00` on job 200276 accepted → ends
2026-10-09 14:17:38 UTC.

## R005 — c44d prefill GPU memory access fault 12 min into profiling (10:25:09)

c44d = c44c + index_share false (verified live). Warmup 488/488, 0 errors; profiling healthy
10:13→10:25 (prefix hit 92-93 %, in 214-256k tok/s, out 767-864 tok/s, decode accept
3.55-3.68). Decode survived (0 MAF) — past c44c's fault point.
**Fault.** prefill-0.log `Memory access fault by GPU node-3 … 0x7c7812a08000. Reason: Unknown`
→ `Fatal Python error: Aborted`. node-3 = HIP dev 1 = prefill GPU1 = DP1. DP1's last batches:
chunk 1 of a long prompt (8192 new, 123,328 cached) at 10:25:08.648, chunk 2 (4277) at
10:25:09.206, fault 10:25:09.214. Aborting process: main thread in `process_batch_result_idle`
→ `synchronize`; one mooncake transfer worker mid-copy (`send_kvcache` → `_transfer_data` →
`batch_transfer_sync`, HIP IPC). Async fault → kernel NOT attributable from logs. Decode: no
error, 0 transfer failures. Stall since 10:25 (server counters frozen). Stopped 10:31.
**Discarded lead:** "untuned bf16 GEMM shape" line adjacent to both faults — background noise
(3,240 such lines vs 3,384 batches in this log).
**Upstream:** sglang #37648 (OPEN, 2026-09-02): GLM-5.3-MXFP4 TP8 + EAGLE on MI355X, same KFD
fault after ~24 min of high-concurrency long-context agent load; no PD, no HiCache, tilelang;
no fix. #23784 (closed, no recorded fix): EAGLE3 high-concurrency MAF on AMD.
**State of knowledge.** 2 faults / 2 runs (6 min, 12 min into profiling), different legs,
different host stacks; prefill has no EAGLE. Cause(s) OPEN. Not attributable without kernel-
level diagnostics (serialized launches / GPU coredump), which invalidate benchmark timing.

## R006 — c44e (identical rerun) ended by the allocation, not by a fault (2026-10-09 11:17:38)

c44e: VRAM drain 10:33→10:57; prefill ok 11:02:36; decode a1 ok 11:08:28; router 11:08:34;
bench started ~11:09. Warmup reached sent=488 / completed=487 with 0 MAF / 0 abort on both legs;
both server logs END at 11:17:38 = job 200276 end (sacct: COMPLETED 2026-10-09T11:18:48).
**Leader error:** the `scontrol update TimeLimit=1-03:00:00` was accepted, but the batch script
itself is `sleep 86400`, so the job COMPLETED at its 24 h mark regardless of the raised limit.
Raising TimeLimit does not extend a hold job whose script exits. c44e therefore yields no fault
sample beyond "no MAF through ~8 min of warmup".
**Allocation now (2026-10-10 02:33 UTC):** node 102 allocated to another job; our 202546
(`-w crsuse2-m2m-102`, 24 h) PENDING (Resources), scheduler estimate start 2026-10-10T12:13:48.
Session was killed by an admin in between; teammates and timers are gone.

## R007 — phase 2 (2026-10-10): research workflow → H1 confirmed at code level

User directive: relax first, then step back; nodes 136-138 (all taken by others at 03:10;
135 busy; 139-141 no dockerd) → amd-oai job 205476 PENDING (QOSGrpNodeLimit), 202546 PENDING.
Research workflow wf_63fbfafe-63a (5 angles + synthesis + critic; JSON in agentx44 logs/).
**H1 (leader verified first-hand):** `pool_host/common.py:140-200 _cuda_host_register` splits
host pools > `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB` (default 256 GiB, environ.py:770) into
separate cudaHostRegister/hipHostRegister calls; JIT kernels translate only the tensor base
(`kernels/jit/include/sgl_kernel/runtime.cuh:62-76` → `hipHostGetDevicePointer(data_ptr)`), then
index linearly. At 450 GB (≈419 GiB, 10,016,064 tok/rank) slots ≥ 6,118,144 lie in chunk 1.
Same code in v0519 (environ.py:747) → latent; never hit there (ratio 1.5 ≈154 GB or pool never
filled). Explains c44d prefill MAF after ~30 min of write-through. Stage-1 harness had tried to
set CHUNK_GB=512 via a generic EXTRA_ENV that engine.sh never read.
**H2 strengthened (critic, first-hand):** c44c decode logged ~5,900 eager NCCL collectives/rank
in 6 min (index_share=true → cuda_graph_compatible=False min-reduced over DP); c44d/c44e: 0.
Plan v2 (PLAN.md): E0 micro-repro of H1 (no model) + E1 relaxed baseline ∥ E2 fixed target.
Team: harness-v2 (E1/E2 configs+driver), repro (E0 + rocm-debug-agent smoke). Sub-workspace
`/shared_nfs/yihou_glm52/v0521dbg_20261010/`; CLAUDE.md backed up and rewritten.

## R008 — E0: H1 mechanism CONFIRMED experimentally (138 GPU 7, 2026-10-10 ~03:45)

`diag/h1_chunked_register.yihou.py` (repro), sglang's own `alloc_mmap` + `_cuda_host_register`,
S=24 GiB, chunk 8 GiB (gran 2,875,392 B) → 4 registrations. hipHostGetDevicePointer(base+off) −
(dev(base)+off): chunk0 0; chunk1 −17,178,730,496; chunk2 −34,357,460,992; chunk3 −42,950,451,200
(later aliases placed BELOW earlier ones; host VA ≠ device VA). Reads: chunk0 ok; per-chunk-
translated chunk1 ok; **dev(base)+off (what the JIT kernels do) on chunk1/2 → "Memory access
fault by GPU node-9 … Reason: Unknown" (rc −6)** = production signature. Control (one
registration, same offsets): all ok. Logs `diag/out.e0.yihou/20261010-034046/`.
⇒ c44d prefill MAF explained by H1 (pool > 256 GiB at --hicache-size 450). OPEN: does a single
~419 GiB hipHostRegister succeed (the fix CHUNK_GB=512)? → test next (S=420 GiB).
Infra: /shared_nfs is mounted READ-ONLY on 138 (engine runs there would need node-local caches).

## R009 — ROCm debug agent validated (138 GPU 7)

Image ships rocm-debug-agent 2.1.0.70200 + rocm-dbgapi 0.77.4.70200. Deliberate store to 0x1000
from triton `yihou_oob_store_kernel`: without agent → only "Memory access fault … Reason: Unknown"
(production text); with agent (eager AND CUDA-graph replay) → "wave_1: pc=… (kernel_code_entry=…
<yihou_oob_store_kernel>) (stopped, reason: MEMORY_VIOLATION)" + disassembly. Overhead lean
(HSA_TOOLS_LIB + HSA_ENABLE_DEBUG=1 + --output + --disable-linux-signals): matmul 1.00×, tiny
launches 0.95-1.02× (noise), graph replay 1.00×; full (+--precise-memory --all) 4.3× → not for
long runs. Untested: multi-GPU RCCL/HIP-IPC/mooncake interaction. Logs diag/out.dbg.yihou/.
Decision: lean agent ON in E1/E2 (both legs).

## R010 — H1 at production size + fix validated (138 GPU 7, ~03:46-03:58)

S=420 GiB (450,970,730,496 B ≈ one prefill rank's 450 GB pool), gran 2,875,392 B:
- **default CHUNK_GB=256 → 2 registrations**, boundary at off 274,875,924,480 B (255.998 GiB =
  floor(256 GiB/gran)·gran); chunk-1 device alias ≈ −420 GiB below base. Read at bnd−4096 ok;
  **read at bnd+4096 → "Memory access fault by GPU node-9 … Reason: Unknown", rc 134** (+ "GPU
  coredump: handler exited with error" — same as production c44d).
- **CHUNK_GB=512 → 1 registration**: hipHostRegister OK (50.9 s; MAP_POPULATE alloc 78.8 s);
  reads via dev(base)+off at 0 / 200 / 300 / 419.999 GiB all correct.
Logs diag/out.e2.yihou/20261010-034633.s420.c512/ and …-035214.s420.c256/.
⇒ FAULT 2 root cause established (H1); fix = prefill env SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512.
Caveat: 4 concurrent ~419 GiB registrations + DSA indexer pools not tested in isolation (node
has 2.7 TB; production prefill pins ≈2.2 TB) — E2 tests that for real.

## R011 — v0519 sanity on 136: decode RCCL init fails 3/3 (04:10-04:15)

Prefill healthy 04:09:04 (pools identical to 102). Decode a1/a2/a3 all die ~8 s after
`nccl==2.27.7`: `p2p.cc:256 NCCL WARN hipIpcGetMemHandle failed : invalid argument` → `HIP failure:
'invalid argument'` → `NCCL error: unhandled cuda error` (= Stage-1 N1 signature). Unlike 102
(first retry always succeeded), every attempt here started with prefill healthy and idle — no
concurrent communicator init — so the "init race" reading does not fit. OPEN.
Next (multi-hypothesis, minutes, prefill kept up): RCCL-only 4-proc probe on GPUs 4-7 with the
decode container's docker flags: T1 default, T2 HSA_ENABLE_IPC_MODE_LEGACY=1 (ROCr legacy IPC
vs dma-buf; used by upstream #37648's reporter), T3 NCCL_P2P_DISABLE=1; then prefill-down
control if T1 fails.
R011 probe (prefill UP, decode docker flags, GPUs 4-7, 4-rank init+64 MiB all_reduce): T1 default PASS
(367× P2P/IPC, 0 hipIpcGetMemHandle failures), T2 IPC_MODE_LEGACY PASS, T3 P2P_DISABLE PASS (SHM),
T1b PASS. ⇒ plain RCCL IPC works; failure is specific to sglang decode init (pynccl ncclCommInitRank
in init_model_parallel_group, parallel_state.py:462 / pynccl.py:122, DP0 TP0). Next: a4 default
(control) → a5 +HSA_ENABLE_IPC_MODE_LEGACY=1 → (if both fail) T5 probe with decode's full env +
RCCL INFO preamble diff vs r013 on 102.
R011 cont.: a4 (default env, control) FAIL 04:29, same p2p.cc:256 signature → 4/4 default-env
failures on 136. a5 (+HSA_ENABLE_IPC_MODE_LEGACY=1, confirmed in container env) passed pynccl
ncclCommInitRank, loaded 282 shards by 04:34. n=1 vs 4/4 — suggestive, not proof; mechanism OPEN
(plain RCCL probe passes without legacy, so the trigger is in sglang decode init). Watch: mooncake
HIP-IPC KV transfer + census under legacy mode.
R011 cont.: a5 (legacy only on decode) healthy 04:40:52, router up, but e2e FAIL: prefill mooncake
`hip_transport.cpp:70 hipIpcOpenMemHandle failed` (err 201 ×2, 17 ×3) → `Failed to send kv chunk` ×5;
decode `Failed to get kvcache` ×66. Census clean (0 HCAs ×4, HIP ×4, rdma 0, EINVAL 0, MAF 0).
⇒ IPC mode must match on both legs (hypothesis). Next: full rerun with HSA_ENABLE_IPC_MODE_LEGACY=1
on BOTH legs (round sanity-v0519-legacy). Parallel research (repro): why default dma-buf IPC mode
fails in sglang decode init on 136 but not 102.
R011 probes 2 (prefill UP, GPUs 4-7): T5 torch nccl with dead a1's FULL env (67 -e) + tmpfs/binds PASS;
T6 sglang PyNcclCommunicator (gloo + ctypes ncclCommInitRank, the failing call) PASS; T6f +full env
PASS; T7 full env + MooncakeTransferEngine (0 HCAs, HIP ×4) before pynccl + all_reduce PASS.
RCCL INFO preamble 136-a1 vs 102-r013: identical (RCCL 2.27.7 0d2c4fd, HIP 7.2.26015, ROCr 1.18,
"Dmabuf feature disabled", same topology/48 channels, no cuMem); 136 dies at first P2P/IPC channel
setup on rank 0. Env bisection ruled out. Remaining diffs vs real decode: torch.distributed world
init, 5 bind-mounted sources, aiter imports, model memory state. Next: decode-solo (prefill DOWN,
default env), then full rerun legacy IPC both legs.

## R012 — ROOT CAUSE (candidate, high confidence) of the same-node "RCCL race" (N1)

repro, first-hand ROCr source (rocm-systems rocr-runtime runtime.cpp:1335-1482 Runtime::IPCCreate,
rocm-7.2.0): default dma-buf IPC creates a per-process fd-broker bound to ABSTRACT unix socket
"\0hsa<getpid()>" (:1452); bind failure → `return HSA_STATUS_ERROR` (:1466-1469) → HIP
hipIpcGetMemHandle "invalid argument". Abstract sockets are per NETWORK namespace (engines run
--network host), PIDs per PID namespace (private per container) → a decode rank with the same
in-container PID as a prefill rank cannot export. Legacy mode (HSA_ENABLE_IPC_MODE_LEGACY=1, flag.h:244)
uses KFD hsaKmtShareMemory, no socket → immune; handle formats differ → both legs must match.
Still "xhsa%i"+pid on ROCr develop (:1661/1702).
Evidence: 136 /proc/net/unix LISTEN @hsa688/833/982/1135 = prefill DP0-3 NSpids; decode rank0 = pid 688
in all 4 failures (only rank0 failed). 102 r012: prefill 988/1133/1282/1440, decode a1 rank0 = 988
→ FAILED (same WARN); a2 (688/827/976/1134) no overlap → passed. r013: no overlap → passed. Torch
probes passed (small non-colliding PIDs). Upstream (second-hand): rocm-systems#10609 (ex rccl#2188)
same signature, closed "Solved with pid fixes"; sglang/vllm rocm Dockerfiles set
HSA_ENABLE_IPC_MODE_LEGACY=1.
⇒ "RCCL same-node init race" (Stage-1 N1, c44a/c44c retries) is a PID collision, not timing.
Fix options: --pid host on both engines (unique PIDs; keeps dma-buf semantics) or legacy IPC on both
legs. Tests in flight (sanity, 136): decode-solo default (predict PASS), full rerun with --pid host
(predict decode init first-try PASS + KV OK), 2-MiB socket-squat test (predict rc 0 / invalid arg /
legacy rc 0).
R012 cont. (repro, read-only ROCr 7.2.0 + mooncake faae8dd4): (1) silent wrong-dma-buf in default
mode needs name loss + same-pid re-bind + bit-identical exported VA (ASLR) → negligible; collisions
fail loudly (export bind error / importer fd -1). Exception: mooncake FABRIC mode (MC_USE_HIP_IPC=0,
pidfd_getfd across PID namespaces) — real hazard, not our config. (2) --pid host: fixes ROCr @hsa<pid>
and mooncake pid-keyed shm names for OUR containers only; other tenants' net=host/private-pid
containers (e.g. dev_xb on 136) could still collide (low odds); ops hazard: in-container
kill/pkill see host processes (harness uses none); tini not PID 1 (harmless). Legacy IPC on both legs
is the PID-independent alternative (upstream sglang/vllm rocm images set it); its open question is
mooncake HipTransport KV transfer with legacy handles → test as a second arm later.
R012 test 1 (prediction confirmed): decode-SOLO on 136, DEFAULT env (no legacy), prefill DOWN → a1 PASS
first try (/health 05:01:36; 8× RCCL Init COMPLETE; 0 hipIpcGetMemHandle failures; 2,609,280 tokens
= 102). Tally default env: prefill up 0/4, prefill down 1/1.
Side observation (sanity): leftover 3,739 MiB/GPU on 4-7 from dead decode attempts drained ONLY while
the prefill container shut down (04:49-04:55) → another process (prefill) held references to that
memory. HYPOTHESIS for Stage-1 N3 "slow VRAM release": cross-container IPC references (imports or
broker-held dma-bufs) keep a dead peer's VRAM alive until the holder exits. Not established.
Next: full rerun with --pid host on both engines, default IPC (round sanity-v0519-pidhost).

## R013 — FIX VALIDATED end-to-end: legacy IPC on both legs → v0519 sanity PASS on 136 (05:16:52)

Round sanity-v0519-legacy (HSA_ENABLE_IPC_MODE_LEGACY=1 on BOTH legs, verified in env): decode /health
on FIRST attempt with prefill up (05:15:27); census both legs 0 HCAs ×4, HIP ×4, rdma/tcp 0, EINVAL 0;
PD split prefill 30 Prefill/0 Decode, decode 28 Decode/0 Prefill; kv_send/get fail 0, MAF 0, OOM 0,
Traceback 0; e2e " 2, 3, 5, 7, 11. A prime number is…" (= 102); intrinsic coherence 6/6; burst 64/64
(17.4 s, 1,879 tok/s; not a controlled comparison vs 102's 618). Capacity identical to 102.
⇒ mooncake HipTransport works with legacy handles; legacy-both-legs is the validated fix for the
PID-collision IPC failure (R012). Decision: E1/E2 default IPC_MODE_LEGACY=1, ENGINE_PID_HOST=0;
burst-node sanity = original recipe + legacy both legs; --pid host remains an evidence arm on 136.

## R014 — PID-collision mechanism CONFIRMED in isolation (136 GPU 7)

repro's `diag/ipc_sockname.yihou.py` (2 MiB hipMalloc + hipIpcGetMemHandle, --network none):
control rc=0; prebind (squat "\0hsa<own pid>" first) rc=1 "invalid argument"; prebind +
HSA_ENABLE_IPC_MODE_LEGACY=1 rc=0 — all three exactly as predicted (sanity/logs/ipcsock.136.yihou.txt).
Root cause of the same-node "RCCL init race" (Stage-1 N1, c44a/c44c retries, 136 0/4) is ESTABLISHED:
ROCr default dma-buf IPC abstract-socket name collision across containers sharing the host netns
with private PID namespaces. Fix validated end-to-end (R013): legacy IPC on both legs.
Burst nodes 227/067 preflight PASS (8 idle GPUs, NFS rw); v0519 image loads running.

## R015 — E2b decode fails deterministically: fixed port inside the ephemeral range (102, 05:39-05:43)

decode a1 (05:39:39) and a2 (05:41:40): `PortArgs.init_new` (server_args.py:854) → `scheduler_input_port at
39595 is not available in 30 seconds` (dist_init 39590, port_base 39591, scheduler_input = port_base+4).
Inside the decode container `ss -ltn` shows 39595 LISTEN with NO pid (not in decode's PID ns); in the
prefill container it is pid 684 = `sglang::scheduler_DP0_TP0`, which holds ~20 OS-assigned ports
(15887, 19979, 25428, 34139 … 39517, 39595, 39647, 59167 …: mooncake TE / zmq / torch). Node
`ip_local_port_range` = 32768-60999 (reserved 30000-32767). Our fixed ports (engine 39101/39357 + derived
blocks, router 38100, bootstrap 38198, kv-event 35657, snapshot 38901) lie INSIDE the ephemeral range, so
the second-starting leg (decode) can collide with the first leg's ephemeral sockets; deterministic for
the life of that prefill. Phase 1 on 102 was lucky; reference packups on 137 used 29xxx (below 32768).
Fix: all fixed ports < 30000. E2b: decode-only port override (prefill keeps its ports — it binds first).
Same-class lesson as R012: two containers on one host share the network namespace.

## R016 — E2b (102, v0521, legacy IPC both legs + debug agent): KV transfer fails hipIpcOpenMemHandle 201

R015 fixed via ENGINE_PORT_STRIDE=22000 (decode 61101, derived block 61334-61344 above the ephemeral
range); a4 decode up 05:51:09 (driver's "exited" was a false negative from an empty docker inspect over
spur exec — fixed). Census clean both legs (0 HCAs ×4, HIP ×4, rdma 0, EINVAL 0, hipIpc 0 at bring-up),
prefill 4×450 GB single-registration pools (CHUNK_GB=512), decode 2,773,376 tokens, 0 eager NCCL lines,
debug agent silent, 0 @hsa sockets (legacy). AUDIT PASS. SMOKE FAIL 05:53:16: prefill
`hip_transport.cpp:70 HipTransport: hipIpcOpenMemHandle failed (201 invalid device context)` →
`Failed to send kv chunk`; decode `Failed to get kvcache`.
Diff vs 136 sanity-v0519-legacy (PASS): image v0519→v0521, debug agent OFF→ON (HSA_ENABLE_DEBUG=1 +
HSA_TOOLS_LIB), node 136→102. Phase-1 c44e (102, v0521, DEFAULT IPC, no agent) moved KV fine.
Hypotheses: Hdbg (debugger/HSA_ENABLE_DEBUG breaks legacy KFD IPC import), Himg (v0521 runtime + legacy).
Tests in parallel: (a) e2b2 on 102 = E2b with DEBUG_AGENT=0; (b) repro no-model cross-container IPC
matrix {default,legacy}×{agent off,on}×{v0521,v0519}×{importer other/same GPU} on 136;
(c) E1 on 136 with DEBUG_AGENT=0.
R016 test (b) (repro, 136): cross-container IPC matrix 16/16 PASS — {v0521,v0519} × {default,legacy} ×
{agent off,on} × importer {other GPU, same GPU}; whole-allocation hipMalloc 64 MiB, 1 visible GPU, main
thread; effects verified (@hsa socket only in default mode; agent lib mapped only when on). ⇒ neither
the debug agent nor the v0521 image breaks legacy IPC in the simple case. Remaining suspects: (a)
fragment/sub-allocation export (torch caching allocator; ROCr useFrag path runtime.cpp:1356-1362),
(b) multi-GPU importer opening from a mooncake worker thread without hipSetDevice ("invalid device
context"), (c) repeated/many opens. Extended test ordered; e2b2 (agent off, 102) and E1 (136, agent off)
proceed as production data points.
R016 test (b2) (repro, 136, v0521, agent off): exports of an interior slice (S), a caching-allocator
sub-allocation (T) and a ROCr fragment (F), imported with 4 visible GPUs from main/worker thread with and
without hipSetDevice, both IPC modes → hipIpcOpenMemHandle rc=0 everywhere (no 201/17). BUT byte 0 of an
opened interior-pointer mapping = the ALLOCATION BASE (data matches at opened + (ptr − base)); HIP opens
at base, CUDA-style, both modes. mooncake faae8dd4 IPC path: registerLocalMemory desc.addr = addr
(hip_transport.cpp:687-700), relocation dest − entry.addr + shm_addr (:799-806) → if a registered addr
is NOT an allocation base, writes land (addr − base) bytes low = silent corruption / OOB fault near the
end (fabric path uses hipMemGetAddressRange; NVLink IPC path same pattern as HIP). Device context IS set
before open (:457-466, :229-250). LATENT HAZARD — not yet tied to a symptom. Next: (c) multi-open of
interior pointers of one allocation; source diff of what sglang registers (bases vs views) v0519→v0521.
R016 test (c) + source (repro): multi-open of n=2/8 interior views + base of ONE allocation, both modes →
all rc=0, each handle maps the whole allocation at a distinct VA from its base; no 201/17. Registration
(conn.py _registerable_regions → batch_register) IDENTICAL v0519/v0521: per-layer standalone torch.zeros
for MLA KV (memory_pool.py:4671-4686) and DSA index-K (index_key_cache.py), shared-topk layers 0-row
(skipped), aux MetadataBuffers on CPU (TCP), HiCache host pools not registered. Base-ness is runtime-
dependent (caching allocator may carve a cached block) → add a REGBASE logging mount. 201 still
unexplained by IPC primitives; next: real mooncake TransferEngine pair test (includes setupP2PAccess).

## R017 — second fix also validated: default IPC + --pid host → v0519 sanity PASS on 136 (06:01:36)

Round sanity-v0519-pidhost (PidMode=host both engines, no legacy env): decode first try (06:00:27) with
prefill up; census clean; kv_send/get 0; MAF 0; PD split 30/29; e2e/coherence PASS; burst 64/64 (16.2 s).
PID census: 8 schedulers (host PIDs 2263443…2274590) each own exactly one @hsa<pid>, 0 ns-PID collisions.
Shutdown under --pid host identical to r013 (SIGTERM → SIGQUIT → kill_process_tree), no foreign effect.
Tally on 136 (v0519): default+private PID 0/4; default+--pid host 1/1; legacy both legs 1/1.
Decision: E1 (136, v0521) = default IPC + --pid host + agent off (v0521+default IPC proven to move KV in
phase-1 c44e; v0521+legacy is the only combo with the unexplained 201). e2b2 (102) keeps legacy + agent
off to isolate the debug-agent question.
Observation (repro, first-hand, 136 ~06:15): new containers' python stuck in uninterruptible sleep
(STAT D, wchan flush_workqueue) inside HIP/KFD init for >10 min while rocm-smi shows the previous
(sanity, HiCache 4×553 GB pinned host pools) processes' VRAM still allocated with 0 % use and no KFD
PIDs → KFD teardown of the dead processes blocks new KFD opens. Mechanism of Stage-1 N3 "slow VRAM
release" = driver-side teardown (likely unpinning ~2.2 TB of registered host memory + device memory),
not a userspace leak; correlated with HiCache size. Not established beyond this observation.
REGBASE logging mount ready (repro): harness/mounts-v0521/src/conn.yihou-regbase.py (image conn.py +
per-region hipMemGetAddressRange log 'YIHOU-REGBASE … is_base='); added to e2b2 and E1.

## R018 — mooncake HipTransport interior-registration bug CONFIRMED end-to-end (silent KV misplacement)

Real mooncake TransferEngine pair test (repro, 136, v0521, agent off, engine-like flags/env, 0 HCAs, HIP
transport + setupP2PAccess ran): dst ('decode', HVD 4-7) registers 8 per-layer torch.zeros(64 MiB) +
1 INTERIOR region (64 MiB view at +4,198,400 of a 128 MiB tensor) via batch_register_memory; src
('prefill', HVD 0-3) transfer_sync_write per region. Both IPC modes: no 201/17; 8 per-layer regions
correct (first write ~0.5 s = handle open, then ~1.2 ms/64 MiB). INTERIOR region: rc=0 but data landed
at holder offset 0..64 MiB, i.e. (ptr − allocation base) bytes too low → silent overwrite of the
preceding 4 MiB, region tail never written. Cause (R016 b2): HIP opens interior handles at the
allocation base; mooncake relocates with desc.addr = registered ptr (hip_transport.cpp:700, 799-806).
Production impact depends on whether any registered region is interior → REGBASE logs (mounted in
e2b2/E1). 201 still unexplained (only E2b: legacy + debug agent ON in production).
Logs 136:/mnt/m2m_nobackup/yihou_v0521dbg/tepair/20261010-060426/.

## R019 — the misplacement bug is known upstream: Mooncake PR #4340 (open, approved, MI355X-tested)

repro (second-hand from GitHub, cross-checked against our first-hand R018): kvcache-ai/Mooncake#4340
"[TE] Fix HIP and NVLink IPC relocation for suballocated buffers" (salexspb, head e8062249; approved by
Lzy17, re-tested on MI355X/ROCm 7.2.0 on 10-08). Fix: export from hipMemGetAddressRange base, carry the
offset (64 B handle + uint64 = 72 B payload), add on relocation; offset-0 keeps 64 B (compatible); old
importer rejects 72 B → both peers need it. Stated trigger: "SGLang prefill/decode disaggregation on
MI355X. PyTorch placed DSA index-K buffers and a MoE weight-scale tensor in one allocation. KV transfers
overwrote the weights." — our exact stack → plausible contributor to decode-side corruption/faults
(c44c). Mooncake main bb6d1126 (10-10) still unfixed; faae8dd4 has the bug. Alternative #3548.
Also noted: #4488 / rocm-systems#12978 (CLR owner-device ordinal quirk in hipIpcOpenMemHandle) —
ordinals in range in our 4+4 layout, not shown to cause the 201.
Actions: builder builds infera-sglang:v0521-yihou-1010-mc4340 (faae8dd4 + #4340) on 102 and 136;
validate with diag/run_te_pair.yihou.sh (interior region must be correct); switch E-runs to it.
R019 build (repro): infera-sglang:v0521-yihou-1010-mc4340 built on 102 (9d755dd6…) and 136 (9872bbda…);
loaded engine.so sha256 3c96489a… identical on both (was f22b1364… in 1009). Full #4340 diff drifts on
faae8dd4 (nvlink/transfer_engine_impl/tests) → faae8dd4 port: hip_transport.{h,cpp} hunks verbatim +
hand-rebased transfer_engine_impl unregister hunks; NVLink/MUSA/tests dropped (not compiled under
USE_HIP). Recipe on clone branch: cdba2446 + 68f06e29 (Dockerfile.sglang.v0521-mc4340.yihou,
build_mooncake_sglang.mc4340.yihou.sh with git apply --check + source/artifact assertions,
patches/mooncake/mooncake-pr4340-e8062249.faae8dd4-port.yihou.diff). verify_v0521 + verify_mc4340
fail=0. GPU te_pair validation pending free GPUs. Both legs must run the same build (72 B handle).

## R020 — v0.5.21 + legacy IPC (both legs) fails KV import with 201; debug agent EXCLUDED (06:46-06:50)

First-hand, same signature on two nodes, agent OFF, private PID, legacy both legs, ports < 30000:
- e1a @136 (E1 relaxed, image a3a45025): smoke 06:46:16 FAIL; prefill
  `hip_transport.cpp:70 hipIpcOpenMemHandle failed (Error code: 201 - invalid device context)`
  → router `Decode transfer failed for request rank=0`.
- e2b2 @102 (E2b): same 201 at 06:49:49.
Census both: 4× Found 0 HCAs, HIP installed ×4, hipIpc failed 0 at bring-up, no MAF.
(e2b2's census STOP on "Traceback 0/4" was health_generate TimeoutError aborts during bring-up,
not faults; gate fixed by harness-v2.)
Matrix: v0519+legacy PASS (136) | v0521+legacy FAIL ×3 (agent on/off, 102/136) | v0521+default
IPC moved KV (c44e). The variable that tracks failure is the image under legacy IPC (same
mooncake engine.so f22b1364 in both) → suspect ROCr/HIP runtime of the v0.5.21 base. Mechanism
OPEN (te_pair/simple tests did not reproduce; repro's IPC matrix to localize).
Decision: bypass legacy; use the other validated collision fix (R017): `--pid host` + default IPC.
R020 update (repro, first-hand, read-only inspection): v0519 fd7220a5 / v0521 a3a45025 / mc4340
9872bbda share a byte-identical HIP/ROCr stack (libamdhip64 36fb0b13…, libhsa-runtime64 c534feb1…,
rocm-core 7.2.0-43), same torch; v0519 and v0521 share mooncake f22b1364; Env differs only in
AITER_COMMIT. → "ROCr/HIP runtime" suspicion REFUTED. Also v0521 + legacy real-TE pair
(cuda:0→cuda:0, 9 regions) PASSED in isolation. Remaining suspects: cross-ordinal import (ranks
1-3), registration count/content of sglang v0521, thread/device-context of the importing thread
(201 = invalid device context → import on a thread with no current device?).
R020 MECHANISM (repro; leader re-verified first-hand 06:5x): CLR rocm-7.2.0 hip_memory.cpp:3357
stores exporter `owners_process_id = getProcessId()` in the IPC handle; :3381-3382 hipIpcOpenMemHandle
returns hipErrorInvalidContext (201) if it equals the importer's PID ("same-process" guard). With
private PID namespaces, prefill DP0 and decode DP0 both got pid 973 in e1a (aiter pname lines,
re-grepped by leader) → valid handle rejected with 201. Same PID-collision class as R012 (default
mode collides at export socket "@hsa<pid>"; legacy collides at import guard). v0519 passed by PID
layout luck. FIX: `--pid host` (node-unique PIDs) covers both modes; legacy alone does NOT.
Next-wave choice (pid-host + default IPC) is therefore the correct fix, not just a bypass.
Pending: repro's PID-1/PID-1 tiny confirm (predict 201) vs --pid host (predict OK).
R020 CLOSED (repro first-hand, 136, 07:01, v0521-mc4340): two containers, exporter GPU4 → importer GPU0.
legacy/private (pid 1==1) → 201; legacy/host → rc=0 data OK; default/private (1==1) → 201;
default/host → rc=0 data OK. Logs 136:/mnt/m2m_nobackup/yihou_v0521dbg/pidguard/20261010-070134/.
te_pair gate (mc4340, PIDHOST=1, default IPC): 9/9 writes rc=0, 8 per-layer + INTERIOR region
correct (PR #4340 works end-to-end). GO.e1b created 07:01.

## R021 — sglang PR #42553 (HiCache host pools via hipHostMalloc/GTT, opt-in) — assessment (07:1x)

PR: open, REVIEW_REQUIRED, head 8c27ef48; files pool_host/common.py (+ test + manual bench).
Mechanism: on ROCm with `HSA_USERPTR_FOR_PAGED_MEM=0`, default-allocator pinned host pools come
from one exact-size `hipHostMalloc` (GTT) instead of mmap + chunked `hipHostRegister` (USERPTR).
GTT pages never migrate → no KFD eviction stall (author: induced 64 s stall → 0; GLM-5.2 AgentX C8
HiCache 180 GB: total tput +5.1 %, P90 interactivity +3.7 %, output tput −5.0 % — second-hand).
First-hand checks:
- `git apply --check` clean on the v0521 image tree (sglang 9f5d8712, 2026-10-08).
- Both our pools take this path: MLA KV (memory_pool_host.py:268) and DSA indexer
  (pool_host/dsa.py:250) via ALLOC_MEMORY_FUNCS → alloc_with_host_register, default allocator.
- Side effect for us: bypasses `_cuda_host_register` entirely → H1 (chunked registration) cannot occur.
- HARD LIMIT: GTT = 1,477,361,6xx,xxx B on 102 and 136 = exactly MemTotal/2 (TTM global page
  limit, node-wide, shared by all ranks). Target prefill needs 4 × (450 + 103) ≈ 2.21 TB → does NOT
  fit; the PR raises (no fallback). E1 (ratio 1.5: 4 × ~189 GB ≈ 0.76 TB) fits. Max per-rank pool
  (KV+indexer) ≈ 369 GB minus other pinned use → hicache-size ≲ 290-300 GB.
- Our nodes: kernel.numa_balancing=0 on 102 and 136, THP madvise/never → the main migration
  trigger is off here; benefit on our nodes is UNVERIFIED (compaction can still migrate).
Verdict: useful as an opt-in A/B arm for E1-class sizes, and as an alternative to 450+CHUNK512 at
≤ ~290 GB; NOT for the 450 GB target. Not injected into running e1b/e2b3 (one variable at a time).
R021 GPU validation (repro, 136 GPU7 07:14, 1009 image + PR mount; leader re-read logs
diag/pr42553/run.20261010-071426.yihou/): PR unit test 15 passed + 17 subtests (UT_EXIT=0).
150 GB page_first MLA pool via ALLOC_MEMORY_FUNCS, HSA_USERPTR_FOR_PAGED_MEM=0 → path=hipHostMalloc,
alloc 5.31 s (GTT 0.2→150.2 GB), AOT transfer_kv_all_layer_mla D2H + per-layer H2D on slots at
start/middle/END of pool: data OK; free 0.52 s. Untested: JIT hicache kernels, DSA indexer pool,
4-rank GTT sharing vs 1477 GB cap, stall/throughput claim. Arm G is ready for an A/B after E1 passes.

## R022 — `--pid host` is DENIED on spur nodes (102); fix the PID collision with a PID offset instead (07:2x)

First-hand (harness-v2, runs/e2b3/engine-prefill.log): `docker: Error response from daemon:
authorization denied by plugin spur-authz: spur-authz: denied [N1]: --pid=host`. 136 (plain ssh,
no spur-authz) allowed it. We do NOT try to circumvent the authz policy.
Alternative that keeps PID-namespace isolation: make the two legs' rank PIDs disjoint by
advancing the decode container's PID counter before exec (fork + exit K throwaway subshells, e.g.
K=30000; pid_max on 102 = 4,194,304). Guards both collision points (ROCr "@hsa<pid>" export
socket in the shared host netns; CLR owners_process_id import guard), in default IPC mode.
Validation plan: repro pidguard pair on 102 (exporter pid 1 vs importer offset) → expect rc=0;
then e1c on 102 with DECODE PID offset + census assertion "rank PIDs disjoint across legs".
R022 VALIDATED (repro, 102 GPU4→GPU0, mc4340, private PID, default IPC, 07:25): control pid 1==1 → 201; exporter offset K=30000 (loop 7.9 s, pid 30010) vs importer pid 1 → rc=0, data OK both ways. Caveat: exec keeps the shell PID, so only later-forked processes (sglang rank schedulers) get > K; census must confirm rank PIDs disjoint. GO.e1c created.
R018/R019 follow-up (e1c 07:36, first-hand via REGBASE mount): prefill 4 DP ranks each 'kv/aux/state summary n=166 interior=0' → no interior registrations on prefill (E1 config, fused indexer off). Decode side pending.

## R023 — e1c: FIRST v0.5.21 KV transfer since the 201 diagnosis; bench harness path bug (07:51)

e1c @102 (E1 relaxed, mc4340 + REGBASE, default IPC, private PID, DECODE_PID_OFFSET=30000):
prefill ok 07:36:42, decode first try 07:50:57, router 07:51:03, smoke OK 07:51:08, audit PASS.
Census (leader re-read): rank PIDs prefill [968 1112 1258 1413] / decode [30968 31107 31254 31407]
disjoint; 4× Found 0 HCAs both legs; OOM/Traceback 0/0; REGBASE interior=0 and 'extends past'=0 on
BOTH legs → R018 misplacement not triggered by this config's registrations. R020/R022 fix works.
Bench FAILED after 55 s (harness, not engine): client container cannot see
$WS/cache/InferenceX (symlink → agentx44 tree, not mounted) → benchmark_lib.sh not found.
Stack left up; harness-v2 fixing the mount and re-running the bench only.

## R024 — retro: which findings duplicated post-0920 packups (user question, 08:5x)

Checked first-hand by grep over post-2026-09-20 packups (pd/yihou 0920-0924, view p8d8 0920, oai 1009):
- DUPLICATE: H1 fix `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512` was already in the v0519 oai recipe
  (scripts/drivers/r012-1p1d-xgmi.run.sh:53 EXTRA_ENV; research/R1-bringup-recipe.yihou.md:123,130;
  s=450 pack-up arm R-A "hc450chunk512"). Phase-1 v0521 configs (agentx44 harness) did not carry it →
  c44d prefill MAF + E0 re-derivation. Only new part: the mechanism (base-pointer translation).
- KNOWN, reused: index_share=true MAF history; ENGINE_PORT_STRIDE (0922 pack-up); MC_TE_FILTERS/0 HCAs;
  "RCCL race → start gate / retry decode" as a workaround (0922 pack-up left root cause open).
- NEW: RCCL-race root cause = ROCr "@hsa<pid>" socket PID collision (R012); CLR same-PID 201 guard (R020);
  spur-authz denies --pid host → PID offset (R022); ephemeral-range port collision (R015); Mooncake
  interior-registration misplacement (R018/#4340); v0521-only R001 (blocking kv-events PUSH) and R003
  (absorb-bmm config) — R001/R003 novelty not re-verified against every pack-up.
Lesson: on a new base, diff the new launch env/args against the proven run's ACTUAL params (driver
EXTRA_ENV / params.txt), not only the README knob list.
R024 CORRECTION (repro diff harness/diff.e2b4-vs-v0519r013.yihou.txt; leader re-verified 09:1x):
r013's driver passed EXTRA_ENV=…CHUNK_GB=512, but the v0519 kvcap engine.sh reads only
PREFILL_EXTRA_ENV/DECODE_EXTRA_ENV (engine.sh:51,64); kvcap rounds/r013-1p1d-validate/engine-prefill.log
has 0 CHUNK_GB matches. So the proven oai run did NOT apply the knob: H1 was latent there too (short
validation never filled past chunk 0). H1 was therefore not a simple "dropped known fix": the knob
name existed in the recipe text (and the s=450 pack-up arm R-A), but no documented mechanism and the
oai run never exercised it. e2b4 carries it (verified). Other diff results: (c) MISSING = none;
decode mem 0.92 (r013) → 0.95 (target); r013 used REAL MTP acceptance whereas e2b4/e1c use
SIMULATE_ACC_LEN=3.61 (as the 0923 AgentX c40 pack-up). Stage-1 rounds r010-r014 never ran AgentX.

## R025 — FIRST v0.5.21 PASS: e1c AgentX full C40 (09:24)

runs/e1c/agentx-c40-r2/agentx_conc40.json, driver: `bench ok rc=0 AgentX passed`.
Config: E1 relaxed (hicache ratio 1.5 ≈154 GB/rank, decode mem 0.85, fused indexer OFF,
index_share false, CONC 40, 3600 s, warmup 10/lane), mc4340 image, default IPC, private PID +
DECODE_PID_OFFSET=30000, SIMULATE_ACC_LEN=3.61. Profiled 4174 ok, 2 InvalidInferenceResultError
(error-dropped), 5 cancelled at grace end; duration 3625.6 s. Total 168,389 tok/s (21,049/GPU),
input 167,176, output 1,213 tok/s. TTFT p50 2.85 / p90 11.6 / p95 17.6 s; ITL p50 13.2 ms;
interactivity p50 75.8 tok/s/user. GPU cache hit 0.944, CPU 0.014 (host pool 100 % full).
Server logs: MAF 0, 201 0, KV transfer fail 0, Traceback 0 on both legs.
Reference (0923 c40-full, v0519, 137, index_share ON): 168,260 tok/s, 4157 req, 1→4 errors.
→ v0.5.21 matches the v0519 reference at C40 under the relaxed config.
H1 FIX CONFIRMED IN SERVICE (e2b4, 10:20): server cpu_kv_usage 65.9% (> 61% = slot 6,118,144 of 10,016,064; if it is a mean over ranks, ≥1 rank is past the boundary) with prefill MAF 0, done=1256 ok=1256 err=0. c44d (no CHUNK) faulted at this regime. Keep watching to 100%.

## R026 — e2b4 PASS: target minus fused indexer, AgentX full C44 (11:09)

runs/e2b4/agentx-c44/agentx_conc44.json, rc=0. Config: hicache-size 450 + CHUNK_GB=512, decode mem
0.95, CONC 44, fused indexer OFF, index_share false, triton DSA, EAGLE 5/1/6 (sim acc 3.61), mc4340,
default IPC, DECODE_PID_OFFSET=30000. 4138 ok / 1 InvalidInferenceResultError; 162,774 tok/s total
(20,347/GPU), out 1,170; TTFT p50 5.41 / p95 30.5 s; ITL p50 14.0 ms. Host pool reached 100 %
(crossed the 6.12M-slot H1 boundary ~10:20) with prefill MAF 0 → H1 fix holds over the full pool
(one 60-min observation). Both legs: MAF/Traceback/KV-fail/hipIpc-fail 0.
vs e1c (C40, ratio 1.5, mem 0.85): total −3.3 %, TTFT p50 2.85→5.41 s (C44 + larger host pool; not split).
Warmup-slowness hypothesis "fused-indexer OFF slows prefill batches": weakened by harness-v2's DP0
per-batch data (logs/prefill-tput-window18.yihou.txt: per-batch p50 similar across c44d/e1c/e2b4;
e2b4 warmup had fewer DP0 batches and 0 cached tokens on DP0) — open: DP routing during warmup.
Next: e2c = E2 target (fused indexer ON) — tests H3.

## R027 — TARGET PASS: e2c, full target config, AgentX full C44 (13:17)

runs/e2c/agentx-c44/agentx_conc44.json, rc=0. Config = mission target on v0.5.21: triton DSA, prefill
hicache-size 450 (+CHUNK_GB=512), decode mem 0.95, EAGLE 5/1/6 (sim acc 3.61), fused DSA indexer ON
(v0521 default), index_share false, CONC 44, 3600 s, warmup 10/lane; mc4340 image, default IPC,
private PID + DECODE_PID_OFFSET=30000. 4179 ok / 2 InvalidInferenceResultError, 7 cancelled at grace end;
164,194 tok/s total (20,524/GPU), in 163,011, out 1,182; TTFT p50 4.63 / p90 19.95 / p95 28.65 s;
ITL p50 14.7 ms / p95 21.6 ms; interactivity p50 67.9; E2E p50 13.3 / p95 72.0 s. GPU hit 0.907, CPU
hit 0.040. Host pool crossed H1 boundary ~12:26 → 100 %. Both legs: MAF 0, Traceback 0, KV fail 0, 201 0.
H3 (fused indexer faults): NOT reproduced in 60 min at target (one run). vs e2b4 (fused OFF): +0.9 %
total tput, TTFT p50 5.41→4.63 s, ITL p50 14.0→14.7 ms (single samples; not significant claims).
Open: index_share=true (H2, E4) not retested on v0521; replicates; real-acceptance run; arm G.

## R028 — perf phase (user 15:5x): "run C40; v0519 best ~22k/s/chip; find why lower; goal > 22k"

Reference check (first-hand): best documented v0519 same-node number = 21,032 tok/s/chip (0923 packup,
C40, triton + index_share TRUE, node 137, hicache ratio 1.5, decode mem 0.85). No 22k figure found in
any yihou pack-up (grep). v0521 e1c (C40, index_share FALSE, ratio 1.5, mem 0.85, fused OFF) = 21,049
→ at C40 v0521 is NOT slower than v0519. The "drop" is e2b4/e2c at C44 (20,347 / 20,524).
server_args diff ref vs e2c (launch_command): only --hicache-size 450 (ref ratio 1.5), index_share
true→false, decode mem 0.85→0.95 (+ v0521 version, fused indexer default ON, CHUNK_GB env).
Run: C40 bench on the live e2c stack after /flush_cache on both legs (first attempt with warm caches
aborted at warmup to keep cold-start like-for-like), start 16:01:37 → runs/e2c/agentx-c40-e2cstack-flushed.
Separates CONC (40 vs 44) from config. Lever under research (repro): index_share TRUE (H2 MAF on v0521).

## R029 — index_share=true MAF on v0521: research (repro, read-only, v0521 tree)

First-hand: the H2 control chain exists (seedless IndexShare round → cuda_graph_compatible=False,
min-reduced over DP → eager DP-padded draft; pad_batch / _pad_inputs_to_size do not pad
spec_info.dsa_topk_indices: eagle_info.py:195-201, forward_batch_info.py:1955-1970). BUT the triton
decode consumer -1-pads topk to the query rows and asserts (dsa_backend.py:2483-2484, 3763-3784), so
the "kernel reads past the seed" OOB is REFUTED for our path. The host-stack kernel
fused_dsa_draft_extend_metadata looks bounded; the async fault likely came from an earlier kernel in
the iteration (inference). Stale-seed risk: the eager draft-extend seed buffer is -1-filled only on first
allocation (eagle_worker_v2.py:1036-1046) and publish() rewrites only selected rows
(index_topk_share.py:62-72) → stale but in-pool slots (inference: wrong data, not MAF).
Upstream: sglang #32209 (OPEN, not in v0521) "Fix PD decode hang with DP attention and GLM-5.2 MTP",
same chain (DP-padded eager draft, DSA shape mismatches on idle/padded ranks); 12 files.
Faulting kernel NOT identified. The v0519 reference also used simulated acceptance 3.61 (README).
Next (combined run): index_share=true + lean debug agent (names the kernel) + env-gated seed guard
(count/mask out-of-pool seed entries; -1-fill seed buffer on reuse) → MAF yes/no, kernel name, bad-seed
count, and the index_share throughput at C40, all in one run.
R029 cont.: patch C delivered by repro (eagle_disaggregation/eagle_worker_v2 .yihou-seedguard.py; unit test 31 checks PASS; CPU-only, integration = first live run). Round e2d prepared (env.e2d-102.yihou.sh, config.yihou.seedguard.sh): e2c + index_share TRUE + CONC 40 + guard=count + decode-only lean debug agent; tokdiff vs e2c verified. C40-on-e2c flushed run confirmed cold (first-turn #cached-token 0). e2d queued to harness-v2 after the C40 bench.

## R030 — target config at C40 (e2c stack, caches flushed, cold): 21,696 tok/s/chip — PASS (17:20)

runs/e2c/agentx-c40-e2cstack-flushed/agentx_conc40.json: 4237 ok / 3 err; total 173,570 tok/s
(**21,696/chip**), in 172,319 / out 1,251; TTFT p50 3.30 / p90 11.94 / p95 20.05 s; ITL p50 11.7 / p95
18.1 ms; interactivity p50 85.6; GPU hit 0.929, CPU hit 0.0165. MAF/TB 0.
vs v0519 ref (21,032/chip, index_share ON): **+3.2 %**; vs e1c (v0521 C40 relaxed, 21,049): +3.1 %;
vs e2c at C44 (20,524): +5.7 %. Input mean 147.6k tok/req vs 145.3k (ref) / 145.2k (e1c): workload
comparable (+1.6 %). → The "drop" was CONC 44 vs 40, not v0521: at C40 the target config beats
the v0519 best. Still 1.4 % short of the user's 22k. Next lever: index_share TRUE (e2d, launched
17:22 by the waiter, guard=count + decode debug agent).
R029 cont. (18:04): seedguard live-integrated — decode YIHOU-SEEDGUARD lines from all ranks (first 18:04:40: extend_seed.bad_entries=0 rows=3 pd_seedless_batches=3 draft_eager=1/3 iters).

## R031 — e2d: index_share TRUE @C40 = 21,383 tok/s/chip, PASS, 0 MAF (19:21)

runs/e2d/agentx-c40/agentx_conc40.json: 4198 ok / 3 err; total 171,065 (21,383/chip), in 169,842 / out
1,223; TTFT p50 3.53 / p95 21.96 s; TPOT p50 12.0 ms. vs same config index_share FALSE (R030):
173,570 (21,696) → −1.4 % (single runs; within plausible run-to-run noise, not measured). Decode had the
lean debug agent (measured ~1.00x earlier). MAF 0 both legs over 60 min, kernel_code_entry 0.
Seedguard (count): bad_entries 0 in ~1.24M checked seed rows; draft_eager ≈ 4.4 % of draft iterations
(3.65-3.74k / 84-88k per rank); pd_seedless 986-1360 per rank. → On v0521 at C40, index_share TRUE ran
60 min clean (c44c @C44 faulted at 6 min; not reproduced; cause of c44c still open) and gives NO
throughput gain. Lever rejected for the 22k goal.
Load profile of the C40 index_share-false run (logs 16:17-17:17): decode running-req mean 3.67/rank
(DP4 → ~15 decoding), #transfer-req>0 in 54.5 % of decode lines, prefill #queue-req>0 in 26 % of
batches, pending-token mean 145k; Little's law: rps 1.17 × E2E 18.1 s ≈ 21 in flight of 40 lanes
(AgentX inter-turn gaps). → Server-side, prefill/transfer (TTFT ≈ 5.6 s mean) is the critical path;
decode is lightly loaded (ITL 11.7 ms).

## R032 — TTFT decomposition of the 21,696 run (repro, metrics timeslices in the profiling window)

Per request (mean): prefill queue 1.10 s (21 %, tail-heavy, p50 5 ms) | bootstrap 0.02 | compute 2.28 s (43 %:
non-final chunks 0.64 + final forward 1.64; HiCache load-back negligible) | **KV transfer 1.91 s (36 %)**.
Sum 5.31 s ≈ aiperf TTFT 5.61. Transfer volume 7.78 GB/request (whole prompt KV; decode radix cache is
rejected with EAGLE in v0521: disagg.py:85-88) → effective 4.1 GB/s vs 58 GB/s measured for one
contiguous 64 MiB GPU0→GPU4 copy on the same mooncake HIP path (te_pair) → transfer runs at ~7-12 % of
link. Inference: host-bound per-block submission (one block per contiguous page run per layer buffer,
one batch_transfer_sync per request on one thread; HipTransport per-block attr/setDevice/event sync,
hip_transport.cpp:456-550). Prefill DP byte load balanced ±10 %.
Model (inference): closed loop, ~16.4 s client gap/cycle; ΔE2E → Δrps ≈ ΔT/34.5 s; transfer 1.9→0.7 s ≈ +3.5 %.
Levers: L1 transfer efficiency (instrument, then parallelize / thread pool), L2 chunked-prefill 65536,
L3 prefill queue HOL.

## R033 — transfer lever facts (repro, v0521 tree, first-hand) and plan for e2e (19:3x)

Sender: QUEUE_SIZE=4 queues, 1 transfer_worker thread each (conn.py:308-330); requests sharded by
destination port sum (conn.py:2919-2923); 1:1 prefill dpX ↔ decode rank ⇒ every request of a prefill rank
runs on ONE thread, all layers in ONE batch_transfer_sync (conn.py:1134-1137) — KV transfer fully serialized
per rank. THREAD_POOL_SIZE executor used only on the custom-mem-pool path (NVLink/Barex only, not ROCm).
No env-only parallelism. HipTransport code is concurrency-capable (mutexed stream/event pools, thread-local
slice cache) but >1 concurrent caller is untested here. Prefill: effective chunk = 8192 tokens/rank
(32768/DP4); avail 41.5 GB after KV pool; chunk 65536 not evaluated this round (DSA indexer temporaries risk).
Arithmetic (inference): 7.78 GB/req at ~42 KiB/block ≈ 1.8e5 blocks; ~10 µs/block ≈ 1.9 s = measured.
Plan e2e: target config, index_share false, C40, YIHOU_XFER_LOG=1, YIHOU_XFER_PAR=4 (layer groups as futures).
Gate: concurrent-caller te_pair test on GPU0/4 byte-exact (simulated acceptance cannot detect KV corruption).
e2d stack stopped ~19:33.

## R034 — concurrent-caller test (repro, 102 GPU0→4, mc4340, 20:12-20:14): byte-exact; k-way submission gives no gain

65,536 non-contiguous 41,984 B blocks (2.56 GiB): t1 warm 0.441 s (6.72 µs/block, 5.81 GiB/s); t4 layer
quarters 0.419-0.433 s; t4b interleaved 0.426-0.437 s. 164 KiB blocks: t1 0.108 s (23.7 GiB/s), t4 0.103-0.105.
All 13 phases byte-exact. → ~6.5 µs fixed per block, serialized process-wide (threads don't help); speed
scales with block size (6 → 24 → 58 GB/s at 41 KiB → 164 KiB → 64 MiB). Production: 7.78 GB/req ≈ 1.85e5
blocks × 6.6 µs ≈ 1.2 s per-op + ~0.7 s queueing behind the single per-rank worker ≈ measured 1.91 s.
Real lever: fewer, larger copies — sglang's staging gather/scatter path (conn.py:653-783), gated today to
heterogeneous TP (needs_staging, conn.py:625-636). Being scoped (repro).
e2e re-armed (20:16) with YIHOU_XFER_PAR=1 (no gain expected) and PREFILL_CHUNK_SIZE=65536 (L2) + XFER_LOG=1;
GO given 20:16 after byte-exact PASS. Tests: production wait/wall split; chunk-size effect.
R034 cont. — staging scope (repro, first-hand): v0521 staging is MHA-only (prefill.py:347-352 `not
is_mla_backend`; K/V head-slice kernels staging_buffer.py:32-68); DSA index_k + draft KV go via separate
per-block state path (conn.py:900-925, 1840-1870). Porting to MLA/DSA/MTP = new kernels + 2-leg protocol,
days, hot-path corruption risk → REJECTED. Alternatives: (A) read e2e YIHOU-XFER avg_block_KiB first;
(B) prefill-side single-launch peer-copy kernel into decode's IPC-mapped KV (same gain as staging, no
decode protocol change); (C) room-sharding (queueing only). Doing now: KV checksum mount (YIHOU_KV_CKSUM)
for transfer-correctness proof; B pre-scoped, go/no-go after e2e block stats.
R034 cont. — KV checksum: v0521 has it built-in: --disaggregation-enable-kv-checksum (both legs; disagg.py:72-77): Adler-32 over all kv buffers + DSA state of every request on prefill last chunk, verified on decode before use; mismatch → logger.error + abort (decode.py:3061-3115). ROCm wavefront-64 handled. Cost unmeasured (full 7.8 GB scan both legs, sync on scheduler threads) → use for a correctness run of any new transfer path, not the score run.
R034 cont. — option B pre-scope (repro): Python has no access to mooncake's peer mappings (remap_entries_ in C++). B (sglang-side IPC handles + triton kernel) ≈ 2-3 days, 2-leg protocol. B' (RECOMMENDED): inside mooncake HipTransport::submitTransfer, relocate all entries once and launch ONE batched peer-copy kernel per batch (gate MC_HIP_BATCH_KERNEL=1), same sync contract, prefill-only, image rebuild ~1 day. Go/no-go on e2e avg_block_KiB; microbench (kernel peer-store BW for 41 KiB items, CU contention, checksum cost) first.
R034 cont. — B' prep (repro, offline): CLR rocm-7.2.0 hipMemcpyBatchAsync = loop of ihipMemcpy (hip_memory.cpp:2805-2836) → no help; B' needs a real kernel. Microbench diag/bprime/peer_copy_bench.yihou.py (memcpy_loop / memcpy_batch / triton kernel g8-1024, stale-L2 + byte-exact; adler32 checksum cost). Draft mooncake patch diag/bprime/mooncake-hip-batch-kernel.on-faae8dd4-pr4340port.yihou.diff (+231/-1; gate MC_HIP_BATCH_KERNEL; git apply --check OK; not built). e2e prefill ok 20:24:40.

## R035 — first-hand production transfer stats (e2e YIHOU-XFER, warmup 20:32-20:38): blocks are LARGE; per-op model refuted

Per prefill shard: ~75 blocks/call (= 78 layer buffers, one contiguous run per layer per chunk),
avg_block ≈ 9 MiB (16384 tok × 576 B), 0.69 GiB/call, wall 2.4-6.0 s per call → 0.11-0.40 GiB/s;
wait_mean 1.4-4.1 s (queue behind the single per-rank worker). te_conc: 64 MiB copy = 58 GB/s, so
production is ~150-500× below link speed with large blocks → R034's "1.85e5 small blocks × 6.5 µs"
inference was WRONG for production (pages are contiguous per chunk); B' (batched copy kernel) is MOOT.
New hypothesis H-X1 (unverified): mooncake HIP copies serialize with prefill compute via stream
semantics — e.g. blocking (non-NonBlocking) pool streams implicitly sync with the legacy NULL stream that
torch/sglang prefill uses, so each copy waits for enqueued prefill forwards (seconds at 16k-token long-
context chunks) and vice versa. Checks: mooncake stream_pool stream creation flags; torch stream used by
prefill; a microbench with a concurrent NULL-stream compute load on the src GPU.
R035 cont. — H-X1 REFUTED at code level (repro): mooncake streams are hipStreamNonBlocking (stream_pool.cpp:74), hipMemcpyAsync on pool streams + per-block hipEventSynchronize, no NULL-stream/device sync; sglang forward runs on torch-created streams (scheduler.py:1112-1118). wait_event.synchronize (conn.py:2336-2345) is outside the timed wall (inflates wait_mean only). 0.1-0.4 GiB/s on 9 MiB blocks UNEXPLAINED; candidates: (a) blit-kernel copy starved behind long prefill kernels, (b) SDMA/XGMI contention incl. HiCache write-through H2D/D2H, (c) host-staged peer fallback. Microbench copy_interference.yihou.py ready (idle / NULL-stream GEMM / torch-stream GEMM / spin kernels × SDMA on/off); needs a GPU window after e2e.
R035 cont. (repro, e2e log correlation, approximate): first data call 20:32:38 DP1 0.69 GiB in 5.95 s with only ONE 16k-token prefill batch on that DP in the window (~76 ms per 9 MiB block); steady cold warmup 0.19-0.56 GiB/s with transfer worker ~100 % busy (summed wall 60-62 s/min) → transfer is a hard bottleneck; later high-cache-hit windows 1.4-2.7 GiB/s (5-10× faster per byte) at similar prefill batch counts → HiCache traffic not obviously harmful; weakly consistent with (a) compute/CU contention (more new-token compute per transferred byte → slower copy). HiCache io_backend=kernel (CU kernels, not SDMA). Suspended until microbench.
R035 cont. — e2e profiling first 5 min (harness-v2 xfer_summary): 1,629 calls, 1,985.7 GiB, Σwall 384.9 s → 5.16 GiB/s per call stream; per call wall mean 0.24 s (max 4.25), queue wait call-weighted mean 1.34 s (max 19.2 s) → in steady state the queue wait (≈5.6× wall) dominates, not copy speed. Large calls reach ~20 GiB/s (e.g. 1,201 blocks/call, 53 GiB in 2.45 s). Inference: the single per-rank worker serializes items; each item's kv_chunk.wait_event.synchronize (waits for that chunk's compute, outside wall) blocks later already-computed items → lever = more workers per destination (room-sharding / per-request concurrency), not faster copies.

## R036 — root cause of the KV-transfer queue wait: our own non-final-chunk wait_event patch over-waits (repro, first-hand)

transfer_worker is strict FIFO per queue; in 1:1 DP every chunk of a prefill rank lands in ONE queue (other
3 idle). Per item: get → wait_event.synchronize() (outside wall) → batch_transfer_sync → (last chunk) state/
aux/conclude. The INFERA recipe patch _infera_nonfinal_chunk_wait_event (image prefill.py:1107-1114;
patches/sglang_disagg/v0521/patch_prefill_nonfinal_chunk_wait_event.py) records the event on forward_stream
INSIDE process_batch_result(N), but the overlap loop (prefill.py:796-842) has already launched batch N+1 on
forward_stream → the event also waits for the whole NEXT forward (up to a 16k-token step). The chunk's KV is
already complete (copy_done.synchronize() precedes it) → pure over-wait that blocks every queued item
(head-of-line). The patch header itself says "OPEN whether still load-bearing ... cheap (an event on an
already-finished forward)" — premise false under overlap. prefill_transfer_kv_cache (1.91 s) spans
last-chunk enqueue → Success poll, so it includes this queue wait.
Plan e2f: YIHOU_NONFINAL_EVENT=launch (record at N's launch) + YIHOU_XFER_SHARD=room (dilutes residual
HOL) + sync/idle logging; correctness via --disaggregation-enable-kv-checksum (cost to be measured).
R036 cont. — mounts ready (repro; 17/17 CPU tests; py_compile in image): prefill.yihou-nonfinal-event.py (YIHOU_NONFINAL_EVENT orig|launch|off), conn.yihou-xferfix.py (xfersync + YIHOU_XFER_SHARD dest|room), mounts.yihou.v2-xferfix.tsv. Leader wrote config.yihou.xferfix.sh + env.e2f-102.yihou.sh (launch + room + xferlog, KV_CKSUM switch adds --disaggregation-enable-kv-checksum to both legs); dryrun rc=0, tokdiff vs e2c = conn swap + prefill.py mount (both legs) + 3 prefill env.

## R037 — e2e: chunked-prefill 65536 (+xferlog, PAR=1) @C40 = 19,565 tok/s/chip, PASS — L2 REJECTED (21:54)

runs/e2e/agentx-c40/agentx_conc40.json: 3979 ok / 3 err; total 156,520 (19,565/chip, −9.8 % vs 21,696);
TTFT mean 6.94 (p50 3.27, p95 25.1) s; TPOT mean 14.7 ms (+20 %); MAF/OOM/TB 0; max #new-token 16,384/rank.
Deltas vs the 21,696 run: chunk 65536, xferlog conn (log-only), THREAD_POOL 16 (executor unused at PAR=1).
Attribution: chunk 65536 is the only behavioural change → reject (keep 32768). TPOT rise on decode is
consistent with longer prefill steps → burstier KV arrival / longer HOL (inference).
XFER over the full profiling window: 16,694 calls, Σwall 6,114.6 s, 4.79 GiB/s per call stream, queue
wait call-weighted mean 1.63 s (max 41.6 s) ≈ 4.4× wall (0.37 s) → HOL dominates (R036).
e2e stopped 21:57; e2f (launch event + room shard, chunk 32768) armed on GO.e2f after the 22:xx microbench.

## R038 — copy-interference microbench (repro, 102 GPU0→4, 22:26-22:27, byte-exact all arms)

75 × 9 MiB mooncake-style: idle 10.9 ms (60 GiB/s); NULL-stream GEMM loop 10.9 ms (H-X1 refuted
experimentally); torch-stream GEMM 10.9 ms (SDMA default) / 22.7 ms (SDMA=0); pinned-host D2H+H2D 90 GiB/s
background: no effect with SDMA default; **long full-GPU 200 ms kernels on src: 200 ms / 3.3 GiB/s with
SDMA default AND SDMA=0** → the peer D2D copy waits for the resident kernel (behaves like a CU-needing blit
copy or is gated behind it). Production inference: copies stall behind back-to-back long prefill
attention/MoE kernels → explains ~76 ms per 9 MiB block in e2e (consistent, not proven).
Built-in checksum: first call 13.2 s (JIT) per process; idle 2-5 ms/request (16k-128k tok); under long
kernels up to ~200 ms (starved like the copy; synchronous on scheduler thread) → keep OFF for score runs.
Levers to microbench next: high-priority copy streams (hipStreamCreateWithPriority in mooncake StreamPool),
CU-masked copy stream, ROCr env to force SDMA for peer copies (e.g. HSA_ENABLE_PEER_SDMA). e2f GO 22:27:35.

## R039 — R038 reinterpreted from ROCm 7.2.0 source (repro): false dependency via shared HW queues, not CU starvation

CLR copyBuffer: peer D2D > ROC_P2P_SDMA_SIZE (1 MiB, flags.hpp:220) → DmaBlitManager::rocrCopyBuffer with
forceSDMA (rocblit.cpp:474-557, 2256-2330); ROCr GetBlitObject: same-hive GPU→GPU → xGMI SDMA unless
HSA_ENABLE_PEER_SDMA=0 (amd_gpu_agent.cpp:2418-2450, flag.h:93) → our 9 MiB copies ARE SDMA. But CLR maps
all streams onto GPU_MAX_HW_QUEUES (default 4, flags.hpp:150) normal-priority AQL queues (least-used pool,
rocdevice.cpp:2824-2905) and every kernel/marker packet carries the AQL BARRIER bit (rocvirtual.cpp:82-108,
1729-1755) → the copy stream's event/barrier packets queue behind long forward kernels sharing the HW queue;
hipEventSynchronize waits for them. 64 mooncake streams → always collide. Inference from code; to confirm.
Levers: GPU_MAX_HW_QUEUES=16 (env-only), high-priority copy streams (separate pool + HW priority;
mooncake stream_pool.cpp:74 one-liner), CU-masked dedicated queue. Microbench arms a-g prepared
(run_copy_levers.yihou.sh); window after e2f.
R039 cont. — mooncake priority patch ready (repro, not built): MC_HIP_STREAM_PRIORITY=default|high (stream_pool.cpp only; git apply --check OK; hipcc -fsyntax-only gfx950 OK); recipe commit 702af10f (Dockerfile.sglang.v0521-mc4340prio.yihou). Env-only caveat: torch creates 32 normal + 32 high streams first, CLR assigns least-used queues → GPU_MAX_HW_QUEUES=16 alone likely doesn't isolate; needs MC_HIP_NUM_STREAMS small too (arms h/i added). Plan after e2f: stop → during drain build mc4340prio (CPU) → after drain microbench a-i → e2g.
R036/R039 cont. — e2f warmup XFER (22:48): YIHOU-NONFINAL launch nonfinal_chunks=42 fallback=0 event_done_at_send=42 → the over-wait is gone (event already complete at hand-over). But per shard wall ≈ 61.4 s per 62 s span at 0.27-0.29 GiB/s (5 MiB blocks): the worker is now ~100 % busy inside batch_transfer_sync itself → copy throughput (R039 shared-HW-queue stall behind long prefill kernels) is the remaining bottleneck, as predicted. Priority-stream / HW-queue lever is next.

## R040 — e2f (launch event + room shard, chunk 32768) @C40 = 19,864 tok/s/chip, PASS; R037 attribution WITHDRAWN (2026-10-11 04:45)

runs/e2f/agentx-c40/agentx_conc40.json (bench 22:42-00:03): 4014 ok / 2 err; total 158,913 (19,864/chip,
−8.4 % vs 21,696); in 157,751 / out 1,162; TTFT mean 5.57 / p50 2.97 / p95 19.27 s (best of all C40 runs);
TPOT mean 15.3 / p95 21.9 ms (+25 % vs 12.2); E2E mean 21.0 s. MAF/TB 0; one transient prefill /health 503
at 22:36:53 ("no response from detokenizer for 20 s", bring-up only). YIHOU-NONFINAL: event_done_at_send =
nonfinal_chunks, fallback 0 (R036 over-wait removed). e2f stack left UP after the session was killed by an
admin (~22:50); stopped by leader 04:42 (job 202546 ends 05:25:53; hold script is `sleep 86400`).
**Correction to R037:** e2f has chunk 32768 and is as low as e2e (19,565) → "chunk 65536 caused −9.8 %" is
NOT supported; R037's L2 rejection is withdrawn (open).
First-hand decode-log stats over each profiling window (decode log line = 40 steps per DP; strings+regex,
/tmp/yihou_decstat.py): accept len 3.61 in ALL runs (MTP unchanged). Wall per decode step per DP and mean
running-req per DP: e1c40 45.6 ms / 3.86; e2b4-C44 49.5 / 4.04; e2c-C44 50.5 / 4.17; **e2c-C40 41.9 / 3.64**;
e2d 42.5 / 3.64; e2e 47.9 / 3.82; e2f 50.5 / 4.08. Decode gen rate per DP ≈ running × per-req ≈ 344 (e2c40)
… 328 (e2f) tok/s, roughly constant → decode appears saturated (steps back-to-back; ITL ≈ step / accept)
and the score difference is decode step time vs concurrent decode load, not prefill/transfer (inference;
idle gaps inside the wall not excluded). TTFT improvements cannot raise the score if decode is the
bottleneck: requests just wait longer in decode. Decode launch args identical e2c/e2e/e2f except the conn.py
mount (log/transfer code is prefill-side; YIHOU_* env prefill-only). Open: node drift vs. config effect —
needs an e2c-stack control replicate.

## R041 — decode step time at MATCHED load (repro, first-hand) + backend confirmation + user floor (05:00)

repro (diag/decode_cost.yihou/step_regress.yihou.py): median decode step vs busiest-DP running-req 3/4/5/6/7/8:
e2c40 33.9/36.0/37.9/43.0/45.1/46.2 ms; e2d 33.7/35.9/38.1/43.6/45.1/46.4; e2e 36.4/40.0/42.5/47.0/51.9/49.9;
e2f 36.1/40.0/41.9/48.5/49.6/50.6 → ~26 ms fixed + ~2.5-3 ms per request on the max rank; at equal load e2e/e2f
are ~4-5 ms (~10 %) slower. Admission never binds (prealloc>0 0 %, queue>0 ≤0.1 %, retracted 0). In-flight
transfers do not slow steps within a run. Decode code identical (conn diffs prefill-only). → R040's
"more running-req → slower" causality reversed: slower steps → longer decode residence → more running → lower
score. Candidates: node drift, or prefill-side behaviour coupling into decode (CPU contention, XGMI/HBM traffic).
Discriminator: e2g control replicate. Cost model: DP lockstep (step = max rank), indexer ∝ 6 × Σctx per rank.
Backend: server_args on both legs of all v0521 runs = dsa_prefill/decode 'triton' (overrides.py:668-676 honours
explicit args on HIP). User floor (2026-10-11): exceed the triton packup; first-hand it is 21,032/chip.

## R042 — workflow wf_bebda8fc-5e7 (C40 ceiling; 3 analysts + synth + 2 adversarial reviewers + adjudicator), 05:1x

Survived (first-hand, reproduced by both reviewers; scripts /tmp/yihou_wf_c40/{q5,sk1-sk8,lensb1,drift}.py):
1. C40 is bounded by decode step cost: decode busy 0.93-0.96 per DP, no global gap > 4 s (one 6.6 s in e2e);
   step ≈ 23-26 ms + ~3.0-3.5 ms per request on the busiest rank (busiest/mean 1.46-1.60; lockstep pads to
   max DP bs, decode_cuda_graph_runner.py:1410-1412). Score ≈ 99.3 % input tokens; TTFT/G not the bound.
2. Matched-load (kNN) decode step penalty vs e2c40: e1c 1.083, e2b4 1.093, e2c44 1.141, e2e 1.094, e2f 1.110,
   e2d 1.003 → 5/7 windows slow incl. runs WITHOUT the e2e/e2f changes → those changes are not shown causal.
   Fast windows are 16:15-19:17 UTC (two stacks back-to-back); slow 08:19-13:13 and 20:49-24:00 → time-of-day/
   node drift is the cleanest separator; no GPU telemetry was recorded (aiperf --no-gpu-telemetry).
3. Acceptance is simulated (SGLANG_SIMULATE_ACC_LEN=3.61, as in the packup) → MTP-shape changes not a lever.
Refuted: penalty decays within the hour; e2d = clean fresh-stack replicate (it had index_share TRUE + debug
agent → zero clean fresh-stack replicates of 21,696 exist); MAX_ROWS=48 indexer cliff as cause.
Levers: (1) find/remove the 8-14 % penalty (if external: measurement control); (2) explicit cuda-graph decode
bs incl. 9/11/13/15 (captured list is [1..8,10,12,14,16]; ~2 ms per padded slot) — args only, +0.5-1.5 %;
(3) router INFERA_PD_DP_RANK_AFFINITY=false to cut 1.5× DP imbalance (cross-rank HIP-IPC correctness untested →
checksum run first); (4) one-step decode profile at per-rank bs 4-6 outside the scored window.
Plan: e2g = A1 control with full-hour 10 s GPU telemetry (sclk/mclk/power/temp/util) + host/co-tenant; then
A2 same stack after flush (within-stack drift); levers only counted if larger than the A1-A2 spread.
R042 cont. (repro, first-hand v0521 src): lever = decode `--cuda-graph-bs-decode 1..16` (exec_.py:535-538; cuda_graph_hook.py:83-84 locks it; memory_hook.py:193-198 → max_bs 16 overrides --cuda-graph-max-bs-decode 64; draft runners use the same list, eagle_draft_*_cuda_graph_runner.py:149/156; attn_tp=1 → all values survive). Default list = range(1,9)+range(10,33,2) filtered to ≤16/rank (base_cuda_graph_runner.py:76-93). Cost from e2f log: verify graphs 12 buckets 121.9 s / 4.10 GB, draft 9.6 s / 0.46 GB, extend 3.9 s / 0.42 GB, 10.89 GB/GPU free after capture → +4 buckets ≈ +40 s, ≤ +1.4 GB (inference). e2c40 step 8→9 +4.9 ms vs 7→8 +1.1 ms (pad); 5→6 +5.1 ms between exact buckets (hypothesis: MoE/GEMM tile switch at M 120→144 rows; untested). Profile script drafted: diag/decode_cost.yihou/profile_decode_step.yihou.sh (not run).

## R043 — lever microbench (repro, 102 GPU0→4, 05:36:52-05:38:08, all arms byte-exact): R039 confirmed first-hand

75 × 9 MiB peer copies, load idle | torch GEMM stream | 200 ms full-GPU spin on src (wall ms):
a normal 64 streams 10.9/10.9/200.6; b GPU_MAX_HW_QUEUES=16 64 str 10.9/10.9/200.0; **c high-priority 64 str
10.9/10.9/11.0 (59.7 GiB/s)**; d CU-masked 8 str 11.3/11.2/12.9; e normal 1 str 12.9/12.9/12.9; f
HSA_ENABLE_PEER_SDMA=0 (blit) 11.8/11.7/197.7; h HW_QUEUES=16 + 8 str 11.3/11.3/13.4; i HW_QUEUES=40 + 8 str
11.3/11.3/12.3. AMD_LOG_LEVEL=4: every peer copy rocblit.cpp:549 forceSDMA=1 engineType=4 (xGMI SDMA 0x4/0x10);
exactly 4 SWq→4 HWq priority-1 (shared normal pool). → stall = false dependency via shared HW queues,
engine-independent. Production levers: MC_HIP_STREAM_PRIORITY=high (mc4340prio image, building) or env-only
GPU_MAX_HW_QUEUES=16/40 + MC_HIP_NUM_STREAMS=8. Copy-latency only; secondary while C40 is decode-bound (R042).
GO.e2g touched 05:38:14.
R043 cont. — mc4340prio image built (repro): infera-sglang:v0521-yihou-1010-mc4340prio sha256:aacc9aeb…, 05:39:24Z, clone @702af10f, both patches applied (pr4340 port 6d73b2f5…, stream-priority 8c7205d8…), all build asserts pass; engine.so 6e082ad5… (mc4340: 3c96489a…; both differ from f22b1364 by the #4340 port). Use MC_HIP_STREAM_PRIORITY=high on prefill; unvalidated in production (te_pair byte-exact ~2 min pending a GPU window).

## R044 — e2g A1 (exact control replicate, fresh stack) early read 06:11-06:51: penalty PRESENT; GPUs not clock/power limited

repro step_vs_gpu.yihou.py, matched-load median step ms by max-DP running (e2c40 in []): 2:32.4[30.5] 3:37.2[33.9]
4:39.6[36.0] 5:43.5[37.9] 6:49.0[43.0] 7:50.1[45.1] 8:54.0[46.2] 9:59.6[51.1] → +2..+8 ms (median +4.8, mean +7.8)
≈ e2f. tput_in 141k at 33 min (e2c40 ~164k). → the 21,696 config does NOT reproduce on a fresh stack now; e2c→e2f
code/config deltas are not the cause (R037/R040 attributions superseded).
Telemetry (hostsample 10 s): decode GPU4-7 sclk ~2384 MHz, mclk 2000 flat, ~760 W of 1400 W cap, hotspot 66 °C,
HBM 62 °C, gfx busy 99 %; host loadavg ~20/160, idle 91 %, no foreign containers. Within a fixed load bin, step
time anti-correlates with power (Spearman −0.51..−0.61) and UMC activity (−0.40..−0.53): slow windows draw ~30 W
LESS → GPUs do less work per unit time → extra time is gaps between kernels/graphs (host/scheduler stalls,
DP-sync waits, cross-stream waits), not clock/thermal throttling (inference; 10 s samples vs ~1.7 s windows).
Sampler bug (ssh in while-read ate stdin): no per-thread CPU data for A1 until ~06:55; fixed for A2.
Next discriminator: CPU-side trace in the slow state (py-spy of decode schedulers + kineto CPU/GPU) post-A2.
Open: why the 16-19 UTC windows (e2c40 on a long-running stack, e2d fresh) were fast.
R044 cont. (repro sidecars, first-hand 06:52): decode scheduler main threads 85-95 % CPU, plus one unnamed helper thread per scheduler spinning 98.7-99.4 % (user space, few ctx switches); DP controller 100 %; prefill similar. Main threads pinned to split sets spanning both sockets (e.g. DP0 "0-28,118-146"; node0 0-117, node1 118-235); helpers unpinned; observed main/helper on different nodes. → decode loop is host-bound; drift candidate = host placement (cross-NUMA migration / SMT sharing) — inference, testing with 1-s cpuplace log vs step time. py-spy 0.4.2 in image works from a SYS_PTRACE sidecar (Python stacks only).

## R045 — host placement facts (repro, first-hand, 07:0x): wrong-NUMA scheduler pinning; prefill/decode share CPU sets

py-spy (nonblocking) decode schedulers: MainThread in `synchronize ← resolve_seq_lens_cpu (overlap_utils.py:532) ←
run_batch (scheduler.py:4391)` = v0521 HIP workaround (overlap_utils.py:528-532 "Event.wait() regresses TPOT on
AMD MI355" → publish_ready.synchronize()): CPU spin-waits for the previous verify, then launches → any CPU work after
the sync is on the critical path (fits "slow steps draw less GPU power"). 99 % helper = ROCr KFD event-wait thread
(ioctl /dev/kfd AMDKFD_IOC_WAIT_EVENTS), unpinned. DP controller spins on zmq NOBLOCK (data_parallel_controller.py:806-812).
Affinity: SGLANG_SET_CPU_AFFINITY=1 comes from the BASE image env (sglang default False, environ.py:1541) →
set_gpu_proc_affinity(gpu_id) (utils/common.py:3095-3124, scheduler.py:5890-5891): 29 cores per local gpu_id +
"HT sibling" id+118. On this box: GPU0-3 → NUMA node0 (CPUs 0-117), GPU4-7 → node1 (118-235); SMT siblings are
adjacent (i, i+1). So (a) decode DPk (GPU 4-7, node1) is pinned by local id 0-3 to 0-115 (node0) + 118-233
(node1) halves; (b) the "+118 sibling" is the other socket; (c) prefill DPk and decode DPk get IDENTICAL CPU sets.
--numa-node is a no-op on ROCm (numa_utils.py:364-365; numactl absent). Host Shmem 2.2 TB (HiCache pinned pools);
node1 MemFree 4 GB (MemAvailable 541 GB), no reclaim seen in a 10 s sample.
Hypothesis H-place (to test with the 1-s cpuplace log vs step time): decode step pacing depends on which socket the
spin-then-launch scheduler thread happens to run on / co-runs with prefill → run-to-run variance.
Lever (args only, fresh stack): docker --cpuset-cpus node-local per leg (decode 118-235, prefill 0-117) +
SGLANG_SET_CPU_AFFINITY=0 (or a corrected per-DP pin), if spur-authz allows --cpuset-cpus.
R045 cont. — e2h prepared (harness-v2): spur-authz allows --cpuset-cpus/--cpuset-mems (alpine test on 102). e2h = e2g + prefill --cpuset-cpus 0-117, decode --cpuset-cpus 118-235, SGLANG_SET_CPU_AFFINITY=0 both legs; no cpuset-mems (HiCache pools ~1.8 TB span both nodes; free node0 16 GB / node1 4 GB). engine.sh knobs PREFILL/DECODE_CPUSET_CPUS/MEMS + ENGINE_SET_CPU_AFFINITY (default empty → e2g byte-identical); audit checks added; live e2g containers carry image ENV SGLANG_SET_CPU_AFFINITY=1 (premise confirmed). Order: A1 → A2 → stop → e2h.

## R046 — e2g A1 full window: penalty is a TRANSIENT (rise-peak-decay over the hour); placement does NOT explain it (repro, 07:13)

Matched-load residual vs e2c40 per 10-min bin of profiling: A1 +2.1 | +7.0 | +5.1 | +4.0 | +3.1 | +1.7 ms;
e2f +2.3 | +9.6 | +6.5 | +5.4 | +2.6 | +2.9; e2c40 flat (self-ref) −0.3..+0.2. Full A1 median +3.9 (mean +7.0).
Not tracking (rank r): main-thread migrations +0.15, node switches +0.15, SMT co-location 3.2 % of samples (+0.25),
prefill power −0.07 / gfx +0.05, host idle +0.02, running-req churn / transfers / DP imbalance |r| ≤ 0.18, ctx/req
similar; clocks/power/thermal flat. Tracks: decode GPU power r = −0.64 (60-s bins; slow ≈754 W vs 800 W) → idle
gaps between kernels/graphs. → R045 placement hypothesis NOT supported; e2h (cpusets) deprioritised.
Open hypotheses (time-in-run): (a) GPU-side KV/page-table/indexer behaviour as the decode pool fills/fragments
(usage 0.27→0.45); (b) host-side per-step scheduler work growing then shrinking; (c) node state outside 16-19 UTC.
Note e2c40 ran on a long-running (pre-loaded) stack; e2d (fresh, index_share TRUE) was also fast — so "fresh-stack
transient" is not established. A2 (same stack, after flush) discriminates stack age: flat → stack-age transient.
Profile should be taken 10-20 min into a run (the peak).

## R047 — 10-min-bin residual across ALL runs (repro): clean ↔ wall-clock window 16:15-19:17 UTC, not stack age

Residual vs e2c40 (all | per 10-min bin), stack age at profiling start: e1c 08:19 age 42 min +3.8 (no peak);
e2b4-C44 10:04 age 29 +4.3 (flat); e2c-C44 12:12 age 28 +3.2 (mild hump); **e2c40 16:15 age 271 +0.0 flat**;
**e2d 18:16 age 22 +0.1 flat**; e2e 20:49 age 23 +4.0 hump; e2f 22:59 age 22 +4.4 hump; e2gA1 06:11 age 24 +3.8 hump.
→ stack age refuted as sole cause (e2d fresh & flat; same e2c stack elevated 12:12, flat 16:15); index_share not
necessary for clean. Clean ⇔ 16:15-19:17 UTC so far (inference: external time-of-day factor, e.g. shared NFS /
fabric / power policy) + a run-phase hump (peak 10-20 min) in e2e/e2f/A1. Falsifier: control C40 inside 16-19 UTC
today. Server logs: container json-file → `docker logs -f` → NFS (engine.sh:330-333), decoupled from the container
unless dockerd back-pressures; $WS (NFS) is mounted rw into containers (caches/tmp) — runtime NFS access by decode
schedulers unchecked.

## R048 — e2g A1 (exact replicate of the 21,696 config, fresh stack, 06:11-07:12 UTC) = 19,640 tok/s/chip, PASS

runs/e2g/agentx-c40/agentx_conc40.json: 3993 ok / 4 err-dropped / 3 cancelled; total 157,121 (in 155,974 / out 1,146);
TTFT mean 7.61 / p50 3.94 / p95 28.7 s; TPOT mean 14.3 / p95 21.1 ms; E2E mean 21.7 s; MAF/TB/OOM 0.
→ identical config reproduces the e2e/e2f level (19.6-19.9k), not 21,696 → the 21,696 run was a favourable
window (R047: 16:15-19:17 UTC), not a config effect. A2 (same stack, flush) follows automatically.
R047 cont. (repro, first-hand): decode schedulers have no NFS file I/O on the hot path (open fds only prometheus .db on container tmpfs; TMPDIR tmpfs; NFS = AITER_JIT_DIR, $WS, model ro, caches touched at JIT/load only; lifetime write_bytes 4-6 MB). A2 sampler adds NFS latency probe (idle baseline write+fsync 7.7 ms, GETATTR 125-155/s @0.3 ms node-wide), docker log rate, per-thread /proc io around py-spy records (min 12, min 50).

## R049 — A2 (same stack, 2nd run, 07:31 UTC) is FAST; decode-history table; e2d = fresh but debug-agent decode (repro, first-hand)

A2 matched-load residual vs e2c40, first 18 min: −0.9 ms (0-10m −1.2 n=264; 10-18m −0.5 n=68); tput_in 160k at
15 min (e2c40 153k, A1 131k) at a "slow" hour → R047 time-of-day reading WEAKENED.
Decode steps served before each profiling window (DP0 lines ×40): e1c, e2b4, e2c-C44, e2d, e2e, e2f, A1 = 0
(fresh); e2c40 ≈73k (C44 run 12:12-13:13, idle until 16:15); A2 ≈76k (A1). Fresh → slow except e2d; warm → fast.
e2d is the only run whose decode leg had HSA_TOOLS_LIB=librocm-debug-agent.so.2, HSA_ENABLE_DEBUG=1,
ROCM_DEBUG_AGENT_OPTIONS=--disable-linux-signals (config.yihou.seedguard.sh:20), plus index_share TRUE + seedguard.
No JIT/compile/capture lines in any profiling window.
Hypotheses: (a) persistent warm-up surviving /flush_cache (allocator/IPC-handle caches on the decode receive path,
KV page-table/free-list shape, GPU TLB/page mapping of the KV pool first-touched by transfers); (b) HSA_ENABLE_DEBUG /
debug agent changes ROCr queue/signal/interrupt handling → fast for another reason (env-only lever).
py-spy A2 peak (20 s, ~2000 main-thread samples/scheduler): sync-wait resolve_seq_lens_cpu 66-76 %; run_batch launch
16-17 % of which ~12 % = graph_variants.py:39 `forward_batch.seq_lens.max().item()` (2nd D2H sync per verify,
decode_cuda_graph_runner.py:621, dense/sparse variant choice); DP all_gather 3-6 %; no logging/file I/O.
Next: e2i = fresh stack, e2g + decode debug-agent env only (index_share false) → R1 fresh discriminates (a)/(b),
py-spy at min 15; R2 warm on the same stack.

## R050 — levers drafted (repro): host-side DSA graph-variant choice; what HSA_ENABLE_DEBUG/debug agent change

(1) 2nd D2H sync path: eagle_worker_common.py:615 → eagle_utils.py:621 → decode_cuda_graph_runner.py:1515 load_batch →
:615-620 → graph_variants.py:31-44 DsaGraphVariants.select: seq_lens_cpu None under spec-v2 overlap →
`seq_lens.max().item()`; DENSE iff max_kv_len ≤ index_topk 2048 else SPARSE. Host lower bound max(kv_committed_len)
lags device by ≤1 verify (batch_result_processor.py:805) → patch: if host lower bound > topk → SPARSE without sync,
else original. Exact equivalence; per-rank choice unchanged (no collective in attention at attn_tp=1). Mounts
graph_variants.yihou-hostvariant.py + eagle_utils.yihou-hostvariant.py, gate YIHOU_DSA_HOST_VARIANT=1 (decode).
CPU unit test 7/7; GPU-untested. Expected: removes ~12 % main-thread block in verify prep.
(2) ROCr 7.2.0: HSA_ENABLE_DEBUG only → hsaKmtRuntimeEnable(setup_ttmp) (amd_kfd_driver.cpp:98-99); interrupt wait is
HSA_ENABLE_INTERRUPT (runtime.cpp:2392); CLR active wait independent. Kernel (v6.8 src; node runs DKMS 6.14.14):
debug-agent attach → kfd_dbg_trap_activate: trap debug flag, amdgpu_gfx_off_ctrl(false) during setup, debug_refresh_runlist
(HWS runlist re-mapped in debug mode). Plausible perf links only via HWS runlist/debug mode or power gating
(GFXOFF / deep sleep on idle gaps) — unverified; e2i decides empirically; follow-up HSA_ENABLE_DEBUG=1 alone vs agent.
R050 cont. (repro, first-hand 07:54, warm A2): no GFXOFF on gfx950; DS_GFXCLK etc. enabled (pp_features); gpu_metrics accumulators readable. Prefill GPU0-3 PPT-throttled 10-28 % of time (ACTIVE) at 500-1100 W avg vs 1400 W cap; decode GPU4-7 ppt 0.01-0.39 %, low-util/deep-sleep 0. New sampler gpuacc_sampler.yihou.py (read-only, 10 s). Hostvariant mounts py3.10-compile OK; graph_variants import smoke OK.

## R051 — e2g A2 (target config, C40, warm stack 2nd run after flush, 07:31-08:31 UTC) = 22,114 tok/s/chip, PASS — GOAL MET

runs/e2g/agentx-c40-A2/agentx_conc40.json (verified first-hand): 4286 ok / 1 err-dropped (InvalidInferenceResultError);
total 176,909 tok/s = **22,114/chip** (in 175,652 / out 1,257), 3629 s; TTFT mean 5.30 / p50 3.33 / p95 19.21 s; TPOT
mean 12.1 / p95 17.9 ms; E2E mean 17.6 s; MAF/OOM 0 (one asyncio CancelledError at teardown). Image
infera-sglang:v0521-yihou-1010-mc4340, config = e2c/e2g target (triton DSA both legs, prefill HiCache 450 GB +
CHUNK_GB fix, decode gmu 0.95, MTP 5/1/6 sim-acc 3.61, index_share FALSE), cold caches (first-turn #cached 0).
vs triton packup floor 21,032 (+5.1 %); vs 21,696 (+1.9 %); vs A1 same stack first run 19,640 (+12.6 %).
Protocol (same as the 21,696 run): stack serves one full C40 AgentX run first (A1), then /flush_cache on both legs,
then the scored run. Mechanism of the first-run penalty still open (R049 (a) persistent warm-up vs (b) debug agent →
e2i). e2g stopped 08:35-08:36; e2i started.

## R052 — e2i R1 (fresh stack + decode debug agent, index_share false) is SLOW → (b) refuted; slow state localised to the draft phase

Residual vs e2c40: all +2.0 | 0-10m +1.4 (n=132) | 10-20m +8.7 (n=51) — same fresh-stack hump as e2e/e2f/A1, with the
debug-agent env confirmed (dryrun + 1,679 agent log lines) → hypothesis (b) "debug agent makes it fast" NOT supported;
(a) stack history/warm-up stands.
py-spy slow (e2iR1 min 15) vs warm (A2 min 15.5), ~2000 main-thread samples × 4 schedulers: resolve_seq_lens_cpu sync
64.9-65.7 % vs 66-76 %; **graph_variants.py:39 seq_lens.max().item() 22.3-22.7 % vs 11.5-12.1 %** (+10 pts);
all_gather 7.2 % vs 3-6 %; total blocked-on-GPU ≈ 88 % both. The 2nd sync (verify prep of batch N) waits for the work
queued after N-1's verify = N-1's draft_extend + 5-step draft (scheduler.py:4391-4413 order) → in the slow state that
GPU phase is ~4-5 ms longer (= the residual). Inference, not kernel-traced. GPU-side, consistent with lower power.
gpuacc: prefill PPT 9.8-12.9 % fresh vs 10.4-12.2 % warm (not the discriminator); decode PPT ~0.2 %; decode
idle-downclock 0.00 % both → ruled out.
Coherent candidate (inference): the fresh-stack penalty lives in the MTP draft phase, which with index_share=false
runs the draft layer's own DSA indexer each draft step; e2d (fresh, index_share TRUE reuses the first step's top-k,
eagle_worker_v2.py:280-296) was the only fresh run without the penalty. To test with a kineto trace of draft /
draft_extend kernels fresh (min 10-20) vs warm.
The R050 host-variant mount removes the CPU block at this sync but not the longer GPU work → unlikely to remove the
penalty by itself.
R052 cont. (repro, code): every decode step replays CUDA graphs (cuda graph: False = 0 lines in A1/A2/e2iR1/e2d/e2c40) → runtime triton autotune (keys incl. total_tokens_bucket in nsa_triton_decode fused kernels, captured at bring-up) cannot warm up over a run; aiter paged-MQA compiles per static config; eager-kernel JIT and allocator frames ~0.2-0.4 % in py-spy both states → excluded. No code-level warm-up cache found; remaining = GPU-side, time-in-process dependent → kineto diff (e2j R1 min 12 armed; warm trace after e2j R2 at bs5/ctx256k).
R051 cont. (harness-v2): A2 decode_stats ms/step/DP 40.9, run 3.57, per-req 97.3, acc 3.610, imb 1.41, input mean 148.7k; step by maxrun A2 vs e2c40: 3 33.1/33.9, 4 34.9/36.0, 5 37.4/37.9, 6 42.9/43.0, 7 44.7/45.1, 8 46.4/46.2 → parity (A1 +7-12 %). e2j chain armed (GO 10:00, agent=0) after e2i R2 (~12:00); e2j R1 ~12:20.
R052 cont. — e2i R1 final (fresh + decode debug agent): 4075 ok / 2 err; total 162,970 = 20,371/chip; TTFT mean 6.37 /
p95 21.7 s; TPOT 14.1 ms; MAF/TB 0. First-run class (A1 19,640; e2e/e2f 19.6-19.9k; e2d 21,383 exception). Host HiCache
reached cpu_kv_usage 100 % near the end; one TTFT burst (~30 req ≈80 s) at 10:21 with cpu_kv 88→94 %. R2 (warm) next.

## R053 — e2i R2 (warm 2nd run, decode debug agent on) = 21,921 tok/s/chip, PASS; warm-protocol replicate

4258 ok / 1 err-dropped / 4 cancelled; total 175,366 (in 174,122 / out 1,244); TTFT mean 5.83 / p95 19.4 s; TPOT 12.3 ms;
MAF/TB/OOM 0. Pairs (1st fresh run → 2nd warm run on the same stack): e2g 19,640 → 22,114; e2i 20,371 → 21,921;
(e2c stack: C44 run → 21,696). Warm-protocol C40 runs: 22,114 / 21,921 / 21,696 → mean 21,910, spread ±1 %.
→ the warm protocol reproducibly beats the 21,032 floor (+3-5 %); 22k is at the edge (1 of 3 above) → need a lever
with > ~1 % at warm state (e2j: host-variant sync removal + cuda-graph bs 1-16, next).

## R054 — e2j R1 (fresh; host-variant + cuda-graph bs 1-16) first read + kineto: GPU-side cross-stream stall "gap A" (repro)

Levers verified: capture bs=[1..16] on target-verify/draft-decode/draft-extend (target capture 107.7 s, 4.13 GB; avail 10.82
GB vs 10.89), 0/1146 eager lines; py-spy: `select (graph_variants.py:39)` frame gone (was 12-23 %), run_batch launch 2.8 %
(was 16-26 %), main thread 82-84 % in resolve_seq_lens_cpu sync. Residual 0-10m +3.4 (e2iR1 +1.4, A1 +2.1); raw step at
max-run ≤5 ≈ e2iR1 +1 ms; ≥6 small n. Early low tput/TTFT p95 44 s looks prefill-side workload phase (prefill batches
first 10 min: e2j 836 / e2i 1137 / A1 873) — not attributable to the decode levers.
Kineto (min 12, 8 steps, all 4 DPs; DP0 bs 8-9 ≈ 64 ms/step; GPU busy union 72.7 %): top kernels nccl 17 %, MoE fp4
gemm1 15.6 %, _sparse_mla_fused 13.6 %, indexer paged-MQA 9 %, MoE gemm2 7.7 %, topk 4.5 %. Two idle gaps per step on all
ranks: **gap A 8.5-10 ms** — schedule stream finishes assign_req_to_token_pool/HtoD, then the forward stream stays idle
8.5-10 ms before a kernel the CPU launched 3.5-4.5 ms BEFORE the idle began (after hipStreamWaitEvent, consistent with
forward_stream.wait_stream(schedule_stream), scheduler.py:4396) → a GPU-side dependency stall with work already queued;
gap B 2.4-4.3 ms — CPU-late launch around draft_extend. Gap A ≈ 14 % of the step ≈ size of the fresh-stack penalty.
Hypothesis: false dependency through shared HW queues / AQL barrier (R039 mechanism) between decode streams — e.g. the
forward or schedule stream mapped to the same HW queue as a long in-flight kernel (nccl?) on another stream; queue
mapping order may differ fresh vs warm. Needs the warm trace (after e2j R2). Env-only lever candidate: GPU_MAX_HW_QUEUES
↑ on decode.

## R055 — gap A is a GPU front-end dependency-resolution stall with the whole GPU idle (repro, first-hand except mechanism)

During every gap A (4 DPs × 7 steps) NO kernel/memcpy of the decode process runs on any stream; only 2 streams carry GPU
work (s8 forward incl. NCCL; s7 schedule). s7's last op ends at gap start; the next s8 op (CatArrayBatchedCopy →
build_tree_efficient, draft-input build) was launched 3.5-4.5 ms before and starts 8.5-10 ms after. CPU: hipStreamWaitEvent
~4.45 ms before, then a ~58 ms hipEventSynchronize (publish sync); nothing launched during the gap. KFD sysfs: only the 4 decode
PIDs own queues on GPU4-7 (prefill's are on 0-3); per decode process 4 compute AQL HQDs (1 MiB) + 1 small + 2 SDMA; prefill
4+1 compute, 2 SDMA, 14 XGMI-SDMA. gfx950: num_cp_queues 24/GPU, sched_policy=0 (HWS, oversubscription allowed), mes=0,
cwsr on. No runtime stream creation (KFD queue counts fixed at 7) → stream→HQD mapping fixed at startup (could differ
between processes by creation order — unproven).
Mechanism (inference): s8's op sits behind an AQL barrier waiting on an event signalled by s7 on another HQD; with the GPU
idle the CP/HWS takes ~9 ms to resolve it (poll/time-slice granularity suspect). Fresh vs warm still unexplained → warm trace.
Levers (env-only, decode): GPU_MAX_HW_QUEUES=16 (≤24 HQDs, no oversubscription; may not help if cross-HQD barrier latency
is the cause); GPU_MAX_HW_QUEUES=1/2 (s7+s8 on one HQD → dependency by in-queue order, no cross-queue barrier); NOT 32
(>24 → HWS oversubscription). Microbench (stream B waits on A's event, GPU idle; HW_QUEUES ∈ {1,4,16}, ± third stream with
pending barrier) planned for the next GPU window (after e2j R2).
R055 cont. (repro, CLR 7.2.0 first-hand): ROC_CPU_WAIT_FOR_SIGNAL=true default (flags.hpp:193); rocvirtual.cpp:684-696 — a submission depending on another HW queue's incomplete signal is waited on the CPU (CpuWaitForSignal → WaitForSignal(ActiveWait), :705-725) instead of a GPU barrier → cross-stream waits can become host waits whose wake-up depends on interrupt/polling (HSA_ENABLE_INTERRUPT, ROC_ACTIVE_WAIT_TIMEOUT flags.hpp:189). Caveat: gap-A launch hipLaunchKernel took ~10 µs (no visible CPU block). Microbench v2 run_xstream_wait2.yihou.sh (9 arms incl. ROC_CPU_WAIT_FOR_SIGNAL=0, HSA_ENABLE_INTERRUPT=0, HW_QUEUES 1/2/4/16, queue-log) ready for the post-R2 window.

## R056 — e2j R1 result; host 005 lost; resume on 006 (2026-10-12 03:5x UTC)

e2j R1 (fresh; decode host-variant + cuda-graph bs 1-16): 3938 ok / 2 err; 19,335 tok/s/chip; TTFT mean 7.20 / p95 27.95 s;
TPOT 14.7 ms — first-run class (A1 19,640, e2i R1 20,371); levers not separable from the fresh-stack penalty in R1.
Login host crsuse2-slog-005 became unreachable during the e2j chain: last chain line "stage bench R2" 14:31:17 2026-10-11;
the R2 client never started (no agentx-c40-R2 dir). e2j stack (warm, flushed 14:24-14:30) stayed UP on 102 ~13 h idle.
Teammates, crons, py-spy/kineto watchers and repro's TTFT decomposition (ttft.e2jR1.yihou/: raw scrapes only) were lost.
Job 207495 TimeLimit raised to 2-00:00:00 (hold is sleep infinity → effective; EndTime 2026-10-13T05:26:35).
Resume plan: re-flush + run e2j R2 (warm, scored comparison vs A2 22,114 / e2i R2 21,921); then GPU window: xstream_wait2
microbench + warm kineto; then lever decision (GPU_MAX_HW_QUEUES / ROC_CPU_WAIT_FOR_SIGNAL) and packup.

## R057 — e2j R2 (warm; decode host-variant + cuda-graph bs 1..16) = 21,681 tok/s/chip, PASS — levers give no warm gain

runs/e2j/agentx-c40-R2/agentx_conc40.json (profiling 04:10:07 2026-10-12): 4242 ok / 1 err; total 173,451; TTFT mean 5.21 /
p95 16.57 s (best TTFT tail of the warm runs); TPOT 12.4 ms; decode 42.6 ms/step/DP, run 3.70. Warm-protocol runs now
22,114 / 21,921 / 21,696 / 21,681 → mean 21,853, spread −0.8…+1.2 %. The two decode levers are within that spread (no
measurable throughput gain; the removed sync was not on the GPU critical path). Stack left UP for repro's warm kineto.

R057 cont. — correction from decode.kvcache.report.md (first-hand decode logs): the −0.4 % decode KV capacity on v0.5.21 vs v0519 (R001 framing) is the fused DSA indexer, not the version: e2b4 (fused off) 2,773,376 = v0519 L1-0.95; e2c/e2g (fused on) 2,762,176. Report: /home/yihou/dev/git/infera.glm52.view/decode.kvcache.report.md.
