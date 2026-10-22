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
