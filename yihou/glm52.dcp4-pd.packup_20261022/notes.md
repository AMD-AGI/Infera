# Notes — gotchas, lessons, known risks, open questions (what / why / how / context)

Source abbreviations as in `working_process.md` (poll = `poll.md`; P0xx = `ledger.pdrun.md`; M Rxx = `ledger.mixrun.md`;
Txx = `ledger.benchrun.md`; local workspace `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp/yihou_ws/glm52_dcp_pd_20261021/`;
`PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd`). "Why" is filled only where a log or code read shows it; otherwise it
says "not established".

## 1. v3 burst nodes: weight load hangs in SDMA copies -> `HSA_ENABLE_SDMA=0`

- **What:** on crsuse2-m2m-v3-045 and v3-060 every sglang server hung after "Multi-thread loading shards 100%": VRAM frozen,
  GPU ~0 %, 31 loader threads blocked on one futex in `c10::hip::memcpy_and_sync`, 2 HIP threads in
  `kfd_wait_on_events` (M R4-R8, T2 S1).
- **Why:** not established. Excluded by experiment: sha, cpuset, `SAFETENSORS_FAST_GPU=1`, `enable_multithread_load=false`,
  single-process and 4-process loader-like copy tests (all pass). Node difference: v3 nodes amdgpu 6.16.13 / kernel
  6.8.12-322.crusoe; 054 amdgpu 6.14.14 / 6.8.0-107-generic (M R5).
- **How:** `HSA_ENABLE_SDMA=0` on both PD legs and all mixed servers on v3 nodes (blit-kernel copies). Loads then work
  (main model 83 s vs 38 s on 054 with SDMA, M R8). Label all v3 numbers "v3/SDMA=0".
- **Context:** reproduced on v3-060 by benchrun (T2 S1, poll 16:39Z); reported to the pool manager. PD r5/r5b and E2
  run with it (P016, M E2).

## 2. aiter JIT cache: one directory per GPU slot

- **What:** two containers sharing one aiter JIT dir crashed at the first request (`module_gemm_a16w16_asm.so: cannot
  open shared object file`; `module_custom_all_reduce.so` 0 bytes) (M R1).
- **Why (log read):** aiter's build lock is PID-based; containers have private PID namespaces -> "broke stale lock" ->
  concurrent builds clobbered each other.
- **How:** per-slot dirs, e.g. `cache/<node>/aiter-jit/<image>/g0123` and `.../g4567` (mixrun `serve.sh`).
- **Context:** the first request on a cold JIT dir costs ~8 min (M R2); decode cold JIT in PD ~11 min (P010). Do not
  kill during cold JIT / graph capture.

## 3. Same-node containers: `PID_OFFSET=30000`

- **What:** RCCL init `HIP failure: 'invalid argument'` / `hipIpcGetMemHandle failed: invalid argument` when two
  containers on one node start ranks with equal PIDs (M R2, T1).
- **How:** burn PIDs before exec (`PID_OFFSET=30000`; PD harness `DECODE_PID_OFFSET 30000`, P008). Did not recur after.
  Run collective microbenchmarks serially (T1).
- **Context:** same as the 22k kit's R022 (M R2).

## 4. `spur exec` "transport error" breaks drivers, not clients

- **What:** run.v2 bench drivers, chain drivers, `docker stop` wrappers returned "transport error" (P011, P012, P014,
  P015; poll 14:24Z, 15:32Z, 01:35Z 10-22). The node-side client containers kept running.
- **Consequences seen:** pdBeperf-r2 chain stopped after A1 ("no JSON") and A2 was not started for 70 min (poll 16:53Z);
  pd_r3 stopped at docker stop (P014); r4 A1 never started a client (P015); r5 `status.tsv` showed FAIL rows for a running
  A1 (poll 01:35Z 10-22).
- **How:** drivers retry spur exec and only accept a clean "not running" (`harness/pdchain_a2.yihou.sh`,
  `pd_r3b.yihou.sh` checks `State.Running=false`, `bench_retry.yihou.sh`); verify a bench from the client's exit code and
  result JSON, not from `status.tsv`; run drivers with `setsid nohup` on the login host.

## 5. pd_seq double-bench bug

- **What:** on v3-060 the run.v2 bench stayed attached for the whole run; pd_seq's retry loop would then launch a second
  bench after the JSON already existed (P016, 02:08Z 10-22).
- **How:** pd_seq breaks if the result JSON is present; continuation `harness/pd_cont.yihou.sh` (wait A1 client ->
  flush -> A2). Edit drivers via new file + `mv` (bash reads scripts incrementally).

## 6. ROCm debug agent abort on v3 (`DEBUG_AGENT=1` -> rocm-dbgapi fatal)

- **What:** r5 A1 (v3-060) died at 01:42:52Z 10-22: `rocm-dbgapi: fatal error: os_queue_packet_id 0xb2089 is not within
  [0xb20b7..0xb20f8[ in queue_1` -> `rocm-debug-agent: error: debug_agent.cpp:924 ...` -> `Fatal Python error: Aborted`
  in a prefill scheduler (right after LoadKernel `bf16gemm_fp32bf16_tn_64x64_splitk_clean`); DP0 then died on gloo (P016).
- **Why:** the abort is raised by the debug agent itself (first-hand from the log). Why the packet id was out of range is
  not established. Not seen on 054 (amdgpu 6.14.14).
- **How:** perf runs use `DEBUG_AGENT=0` (r5b). The 22k e2g perf runs also used 0; PD r2/r4 ran with the agent on
  (`spec/design/pd_vs_mixed_decode_diff.md` §1 row 2). Keep the agent for fault localisation only (it named the
  faulting kernels in M R17/R19).

## 7. Decode-only restart under a live prefill/router -> "no content" errors (r3)

- **What:** r3 restarted only the decode (custom AR ON, NCCL debug off) on the live r2 prefill/router. AgentX aborted:
  15/142 = 10.6 % `InvalidInferenceResultError: No responses with actual content`, 132 warmup errors from the first
  request (P014).
- **Why:** not established. r4 (fresh full stack, same decode settings) had 2 warmup / 0 profiling errors (P015), so the
  errors follow the decode-only restart in one comparison each; not separately tested.
- **How:** restart the full stack (prefill + decode + router) for every PD round.

## 8. Decode hang after a client cancel (fixed in code, not GPU-verified)

- **What:** after the r3 client cancelled 36 in-flight requests, all decode ranks logged `KVTransferError`, then hung:
  TP0/1 in broadcast (`_broadcast_reqs_across_ranks`), TP2/3 in `all_reduce` under `pop_transferred` (py-spy, P014).
- **Why (code read, analyst):** deferred decode KV release freed device pages on rank-local acks/deadlines, so prealloc
  admission diverged across attn-TP ranks; `pop_transferred` skipped its poll all-reduce when its local queue was empty.
- **How:** c4eda0343e (MIN-reduced ready flags; group decision for the empty skip; RuntimeError on divergence). 5 new CPU
  tests, 4 fail on old code (analyst). **GPU cancel repro (`tools/pd_cancel_repro.yihou.py`) never run.** r5/r5b carry the
  fix (sha 93034eba91).

## 9. DP2 x DCP2 bug chain (all fixed; not DCP-specific by code read)

Three bugs from DP-attention padding with attn_tp > 1 + ROCm + MTP (RESULTS §4, coder2 ledger):
1. `graph_utils.padded()` did not pad a batch-major `[1, seq, H, D]` DSA triton prefill output -> `prepare.py:246`
   49 vs 50 rows. Fix 31309dc4f1 (cherry-pick b17da53725).
2. ROCm lacked the eager-IDLE indexer short circuit (gated `_is_cuda`) -> idle DP group ran `topk_main_kernel` (v2 paged)
   on padded rows -> MAF. Fix fdf73b6fb2 (cherry-picks e9e1432ce6, da6e383513).
3. `_pad_inputs_to_size` extended `extend_seq_lens_cpu` in place, shared with pre-planned draft-extend DSA metadata ->
   q 42 vs 36 rows -> MAF in `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle`. Fix fa810b9f69 (cherry-pick 5aacce1f1c).
- **Method that worked:** custom AR off / `SGLANG_DP_USE_REDUCE_SCATTER=0` as discriminators, then
  `AMD_SERIALIZE_KERNEL=3` for a synchronous host stack, then the ROCm debug agent to name the kernel, then a bounds check
  (`SGLANG_DEBUG_DSA_ROWS=1`, diagnostic commit bd6a337ab7, not on the main branch) that raises before the GPU faults.
- **Not done:** the same in-place extend remains for `extend_prefix_lens_cpu` and `extend_logprob_start_lens_cpu`; no
  pre-pad reader found (coder2).
- **Mixed AgentX with DP2+DCP2 did not score:** prefill cached 64 % vs ~94 % for DCP4, per-DP cache_hit_rate 0.0;
  leading hypothesis = in-engine DP routing is not prefix-aware (option list read, not measured) (M paired AgentX).

## 10. DP2 PD decode cannot capture graphs with custom AR off

- **What:** pdE2-r3 (custom AR off) crashed at CUDA graph capture: `[PG ID 5 (attention_tp:device) Rank 1] Process group
  watchdog thread terminated ... hipErrorCapturedEvent` (M E2 r3).
- **How:** DP2+DCP2 PD decode in graph mode needs custom AR on as configured. The custom-AR-off arm of the garble
  discrimination is therefore not testable in graph mode.

## 11. KNOWN RISK: E2 intermittent garbled answers under concurrency (graph mode)

- **What:** PD with DP2+DCP2 decode, CUDA graph, sim OFF, 40-concurrent fresh-seed checks: first token(s) correct, then
  junk (`54!!!!!!!!   11,3 the0 ...`, `133 amy`). Rates: pdE2-r2 (index_share false) 3/200; pdE2-r5 (index_share
  true) 13/240; pdE2-r6 (split indexer off) 27/320. Eager gate pdE2-r1: 0/40, one round. check.py 5/5 and long prose
  coherent in all; no MAF/Traceback (M E2).
- **Why:** not established. Ruled out as necessary by experiment: index_share false (r5), split indexer (r6). Rates are
  not separable from round-to-round variance.
- **Decision:** user 10-22 ~02:05Z — recorded as a known risk, not pursued now (chat template + top_p mitigate much of it,
  per the user); no further bisection unless asked (`spec/mission.md` §0).
- **Context:**
  - Resembles the 09-18 finding: `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52-mtp-garbled-decode-rootcause.packup_20260918/`
    (custom AR on the decode leg corrupted output on TP4/DP4 with index_share false; scope correction: index_share ON
    and TP8 not covered there).
  - DCP4 (no DPA) PD decode also runs custom AR + graph: r4/r5 gates were 5 requests only; the planned 3x40 sim-off
    check on r5b was dropped by the user's decision (poll 01:18Z, 02:13Z 10-22). Mixed DCP4 with custom AR passed 40/40
    (M R10) and benchrun gates passed 40/40 (T12, T15) — single rounds each.
  - Untested arms: custom AR on + `SGLANG_DP_USE_REDUCE_SCATTER=0` + KV checksum; kv_b off; eager x6 (poll 01:58Z 10-22).

## 12. `SGLANG_OPT_USE_TOPK_V2` is not read on decode under the split knobs

- **What:** the PD vs mixed diff flagged TOPK_V2 (PD false, mixed true) as a candidate for the r2 trace's coop_topk
  9.3 ms/verify (`spec/design/pd_vs_mixed_decode_diff.md` §1 row 1).
- **Code read (analyst, poll 02:27Z 10-22):** with `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1` + `..._TRITON=1` decode uses
  `fast_topk_v2` locally + the Triton merge, and the gfx950 fused indexer for rows <= 48; TOPK_V2 is never read on
  decode. The r2 9.3 ms coop_topk was the no-split legacy path. Under r5 the top-k is estimated at ~2-2.5 ms/step
  (**estimate**). The planned TOPK_V2 A/B was cancelled.

## 13. Decode radix cache is off in PD

- **What:** PD decode `disable_radix_cache: True`, `disaggregation_decode_enable_radix_cache: False`; every turn re-sends
  the whole prefix KV from prefill to decode (`decode_prefix_len` 0) (`spec/design/pd_vs_mixed_decode_diff.md` §1 row 3;
  `spec/design/perf_knobs.md` K5).
- **Consequence:** DCP4's 3.1x KV capacity helps mixed (prefix-cache hit 93 % vs 27 %, 26,277 vs 5,130) but not PD decode
  hit rate; in PD the decode pool only needs running x context. PD decode usage 0.43-0.47 at C40 (P013, P015).
  Transfers in flight in 71 % of decode lines at C40, 89.9 % at C52 (P015).

## 14. Scoring conventions (do not mix them)

- **PD:** total tok/s / 8 chips (prefill 4 + decode 4), AgentX full (InferenceX, dataset cc-traces-weka-062126) C40,
  3600 s profiling, warmup 10/lane; warm-stack protocol: A1 on a fresh stack (not scored), `/flush_cache` both legs, A2
  scored. Same as the 22k kit.
- **Mixed:** total / 4 GPUs (InferenceX non-multinode). Comparing mixed per-GPU to PD per-chip compares different
  denominators: mixed 26,277 x 4 = 105.1K total; PD r4 A2 18,321 x 8 = 146.6K total (diff doc §0).
- **Smokes** (300-600 s, warmup 1/lane, `--unsafe-override`) are not scores; T5 split +12 % vanished on swapped halves (T6).
- **Decodebench:** cached prompts, waves N, median step ms at running == N, two pairs with GPU halves swapped; DP2 is
  per-DP step at N/2 running.
- **Simulated acceptance:** `SGLANG_SIMULATE_ACC_LEN=3.61` force-accepts drafts; perf only. Correctness checks run with
  sim off; text on a sim stack is garbled by design (P011 invalid check).
- **Run-to-run variance of temp-0 text** is not deterministic here (T2 S3), so text equality is not a 对拍 signal.

## 15. Smaller gotchas

- The checkpoint `config.json:20` sets `index_share_for_mtp_iteration: true`; the infera config overrides false and the
  knob is hard-set in `config.yihou.v0521-base.sh:228` (pdE2-r4 aborted on it). All mixed numbers ran true (T10, M).
- `dcp_replicate_q_proj=True` is inert for the GLM target with fp8 kv_b on gfx950 ("prepared full-head Q weights for 0
  MLA layers"); `SGLANG_DSA_DCP_KEEP_KV_B_BF16=1` makes it 78 layers (P016, T10).
- Custom AR also gates the aiter AR+RMSNorm fusion (analyst, poll 19:17Z); r2 had neither.
- The gfx950 fused DSA indexer only serves rows <= 48; its decline is logged once per reason at capture time, so runtime
  coverage needs py-spy (T4, T5).
- py-spy attached during graph capture coincided with RCCL "invalid device pointer" once (T11); cause not established.
- Triton CPU-interpreter tests passed for a kernel that failed to compile on GPU (constexpr reassigned in
  `tl.static_range`, T14). Gate GPU compile before scheduling A/Bs.
- Startup of a PD stack with HiCache 450 takes ~22 min for the prefill; drain after stop ~13 min (P012, P015).
- Do not use another session's InferenceX cache; private copy at `/shared_nfs/yihou_glm52/dcp_pd_20261021/cache/InferenceX`
  (poll 16:13Z, 16:32Z).
- `spec/research/critic.var_tmp_summary.md` is a byte-identical copy of `spec/research/critic.md` (both copied).

## 16. Open questions (with the measurement that would answer each)

1. **Is PD throughput transfer-bound?** P017: at C40 -> C52 decode running stays ~18.5-19 while decode transfer-in rises
   4.9 -> 18.2 and prefill in-flight 1.8 -> 3.6 per rank; per the bottleneck skill that is the transfer-bound pattern.
   **Unconfirmed.** Measure: per-transfer timing (`MC_TE_METRIC`) on the r5b config at C40 and C52 (mixrun task on v3-070,
   pending), plus the same queue readout on the 22k baseline logs as 对拍.
2. **Prefill is mostly idle in PD:** the 4 prefill GPUs computed 17.9M new tokens in the r4 A2 hour (4.9K tok/s across 4),
   510.8M prefix-cached (diff doc §0). Whether fewer prefill GPUs per decode GPU or a decode radix cache would raise the
   per-chip score is not measured. Measure: a run with a smaller prefill, or decode radix on.
3. **PD decode step vs mixed:** +1.4..+5 ms at matched running vs mixed decode-only lines (small bins, context not
   matched; diff doc §2). Measure: decodebench through the PD stack (transfer-free steady state) vs mixed T9/T16.
4. **r5b score with kvb + split-triton radix + DEBUG_AGENT=0:** pending (A1 running at 02:29Z 10-22). Expected decode
   saving ~5-9 ms/step at the PD operating point is an **estimate** derived from T16 (poll 23:43Z).
5. **E2 (PD DP2+DCP2) perf:** pending; garbling is a known risk (§11).
6. **r3 "no content" errors:** cause not established (§7). Measure: decode-only restart with custom AR off, same bench.
7. **Cancel hang fix on GPU:** run `tools/pd_cancel_repro.yihou.py` on a PD stack (§8).
8. **Mixed DCP4 prefill ~2x slower per uncached 32K chunk** (8K vs 16K tok/s, M R9). Measure: prefill trace. Relevant to
   mixed only (PD prefill has no DCP).
9. **DP2+DCP2 mixed AgentX cache-hit collapse:** prefix-blind DP routing is a hypothesis. Measure: per-DP prefix hit with
   a prefix-aware router or pinned `routed_dp_rank` per trajectory.
10. **E2 garbling mechanism** (§11): untested arms listed there; on hold per user.
11. **20k/chip not reached:** best PD r4 A2 18,321 (C40, warm). C52 gave 18,070 (P015).
12. **SDMA hang mechanism on v3 / rocm-dbgapi packet-id abort:** not investigated; workarounds in §1, §6.
