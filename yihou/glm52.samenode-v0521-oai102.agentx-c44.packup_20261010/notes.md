# Notes — gotchas, wrong turns, error analysis

Full first-hand record: `analysis/working_process.md` (R001-R027) and `analysis/poll.md`.
Each item below gives **what / why / how (fix) / context**. "First-hand" means the code was read, an
experiment was run, or a log was grepped. Everything else is marked.

## A. Faults that block v0.5.21 (all fixed in the passing chain)

1. **Prefill scheduler wedged on a blocking ZMQ PUSH (R001).**
   - **What:** in v0.5.21 the kv_events_publisher's auxiliary metrics PUSH connects to an endpoint nothing
     binds. A blocking `send` hangs the scheduler event loop: after the first `/health` 200 the prefill
     answered no more requests ("no response from detokenizer"), with no MAF, Traceback or OOM.
   - **How:** mount `patches/runtime-mounts/kv_events_publisher.yihou-noblock.py` (NOBLOCK, drop on EAGAIN).
   - **Context:** round c44a.
2. **Triton LLVM abort in absorb-bmm (R003).**
   - **What:** v0521's hand-built `_absorb_bmm_config` at M=256 aborts the triton LLVM backend.
   - **How:** mount `forward_mla_rocm.yihou-noabsorbcfg.py`, where it returns None, which is v0519 behaviour.
   - **Context:** round c44b.
3. **Decode MAF with `index_share_for_mtp_iteration=true` (H2, R004).**
   - **What:** a missing PD DSA seed causes a sustained eager DP-padded draft, and `pad_batch` does not
     pad `dsa_topk_indices`, so decode reads out of bounds.
   - **How:** `{"index_share_for_mtp_iteration":false}`. Cost ≈0.3 % (second-hand).
   - **Context:** c44c, `logs/excerpts/FAIL.c44c-MAF.txt`. v0519 history: ON faulted 5/5, OFF 13 h clean.
     **Not retested on v0.5.21** (open).
4. **Prefill MAF past host slot 6,118,144 (H1, R008/R010).**
   - **What:** sglang `_cuda_host_register` (`pool_host/common.py`) splits pools larger than
     `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB` (default 256) into separate `hipHostRegister` calls. The JIT
     kernels translate only the base pointer (`runtime.cuh:62-76`), so slots in later chunks are unmapped.
   - **How:** `SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512` on prefill, which gives one 420 GiB registration
     (validated: 50.9 s).
   - **Context:** c44d (`logs/excerpts/FAIL.c44d-MAF.txt`). e2b4 and e2c filled the pool to 100 % with
     MAF 0.
   - **Correction (R024):** the v0.5.19 oai recipe *named* this knob in a driver `EXTRA_ENV`, but that
     engine.sh ignores `EXTRA_ENV`. The "proven" r013 run never applied it (its docker cmd has 0 matches);
     its short validation just never filled past chunk 0. Lesson: diff against the **live docker cmd** of
     the proven run, not its README.
5. **Same-node IPC PID collisions (R012 → R020 → R022). This is the old "RCCL same-node race".**
   - **Why there is a collision at all:** each engine container has a private PID namespace, so a prefill
     rank and a decode rank can have the same PID number (e1a: both DP0 = 973).
   - **Default ROCr IPC collides at EXPORT.** ROCr binds the abstract socket `"\0hsa<getpid()>"`
     (`runtime.cpp:1452`) in the shared host netns. The result is `hipIpcGetMemHandle failed: invalid
     argument` during RCCL/mooncake init.
   - **Legacy IPC (`HSA_ENABLE_IPC_MODE_LEGACY=1`) collides at IMPORT.** CLR rocm-7.2.0
     `hip_memory.cpp:3357` stores `owners_process_id`, and `:3381-3382` returns `hipErrorInvalidContext`
     (**201**) if it equals the importer's PID. A PID-1/PID-1 test in two containers gave 201 in both
     modes; `--pid host` gave rc=0 in both (`results/diag`, repro).
   - **`--pid host` works on plain-ssh nodes (136) but is DENIED on spur nodes:** `spur-authz:
     denied [N1]: --pid=host` (`logs/excerpts/FAIL.e2b3-pidhost-denied.txt`).
   - **Fix used:** `DECODE_PID_OFFSET=30000`. The decode container forks and exits 30000 throwaway
     subshells (~8-13 s) before `exec`, so forked rank PIDs are > 30000 and disjoint from prefill
     (≈670-1420). The census gate checks this.
   - **Wrong turns:**
     - legacy on decode only (the mode must match on both legs);
     - suspecting the ROCm debug agent (excluded: 201 with the agent off);
     - suspecting the v0.5.21 HIP runtime (refuted: libs byte-identical across images).
     - v0.5.19 passed only because its PID layout happened not to collide.
   - **Upstream:** an issue draft, NOT posted, is in `scripts/diag/rocm-clr-pid-guard-issue.yihou.md`.
6. **Ephemeral-port collision (R015).**
   - **What:** the second leg's fixed derived ports (base + ZMQ delta 233 …) landed in the first leg's
     ephemeral range (32768-60999), giving `Address already in use`.
   - **How:** all fixed ports < 30000 (etcd 24501/24502, router 28100, engines 29101/29357, bootstrap 28198,
     kv-events 25657, snapshot 28901).
7. **Mooncake HIP IPC interior-registration misplacement (R018/R019).**
   - **What:** HIP opens an interior pointer's handle at the allocation base. faae8dd4 relocated with the
     registered pointer, so KV landed (ptr − base) bytes off, silently.
   - **How:** image with Mooncake PR #4340 (e8062249, ported to faae8dd4).
   - **Context:** the GPU te_pair test shows the interior region correct only with #4340. In all passing
     rounds REGBASE reported `interior=0`, so this config does not exercise it; keep the fix anyway.

## B. Harness and ops gotchas

- **Smoke text is garbled by design.** `SGLANG_SIMULATE_ACC_LEN=3.61` forces the accept count, not the
  tokens. This matches the 0923 AgentX reference. Smoke proves the transfer path, not correctness. Set
  `DECODE_SIMULATE_ACC_LEN=` (empty) for real acceptance; r013 ran that way.
- **InferenceX symlink (R023).** `$WS/cache/InferenceX` was a symlink into another tree, which the client
  container could not follow (`benchmark_lib.sh: No such file`), and the bench failed in 55 s. Fixed by
  `INFERENCEX_DIR=$(readlink -f …)`.
- **Slow VRAM release.**
  - **What:** after a HiCache-450 round, GPUs hold ~250-270 GB with no KFD process for 20-30 min. After
    e1c (154 GB pool) release took < 5 min. New KFD opens can block in D state meanwhile.
  - **How:** gate on all GPUs idle twice, 60 s apart.
  - **Hypothesis only (unverified):** teardown cost scales with USERPTR-registered host memory. The
    PR #42553 GTT smoke freed 150 GB in 0.5 s.
- **Shared nodes loop jobs.** On 136 another user's container re-launched an 8-GPU bench every few
  minutes. A single "GPUs idle" reading was a gap: e1b launched into it and had to be stopped. Hence the
  two-reading gate, and 136 was abandoned for 102 (an oai allocation).
- **Spur jobs.**
  - Hold scripts must `exec sleep infinity`. A fixed `sleep 86400` ended job 200276 despite a raised
    TimeLimit (round c44e lost).
  - Burst QOS: priority 100, and jobs were preempted after ~20 min.
- **Engine/harness editing.** Editing a running bash pipeline script crashes it (bash reads lazily). Edit a
  copy.
- **Team coordination.** A teammate stuck in long blocking poll loops missed two leader messages. Keep
  waits ≤ 5 min.
- **Census pitfall.** Health-check `TimeoutError` tracebacks during bring-up are benign. The census
  counts them separately.

## C. Performance observations (single samples, not significance-tested)

- e2c vs e2b4 (fused indexer ON vs OFF, otherwise identical): total throughput +0.9 %, TTFT p50 5.41 → 4.63 s,
  ITL p50 14.0 → 14.7 ms.
- e2b4 vs e1c (C44 with a 450 GB pool and decode 0.95, vs C40 with 154 GB and 0.85): total −3.3 %,
  TTFT p50 2.85 → 5.41 s. Two variables changed at once, so the effect is not attributed.
- vs the v0.5.19 reference (0923 c40-full, node 137, `index_share` ON): 168,260 tok/s. e2c is 164,194 at
  C44 and e1c is 168,389 at C40. These are not like-for-like.
- **Warmup first wave (open).** The first 48 cold first-turn requests gate prefix reuse. e2b4's first
  wave took 836 s versus 649 s (e2c), 595 s (c44d) and 565 s (e1c), with prefill idle 350-450 s per
  rank and nothing queued (`results/analysis/firstwave.yihou.txt`, `router-firstwave.yihou.txt`). Router
  dispatch and DP balance were the same in all runs. Decode admission state is unlogged during warmup
  (max_tokens=1), so the cause is unknown. A future run should poll the decode `/metrics` prealloc and
  queue gauges.

## D. Open items

- `index_share=true` on v0.5.21 (H2), with ≥2 long replicates if it is ever to be re-enabled.
- Replicates of e2c, to measure run-to-run variance.
- A real-acceptance AgentX run.
- PR #42553 (GTT host pool) A/B at hicache ≤ 280 GB (`patches/optional-pr42553`, R021).
- The first-wave starvation in e2b4 (C above).
