# Notes — gotchas, wrong turns, open questions (what / why / how / context)

Ledger references (R0xx) are in `analysis/working_process.md`. Bring-up failures of the v0.5.21 stack (HiCache chunk
register H1, index_share H2, PID collisions, ports, Mooncake #4340, kv-events PUSH, absorb-bmm config) are documented
in the phase-2 pack-up `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.samenode-v0521-oai102.agentx-c44.packup_20261010/notes.md`.

## 1. The first AgentX run on a fresh stack is ~10 % slower (the warm-stack protocol)

- **What:** first run on a fresh stack 19,335-20,371/chip (A1, e2e, e2f, e2i R1, e2j R1; e2b4/e2c-C44 at C44); second run on
  the same stack after `/flush_cache` 21,696-22,114 (e2c C40, e2g A2, e2i R2). Exception: e2d (fresh, index_share TRUE)
  21,383 with no penalty.
- **Why (what is known):** C40 throughput is bound by decode step cost (decode busy 93-96 %, R042). At matched load
  (median step by busiest-DP running-req) fresh runs are +7-12 % slower (R046, R049) with a rise-peak-decay over the hour
  (worst at profiling minutes 10-20). Not explained by: config deltas (A1 = exact replicate of 21,696), time of day
  (A2 fast at 07:31 UTC; R049 supersedes R047), CPU placement/NUMA (R046), clocks/power/thermal (R044, R055), runtime
  JIT/autotune (all steps replay CUDA graphs, R052 cont.), the ROCm debug agent (e2i R1 slow, R052), NFS/logging I/O (R047).
  py-spy: the extra wait sits in the verify-prep `.item()` sync, i.e. GPU work after the previous verify (draft phase) is
  longer (R052). Kineto (e2j R1, fresh): **gap A ≈ 8.5-10 ms/step with the whole GPU idle** — the forward stream waits
  for a schedule-stream dependency that has already completed (R054, R055). HW queues: 4 compute HQDs per decode process,
  all streams multiplexed (CLR `GPU_MAX_HW_QUEUES=4`), `ROC_CPU_WAIT_FOR_SIGNAL=1` default.
- **How to handle:** score the second run (REPRODUCE §4). Do not restart the stack between warm-up and scored run.
  Candidate fixes to test (not done at pack-up time): decode `GPU_MAX_HW_QUEUES=1/2/16` (not 32 > 24 HQDs),
  `ROC_CPU_WAIT_FOR_SIGNAL=0`, `HSA_ENABLE_INTERRUPT=0`; microbench `scripts/diag/bprime/run_xstream_wait2.yihou.sh`.
- **Context:** the v0.5.19 reference (21,032) was a fresh-stack run with index_share TRUE; fresh-vs-fresh with
  index_share TRUE, v0.5.21 e2d = 21,383 (+1.7 %).

## 2. Wrong turns (kept so nobody repeats them)

- **"chunked-prefill 65536 costs −9.8 %" (R037) — withdrawn (R040):** e2f with chunk 32768 was equally low; both were
  first runs.
- **"Only 16-19 UTC is fast" (R047) — superseded (R049):** A2 was fast at 07:31 UTC; stack history is the separator.
- **KV-transfer levers (R032-R039):** the transfer HOL over-wait (R036) and queue sharding were real and fixed in e2f
  (TTFT best), but the score is decode-bound, so they did not raise tok/s. Mooncake high-priority copy streams fix copy
  stalls behind long kernels in a microbench (R043) — not production-tested.
- **NUMA cpusets (R045):** scheduler pinning is wrong (base image `SGLANG_SET_CPU_AFFINITY=1` pins by local gpu_id; decode
  DPs pinned to the wrong NUMA node, identical CPU sets for prefill and decode), but placement does not track the penalty
  (R046). Knobs exist (engine.sh), round e2h prepared, never run.
- **index_share TRUE:** no throughput gain at C40 (e2d 21,383); MAF history at C44 (H2) → kept false.
- **cuda-graph bs 1..16 + host-variant (e2j):** both verified active (R054); fresh R1 19,335 (first-run class); warm R2 21,681 (R057) — no measurable gain
  (run 2026-10-12 03:56-05:15Z, `results/e2j/R2-warm/`).

## 3. Operational gotchas

- **VRAM drain after stopping a HiCache-450 stack takes ~28-35 min** with no KFD process left (driver-side). Once it
  stayed stuck until the spur job boundary (05:26 2026-10-11). The harness gate waits for < 400 MiB on all GPUs.
- **Hold scripts:** `sleep 86400` holds end at 24 h even after `scontrol update TimeLimit=` (lost c44e and nearly e2g);
  use `scripts/alloc/hold_inf.yihou.sh` (`exec sleep infinity`) and extend TimeLimit; queue follow-on holds with
  `--dependency=afterany:<job>`.
- **Login host loss:** drivers run on the login host (`pipeline.yihou.sh`, chain scripts). When `crsuse2-slog-005` died,
  e2j's R2 driver died before launching the client. Always `setsid nohup … < /dev/null &`; the AgentX client itself runs
  in a node-side container writing to NFS.
- **Do not edit a running bash script in place** (bash reads incrementally; killed the A1 sampler) — write a new file and `mv`.
- **Transient prefill `/health` 503** "couldn't get a response from detokenizer for 20 s" at bring-up (e2f, e2i) — benign.
- **Host HiCache can reach cpu_kv_usage 100 %** near the end of an hour; one TTFT burst (~30 requests at ~80 s) in e2i R1
  at 88→94 % — first sighting, not reproduced in A2/e2i R2.
- **Simulated acceptance:** `SGLANG_SIMULATE_ACC_LEN=3.61` → smoke output is garbled by design; MTP shape changes are not
  legitimate levers.

## 4. Open questions

1. Mechanism of the fresh-stack decode penalty (gap A): HWS/CP dependency-resolution latency vs host-side signal wait;
   why it disappears after ~75k decode steps and survives `/flush_cache`. Warm kineto trace + event-wait microbench pending.
2. (Answered, R057: e2j levers add nothing measurable in the warm state, 21,681.)
3. Run-to-run spread is measured only with 4 warm runs (−0.8…+1.2 %).
