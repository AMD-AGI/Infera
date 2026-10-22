# Notes — g4tune (what / why / how / context)

Source of record: `analysis/ledger/poll.md` (snapshot of the ledger at 2026-10-20 01:19Z; g4tune entries start at "2026-10-19 05:2x UTC - NEW TASK").
Statements below are measurements or code reads; where something comes only from the user/brief it says so.

## 1. Inflight stage (work item 1) — what was established before any run

- **Definition (code read at image commit 11b1fb21de, `/tmp/yihou/sgl_11b1` on the workstation):** prefill.py L1054-1056 `send_kv_chunk(last_chunk)` then `set_prefill_transfer_queue_entry_time`; L1183-1197 poll Success → `release_kv_cache` + `set_prefill_kv_transfer_finish_time` = sglang stage `prefill_transfer_kv_cache`.
- **Measured (pre-g4tune runs, `server_metrics_export.csv`):** `prefill_transfer_kv_cache` avg pf90 C48 4.9 s, G2 C56 11.6, G3 C56 9.9, G2 C64 18.6, G4 C64 15.8; `kv_transfer_total_mb` avg 7.3–8.1 GB/request at every conc.
- **Research workflow (code + data, `analysis/research_wf.20261019.json`), verified items:** the whole context is sent per request (decode radix cache off, rejected with EAGLE); per prefill DP process the sender is `SGLANG_DISAGGREGATION_QUEUE_SIZE`=4 FIFO queues × 1 thread, queue = sum(decode session ports) % 4, one blocking `batch_transfer_sync` per chunk; HIP transport = `hipMemcpyAsync` per block over 64 streams then serial event sync; `MC_DISABLE_HIP_TRANSPORT` / `MOONCAKE_DISABLE_HIP_DMABUF` in our env are read nowhere (no-ops); `MC_TE_FILTERS=yihou_no_such_rdma_device` → tcp+hip installed (code read; runtime route unverified). Mooncake has no per-transfer timing log → instrumented `conn.py` (patches 0001/0002).

## 2. R1 dest vs R2 mix (one variable: `YIHOU_XFER_SHARD`)

- **What:** R1 g4xb5 (dest = upstream) and R2 g4xm4 (mix) on the same OAI node m2m-250, back to back; R2 also on burst v3-017 (g4xm3).
- **Result:** A2 21,262 (dest) vs 22,854 (mix, 250) / 22,495 (mix, v3-017). Transfer stage avg 9.98 → 2.16 s; prefill_forward avg 4.18 → 4.62 s; TPOT 13.3 → 15.0 ms; decode alloc p90 574 → 1.2 ms. Full numbers: `results/results.csv`, `results/hist.g4tune.jsonl`.
- **YIHOU-XFER (`results/xfer_sum.g4xb5_g4xm4.txt`, recomputed for this kit with `scripts/session/xfer_sum.yihou.py`):** dest — every DP uses 1 shard; after warm-up per-DP 5-min rates mostly 1–4 GiB/s (0.2–4.2), worker busy mostly 50–100 % (22–100 %), queue wait mean up to 25.8 s. mix — 4 shards per DP, wait mean 0.07–0.57 s in the windows recorded (server-log copy stops at 19:35Z because of EDQUOT, so mix covers only ~30 min).
- **Decision (user, 2026-10-20, relayed in the task brief; not in the ledger snapshot):** mix is **not adopted** — it makes prefill and decode interfere and hampers analysis. dest stays the baseline for further rounds.
- **Caveat:** n = 1 per point. Warm-up rates (first ~10 min) are ~0.2–0.3 GiB/s/DP in both modes.

## 3. Larger chunk (work item 2)

- **What:** `PREFILL_CHUNK_SIZE` 32768 → 65536. Code read (image): DP attention divides `chunked_prefill_size` by `attn_dp_size` (parallel_hook.py:310) → per DP 8192 → 16384, which equals `max_prefill_tokens` (PrefillAdder budget).
- **g4xc1 (m2m-097):** census confirms `chunked_prefill_size=16384`. Prefill log (`results/rounds/g4xc1/server-logs/prefill-0.log`): first two 16384-token batches at 19:23:47 and 19:23:51, then 19:26:38, 19:29:52, 19:33:08 (~3m15s apart); for comparison g4xb5 runs 8192-token steps ~1.3–1.5 s apart. Ledger: GPU0-3 100 % busy, GPU4-7 0 %; py-spy ×3 of scheduler_DP0: MainThread in `torch.synchronize` ← `process_batch_result_disagg_prefill` (prefill.py:962); warm-up returned 9/619 after 1680 s. Stopped ~19:53Z. **Cause not measured** (kernel-level profile needed). The py-spy dumps were not found in `runs/g4xc1/pyspy/` (empty) — location unknown.
- **User (2026-10-20, from brief/env header):** chunk changes worked on spur before; rerun on another machine → **g4xc2 on m2m-250, started 2026-10-20 01:27:27Z, in progress** (status at kit build: `start`, `ports-all ok`).

## 4. Machine suspicion — microbenchmarks (v3-021, job 225141; `results/machine/`)

- Mooncake HIP transport, 360 KiB × 5825 blocks = 2 GiB per `batch_transfer_sync_write`: idle 0.042 s/call ≈ 47–51 GB/s for one pair; 4 concurrent pairs 0.107 s each.
- 1 × 2 GiB block ~9 GB/s; 64 KiB blocks ~13.6 GB/s.
- bf16 matmul burn on the **sender** GPU with ≥ 2 pairs: senders on GPU1/2/3 5.7–6.0 s/call (0.36–0.4 GB/s, ~50× slower); GPU0 stays 0.11–0.13 s (reproduced 4×, also with nbuf=1). Burn on the **receiver**: no slowdown.
- `MC_HIP_STREAM_PRIORITY=high`, `HSA_ENABLE_SDMA=1`: no change. `HSA_ENABLE_SDMA=0` under sender load: senders did not finish within 600 s; container stopped.
- torch peer copy (image): 59.7 GB/s for one 4 GiB copy GPU0-3 → GPU4-7; 12.9–13.5 GB/s for 4 GiB as 10000 blocks.
- `xgmi_load` (m2m-097, 2026-10-20 01:28Z, **not yet in the ledger**): all 4 pairs at once; no load: push_1 ~36–39 GB/s, push_blk ~11.4–12.4, pull_blk ~10.2–10.9 GB/s; with sender burn: push_1 23–27 GB/s, push_blk 2.9 GB/s (third repetition 1.8 s vs 0.19 s), pull_blk 1.17 GB/s. Different node from the mc_bw runs.
- **Mechanism not established.** Open: why GPU0 is unaffected; whether production prefill compute on GPU0-3 explains the 1–4 GiB/s per DP seen in R1.

## 5. Operations log (what broke, how it was handled)

- **Burst preemption:** g4xb1 (bring-up), g4xb2 (A1 warm-up), g4xb3 (7 min into A1), g4xm1 (bring-up), g4xm2 (22 min into A1), g4xb4 (~66 min into A1) lost. After preemption, prefill/decode containers can stay `Up` with dead engines; they were docker-stopped (never rm). A burst job that ends at exactly 20:00 is the partition `PreemptExemptTime=1200`.
- **v3-026** was granted with docker.service failed → unusable; reported to the pool.
- **/shared_nfs EDQUOT ~19:35Z 10-19:** g4xm4 A1 bench wrapper failed (`tee: Disk quota exceeded`), chain and server-log copies stopped (docker logs intact). The orphaned A1 client container ran to completion (20:32:10Z; aiperf artifacts written, no `agentx_conc56.json`). `scripts/session/a2_after_orphan.yihou.sh` waited for it, flushed both legs and ran A2 (scored 22,854). g4xm4 A1 histograms come from its CSV; it has no score.
- **OAI holds:** self-submitted 224693/224694 went to prolog failure and were cancelled; nodes were then obtained through the machine-pool target (m2m-250, m2m-097).
- **Microbench leftovers:** `docker run --rm --name yihou-mcbw5/6` later reported a name conflict, i.e. containers with those names existed on v3-021 at the time; whether they were cleaned up is not recorded.
- spur006 hung once on a recursive grep over /shared_nfs (10-19 06:2x); avoid recursive scans from login hosts.

## 6. Gaps

- No `collect_env` on g4tune nodes; no ECC capture.
- py-spy dumps for g4xc1 not located; `runs/<round>/pyspy/` dirs are empty for g4xb5/g4xc1/g4xm4.
- Exact docker commands for each microbench log other than `run_sdma.yihou.sh` are not recorded.
- Server logs > 4 MB and aiperf `profile_export.jsonl` / `server_metrics_export.json` are not copied (`results/not_copied.tsv`). The g4xm4 decode server-log copy stops at 19:35Z (EDQUOT); full logs exist only in docker on m2m-250 while the containers exist.
- The user decision on mix and the g4xc2 launch post-date the ledger snapshot (01:19Z).
