# Notes — findings, traps, open questions (what / why / how / context)

## 1. The baseline is a 3-node band, not a number
- **What**: C56 A2 21814 (097), 21538 (268), 20516 (054); mean 21289, -3.6 / +2.5 %. 054's A2 was below its own A1 (21291).
- **Why it matters**: any single-run same-node delta inside ±3 % is not a result. prefetch step 2 (21342) and the shim run
  (21280) sit inside the band.
- **Context**: all three are 0-MAF full A1+A2 runs with index_share=true + idlefix, i.e. the fix also held on 268 and 054.

## 2. CONC 52 same-node: +9.8 % (n=1, 268)
- A1 23048 / A2 23367. transfer_kv p50 7.5 -> 4.3 s, prefill queue avg ~9.5 -> 5.4 s, GPU hit 0.89 -> 0.91-0.93.
- The C48 (268) and C52 replicate (054) rounds were stopped mid-A1 when the user lent both OAI nodes to another session; no score.

## 3. chunk 65536 (054): A1 = baseline, A2 crashed
- 07:56:58Z, ~60 min into A2: one prefill rank `HSA_STATUS_ERROR_OUT_OF_RESOURCES` (queue abort while `ncclDevKernel_Generic`
  ran), then the other prefill ranks lost their gloo peers. No Memory access fault. Cause NOT established; one run.

## 4. Cross-node 1P1D: what was done to make it work first time
- **Rails**: ionic rails are isolated; on 09-18 (v0.5.19, 135<->138) auto NIC choice made GPUs 4-7 unreachable. Here both legs use
  GPU0-3 with the per-GPU map `{"0":"ionic_0",...,"3":"ionic_3"}`, `MC_TE_FILTERS=ionic_0,...,ionic_3`, `PD_DP_RANK_AFFINITY=1`
  (prefill DP i -> decode DP i). Census: 4 x `Found 1 HCAs`, 4 rdma installs per leg.
- **HIP vs RDMA**: Mooncake picks rdma for a cross-host target by itself (`multi_transport.cpp`, `isHipReachableTarget`), so
  `MC_DISABLE_HIP` stays absent as in the base (the audit requires it absent).
- **Image ID**: docker load on 135/136 re-IDs the image to e8e2838f; engine.sh keys the aiter JIT cache by the local ID, so the
  seeded cache had to be copied under that key (otherwise cold JIT; a partial cache once caused the PD warm-up hang).
- **Disk**: 136 was unreachable for days (10-15..10-19) after a full disk. Everything is written to NFS (`NODE_RW_ROOT=`);
  chain_x refuses to start unless docker's root is on the nvme LV and `/` < 92 %, and a 10-min watcher stops the round at 95 %.
  135's `/` was 87 % throughout (not ours).
- **Rail probe**: both ends pinned to ionic_k, every GPU reaches the peer; local-NUMA GPUs ~38.3-38.6 GB/s, cross-NUMA
  27.5-29.6 GB/s; CPU-DRAM rdma 8.3-11.8 GB/s; tcp 0.09 GB/s.

## 5. Cross-node results and how far they go
- C56: G4 A1 24976/24398, A2 24362/24896; G3 A1 25524, A2 25376. Both sweeps peak at C56 (G4 C60 23056, C64 22670;
  G3 C60 23905, C64 22860). G3 > G4 by 2-4 % at C56/C60 and 0.8 % at C64, one G3 bring-up only.
- Transfer: transfer_kv p50 ~1.1 s (same-node ~7.5 s); per call 20.1 GiB/s vs 2.62 GiB/s; queue wait 0.34 vs 5.43 s.
- Raising CONC: transfer 1.1 -> 1.3 s only; prefill queue avg 7.7-8.7 -> 14-16 s and GPU hit 0.85-0.87 -> 0.75-0.81 move.
- Caveat: cross-node ISL mean 120-132k vs same-node 136-141k (fewer long requests completed in the window) — not normalised;
  G3 raises prefill KV tokens 2,938,752 -> 3,092,288 (+5.2 %).

## 6. Cross-node TPOT 15.7 ms vs 12.5 ms (report `results/xnode/tpot_analysis.20261021.md`)
- index_share ON in g4x1 (decode /get_server_info `json_model_override_args`).
- Binned by the busiest DP rank's running requests, per-step cost is the same cross-node and same-node; the same-node curve
  predicts g4x1 TPOT within ~0.1 ms. Cross-node decodes more requests at once (4-DP running 23 vs 15; in-flight ~40 both).
- Hypothesis, untested: faster transfer -> fewer requests waiting for KV -> bigger decode batches -> higher TPOT.
- Code read: every newly admitted PD request is seedless (prefill runs without EAGLE), so the next draft-decode step runs eager on
  all DP ranks; this is invisible in the log (`cuda graph: False` counts target-verify only). Upper bound of such steps
  12.5 % vs 8.6 %; cost not measured.
- index_share gave no measurable per-step gain at this granularity (idx=false g4n3 A2 sits on the same curve).

## 7. Same-node transfer: hipxfer shim (g4h1, whole run)
- Mooncake HipTransport::submitTransfer = enqueue all hipMemcpyAsync(kind Default) + hipEventRecord, then hipEventSynchronize
  each (synchronous per batch; pybind releases the GIL). The shim times exactly that, plus its own events around each copy.
- 40641 batches, 74053 GiB: wall 38903 s (1.90 GiB/s); host enqueue 770 s + gap 351 s; sync wait 31750 s; GPU span 23230 s;
  per-copy GPU time sum 38486 s. Batches > 1 GB: wall/GPU-span median 1.46, per-copy rate median 6.35 GiB/s.
- First 25 min alone gave wall/span 1.02 — the whole run differs; do not quote the early number.
- Open: why copies run at ~6 GiB/s (idle microbench 57-63 GB/s) and what the ~1/3 of wall outside the GPU span is
  (copies start late or completion seen late). Next: SDMA vs blit for these copies; contention with HiCache / offloader H2D/D2H.

## 8. Operational traps seen 10-21
- `spur exec` often returns "transport error" yet the command ran: verify with a second `spur exec` before re-launching.
- A `pgrep -f <name>` guard inside `bash -c "..."` matches its own command line; anchor it (`^bash /shared_nfs/...`).
- An apostrophe inside a single-quoted `ssh host '...'` command string (e.g. in heredoc ledger text) breaks the quoting.
- A workflow subagent wrote `spur006:/tmp/xnode_load.awk` outside a yihou dir; not deleted (awaiting the user).
- Stopped containers of the cross-node rounds remain on 135/136 (docker stop only); `docker rm` awaits the user.
