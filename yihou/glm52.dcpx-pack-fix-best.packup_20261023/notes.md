# Notes - what / why / how / context

1. **Transfer, not wire.** dcpx6/7b: NIC tx 2.5-3.9 GB/s per rail (~38 GB/s capable), rails balanced, 135 CPUs 4-19 %
   busy. Transfer workers sat at ~100 % CPU spinning in `getBatchTransferStatus`; that spin is waiting, not work.
2. **Instrument the boundary.** `[yihou-xfer]` (log-only wrappers on `_transfer_data`, `send_kvcache_dcp`,
   `try_pack_dcp_src`) showed pack = 91 % of send time; sub-timers then split pack into index / launch / sync. The
   idle-GPU kernel test (0.66 ms) proved the rest was queueing.
3. **Wrong turn:** `bootstrap_room % 4` (dcpx7) = prefill DP rank; py-spy showed exactly one worker per rank, a
   different one per rank.
4. **Harness traps hit:** prefill MTP failed the live audit (`speculative_algorithm` expected None) and `agentx_env`
   (`live mtp=True, config expects False`) - both fixed; stopped containers block re-use of a round name (dcpx8 ->
   dcpx8b -> dcpx8c); an orchestrator must also watch the chain log for `STOP` (status.tsv has no fail line then).
5. **Prefill MTP code read** (19-agent verified workflow): EAGLE on a PD prefill loads NextN + draft KV pool (not offloaded
   by OffloaderV2), one extra draft-extend forward per chunk, HiCache packs the draft as an extra host layer; under DCP
   the draft KV is sharded like target (+1/78 bytes); no assert against prefill/OffloaderV2/HiCache/DPA.
6. **Still open:** DCP vs DPA TPOT (18.4 vs 16.7 ms) on the same stack; which half of the pack fix matters; prefill
   queue p90 ~14-16 s is now the next limit; correctness of the no-wait pack under a PP prefill.
