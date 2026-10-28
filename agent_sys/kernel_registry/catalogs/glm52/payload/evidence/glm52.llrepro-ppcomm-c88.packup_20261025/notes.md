> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Notes

1. **PP comm overlap** (`SGLANG_PP_COMM_OVERLAP=1`, prefill only, `scheduler_pp_mixin.py:584-599` puts PP tensor
   send/recv on a dedicated stream): the clearest gain of this pack-up, seen in all three C80 runs (TTFT p90, prefill
   batch rate, decode admission waits). The kit needed a prefill env hook (`PREFILL_ENV_EXTRA`, `worker.sh:126-127`).
2. **`SGLANG_PP_SKIP_PURE_CHUNKED_OUTPUT_COMM=1` breaks prefill MTP**: PP3 asserts `batch.spec_info is
   result.next_draft_input` (`prefill.py:896`). INFERENCE: the EAGLE draft input travels with the PP outputs.
3. **Chunk 4096**: iteration period -37 %, kv_transfer_latency p50 -23 % (iterations per release 3.0 -> 3.7), TTFT p50
   better but prefill queue p90 3.6 s; score neutral. The ~1.5 s "transfer latency" is mostly the prefill-side PP
   release consensus (`prefill.py:1029` -> `:1199`); decode completion is polled independently (INFERENCE from code).
4. **Decode allocation policy `prefill_complete`** is rejected for pp > 1 (`pd_disaggregation_hook.py:198-243`).
5. **Verify-graph kernel breakdown still missing**: torch profiler records no kernels inside the verify graph; graph-off
   eager decode crashes at start-up in this config (`topk.hip:184`); `rocprofv3 --attach` (rocprofiler-sdk 1.1.0) wrote
   nothing. The kernel session has no working method either (10-25).
6. **KF fused indexer** (`INFERA_KERNEL_FUSED_INDEXER`) targets `sglang.srt.layers.attention.dsa.gfx950_fused_indexer`,
   which exists only in the KF session's image; here it is a no-op. Kernel gain 0.1036 -> 0.0762 ms/call; estimated
   e2e ~1 %.
7. **C88** raises the score +1.9 % but decode KV saturates (mean 0.85, prealloc waits 1/3 of the time) and TTFT p90
   returns to ~12 s. C80 stays the comparison point (contributor's numbers are C80).
8. **NFS**: `REDACTED_LOCAL_PATH` (420 TB, cluster-wide) hit 100 % on 10-25 ~07:30Z (AgentX "Disk quota exceeded"). We removed
   18 leftover aiperf tmp dataset copies (~119 GB) of ours; it refilled from other users. Runs since llr21r use
   WORK_DIR=REDACTED_LOCAL_PATH on REDACTED_NODE and rsync compact results to NFS.
9. **Run-to-run spread** at C80: llr14 config +-0.8 % (3 runs), llr21 config +-0.9 % (3 runs).
