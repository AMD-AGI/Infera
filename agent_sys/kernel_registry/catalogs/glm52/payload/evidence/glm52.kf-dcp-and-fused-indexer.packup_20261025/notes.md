> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Notes - what / why / how / context

## 1. Overlay additions (same mechanism as the 10-23 pack-up, notes 1 there)
- DCP sparse: wraps `kernels/ops/attention/dsa/triton_sparse_mla_decode.triton_sparse_mla_decode_splitk` when
  return_lse and lengths are given (the `_forward_triton_dcp` call), 64 heads, d_v 512. The KF launcher keeps its own
  split-K buffers (workspace=None) because its layout may differ from the backend's.
- DCP indexer: replaces `Indexer._dcp_split_decode_topk` for TARGET_VERIFY: sglang's `owned_inputs`, KF logits over
  the owned pages, KF local top-2048 + pack, the REAL `dcp_group.all_gather`, KF merge + slot transform.
- Fused indexer: replaces `Gfx950FusedIndexer.run` (bf16 weight path), reusing the production workspace object.
- Capture: `INFERA_KERNEL_DUMP_DCP_IDX=<dir>` saves the index-K pool of one layer + every N-th TARGET_VERIFY call's
  (q, weights, lens, block_tables) on DCP rank 0. Python hooks do not run under CUDA-graph replay, so the capture round
  runs the decode with `--disable-cuda-graph` (harness knob `KF_DUMP=1`).
- Harness knobs (`config.registry.kf.sh`): `KF_DCP`, `KF_DCP_SPARSE`, `KF_DCP_INDEXER`, `KF_FUSED`, `KF_DUMP`
  (+ `KF_DUMP_SKIP/EVERY/MAX/LAYER`); with `DCPX_DECODE_EC=1` the overlay row is merged into the ec-decode mount tsv.

## 2. Lesson: synthetic KF datasets can hide production behaviour
The DCP indexer won 2.0-2.7x on the driver's synthetic logits and lost in-model; real inputs showed a selector cap
overflow and non-exact rows. A KF task needs real captured inputs (or a validation pass on them) before its kernel is
trusted in e2e. The DCP tasks here still use synthetic inputs.

## 3. Traps hit
- KF runs stopped by "ORCHESTRATION FAILED repeatedly; stopping after 3 consecutive infrastructure errors" (specialist
  infrastructure failed / 1800 s orchestration timeout) exit 0, so the auto-resume watcher treated them as finished.
  `node/campaign.registry.sh` now takes `FORCE_RESUME=1`.
- The in-container watchdog killed both DCP campaigns once (10-23 15:48Z): it read the guard's heartbeat file between
  truncate and write (empty -> age 1.79e9 s). Watchdog re-reads non-numeric values; guard now writes tmp + mv.
- sglang `load_jit` publishes builds with rename(), refused on the workspace-node/compute-cluster NFS home -> `SGLANG_JIT_CACHE_DIR`
  must be node-local (kf-env + the fused-indexer driver set it).
- remote-launcher 0.15 (10-24): `remote-launcher exec` runs as the user, not root. Docker works only where /var/run/docker.sock is 0666
  (compute-node-02/192 yes, m2m-185 no - dropped).
- A launcher loop whose remote-launcher exec hung for minutes later fired its second iteration and started a duplicate kdg1 chain;
  it stopped at the KFD gate but overwrote `kdg1.chain.log` (status.tsv of the real run is intact).
- kde1: the stack was stopped (node end) before aiperf wrote its JSON; TPOT recomputed from
  `profile_export.jsonl` (`inter_token_latency`, profiling phase) - the same method gives 19.43 for kda1 (JSON 19.44).

## 4. Open
- DCP indexer: why 1-3 rows differ from production on real data; a bounded selector worst case.
- Which kernels now dominate the DPA decode iteration after the KF kernels (no per-kernel profile has been taken; the
  fused-indexer choice was made from step-count share, not time share, and turned out small).
- EP8 (user: try when a machine is found); EP4 not pursued (user).
