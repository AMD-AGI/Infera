# Core PR audit after main updates

Audited on November 5 against upstream main [`c07ec416`](https://github.com/sgl-project/sglang/commit/c07ec416dabfae09b7dea24d5dd5ef84d7d1f5a7). The original fixes in all three PRs remain effective in their intended paths; no rebase-induced semantic drift was found. A separate, pre-existing HIP indexer synchronization remains outside #46830's metadata-only fix. The PR source heads were preserved throughout this audit.

| PR | Published head / main base | Latest-main integration and controls | Disposition |
|---|---|---|---|
| [#33059](https://github.com/sgl-project/sglang/pull/33059): Aiter real-row/padding contract | `31d19a218f` / `732d34b93c` | Published and isolated integrated sources each pass 12 CPU cases. Unpatched main fails 10 padded/empty cases and passes two unpadded controls. | Fix remains reachable and needed. Updated stale PR-description source information. |
| [#46820](https://github.com/sgl-project/sglang/pull/46820): nonblocking auxiliary metrics | `d796f86571` / `732d34b93c` | Integrated sources pass four real-ZMQ cases in each of msgpack and pickle modes. Behavioral main controls fail two backpressure cases in each mode. | Fix remains on the scheduler path. Preserve the production-configuration qualification below. |
| [#46830](https://github.com/sgl-project/sglang/pull/46830): avoid eager metadata host synchronization | `a349c3b9fe` / `f64efd63fd` | Published head passes three tests plus two subtests. Latest main plus the PR passes six tests plus two subtests, including three upstream metadata-hook cases. Unpatched and width-only controls distinguish both fixes. | Original fixes remain effective. Track the adjacent HIP indexer fallback separately; no claim that all eager paths are synchronization-free. |

Main advanced seven commits beyond the #33059/#46820 base and sixteen beyond the #46830 base. Integrations were made only in independent audit checkouts. Passing these local checks does not replace the current-head Base runs required by maintainers.

## #33059: row correction and source provenance

HIP still resolves to Aiter, and decode, target-verify and draft-extend still reach `_get_topk_paged`. Aiter allocates logits from query rows and does not accept a separate real-row offset; top-k uses real expanded-length metadata. Trimming query/weight rows and restoring physical output rows with `-1` therefore still implements the required contract.

Both commits replay unchanged onto the pinned latest main. The patched method's AST also equals the method at GPU-validated head `ab4407a57c`. The latest unpatched method equals its historical baseline `23c6442cde`; upstream has not independently incorporated the correction. The zero-real-row guard remains defensive: the historical GPU traces did not establish natural execution of that case.

The PR description still referred to intermediate base `1599b166ac` and an older unchanged-surrounding-files interval. It has been corrected to identify current head/base, the latest-main CPU audit, and historical GPU source separately. Surrounding Kimi/import/hook code has changed since the GPU runs; method equivalence does not qualify a new complete GPU stack.

Sources: [published correction](https://github.com/sgl-project/sglang/blob/31d19a218f1c8927c06965e3a9eb4053759516e6/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L1431), [HIP backend selection](https://github.com/sgl-project/sglang/blob/c07ec416dabfae09b7dea24d5dd5ef84d7d1f5a7/python/sglang/srt/layers/attention/dsa/paged_mqa_logits_backend.py#L26), [Aiter wrapper](https://github.com/sgl-project/sglang/blob/c07ec416dabfae09b7dea24d5dd5ef84d7d1f5a7/python/sglang/kernels/ops/attention/dsa/paged_mqa_logits.py#L95).

## #46820: backpressure protection and its scope

The publisher, scheduler wiring, metrics reporter, socket factory and serialization helpers are unchanged between the published base and audited main. Prefill/decode/idle metrics still reach `emit_kv_metrics`; disabled KV events return before the send. Both serialization modes forward `zmq.NOBLOCK`, only `zmq.Again` is caught, and unexpected errors still propagate. Real KV events and the separate scheduler-load publisher do not replace this auxiliary PUSH channel.

The unpatched test suite first encountered a fixture mismatch because main lacks the PR-added `logger` attribute. That was not counted as behavioral evidence. A logger-only in-memory audit shim then allowed the unchanged tests to reach the original send: both backpressure cases raised real `zmq.Again` after the fixture's bounded 100 ms timeout. Closed-socket and unexpected-`EINVAL` controls passed; disabled-events checks also returned early on both trees.

**Configuration qualification:** the production socket factory already uses `SNDHWM=0` (unbounded), `SNDTIMEO=-1` and default `IMMEDIATE=0`; this was also true before the rebase. The regression deliberately uses finite `HWM=1` to induce pressure. With the factory's actual knobs, an unbound IPC probe on local libzmq 4.3.5 / pyzmq 27.2.0 accepted two messages, and the actual publisher returned on both source trees. This audit did not reproduce the historical c44a hang under current default configuration. The fix supplies nonblocking/EAGAIN protection; it does not bound queue memory or guarantee telemetry delivery. This is an existing scope limitation, not newly introduced drift.

Source: [current metrics send](https://github.com/sgl-project/sglang/blob/d796f86571836564d74301ff7dfcfe7181cefdea/python/sglang/srt/managers/scheduler_components/kv_events_publisher.py#L93). Historical provenance remains inventory S01 / c44a-to-c44b / R001; this audit makes no stack-throughput claim.

## #46830: metadata consumers and an adjacent synchronization

The backend, indexer, indexer metadata, top-k backend, Eagle preparation and overlap relay have identical blobs between `f64efd63` and audited main. The maintainer merge preserved the production changes and test-fixture hook repair.

The backend declares `needs_cpu_seq_lens=False`; overlap processing can remove the host mirror. Draft preparation supplies the fixed `extend_seq_lens_cpu` list that the paged path needs. Non-speculative extend/ragged consumers retain their separate host-data requirements. The production `init_kernel_metadata` implementation is a no-op, and its upstream tests pass with this PR. New [DCP top-k merge #48588](https://github.com/sgl-project/sglang/commit/f3532057e15fb8ede90ce1aeddfb5bd20dfbd4b6) takes device lengths/page tables and does not read the removed host fields; no production caller of its new `DcpTopKMerge` wrapper was found at this snapshot.

Unpatched main fails the two no-host-mirror cases at device max/item synchronization. Applying only the width change still leaves one draft-extend failure at the device prefix copy, independently demonstrating that the second correction remains needed. A real Torch top-k probe with large invalid-tail values confirms masking at valid lengths 0, 2, 5 and 7. The wider eager allocation remains an existing tradeoff; no new GPU-memory or latency qualification is claimed.

**Separate finding:** [the HIP indexer shortcut](https://github.com/sgl-project/sglang/blob/c07ec416dabfae09b7dea24d5dd5ef84d7d1f5a7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L895) still reads `seq_lens.max().item()` for eager DECODE/IDLE with no usable CPU mirror and nonempty device lengths. [Its caller](https://github.com/sgl-project/sglang/blob/c07ec416dabfae09b7dea24d5dd5ef84d7d1f5a7/python/sglang/srt/layers/attention/dsa/dsa_indexer.py#L2040) reaches this decision before the gfx950 fused-indexer dispatch. The [overlap relay](https://github.com/sgl-project/sglang/blob/c07ec416dabfae09b7dea24d5dd5ef84d7d1f5a7/python/sglang/srt/managers/overlap_utils.py#L618) and [Eagle eager fallback](https://github.com/sgl-project/sglang/blob/c07ec416dabfae09b7dea24d5dd5ef84d7d1f5a7/python/sglang/srt/speculative/eagle_worker_v2.py#L708) establish a real source-level route to these conditions. CPU interception of the actual imported method confirms the device-read path. This fallback predates this rebase and lies outside historical `dsa_dp_sync` / #33973, which changed only metadata code. A distributed GPU deadlock has not been reproduced.

A [two-line candidate](patches/pr46830-hip-indexer-no-host-sync-proposal.diff) removes the device-length fallback so the existing `return False` selects the complete indexer path without synchronizing solely to decide an optional shortcut. Eight CPU guard cases pass, retaining host-short/host-long, non-HIP, draft-extend and extend behavior. Short GPU-only eager batches may perform extra indexer work. The candidate is a reviewable follow-up artifact, **not applied to a published PR**. Current CI and source heads remain intact; this finding is not a demonstrated cause of a current Base failure.

A combined audit checkout containing latest main, #33059, #46830 and this candidate passed **18 tests and two subtests** in one invocation (the 12 row-contract, three metadata and three hook cases). The interaction is conditional: when the candidate selects full indexing, the optional gfx950 fused path does not return early, and physical Aiter query rows exceed real metadata rows, that path still needs #33059's row correction. Unpadded batches do not establish this dependency; the audit does not imply a universal merge order. These component tests also do not qualify a combined GPU deployment.

## Reproduction and evidence limits

Use the exact source heads above and the existing repository CPU environment:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/attention/test_dsa_aiter_paged_padding.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=python python -m pytest -q \
  test/registered/unit/managers/test_kv_metrics_backpressure.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 PYTHONPATH=python python -m pytest -q \
  test/registered/unit/layers/attention/test_dsa_eager_metadata_no_host_sync.py \
  test/registered/unit/layers/attention/test_dsa_kernel_metadata_hook.py
```

Run each command in its corresponding PR checkout; those tests are not all present on each individual branch. Run the metrics command with `SGLANG_USE_PICKLE_IPC=0` and `SGLANG_USE_PICKLE_IPC=1` for both serialization modes. For main negative controls, use identical PR test files against unpatched main; the metrics negative control requires only a logger attribute shim as described above. GPU operations are mocked by the existing DSA CPU harnesses; actual modules are imported. ZMQ tests use real sockets.

Exact commands, logs, source hashes, temporary integrations, negative controls and the candidate probe remain in `runtime/ci-watch-20261103/interactive/drift-audit-20261105T1158Z/pr{33059,46820,46830}/`. No GPU was allocated. These results do not establish a new distributed-deadlock reproduction, full model accuracy, performance, or qualification of the combined 48k deployment.
