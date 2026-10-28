Our later PR #45483 independently diagnosed the same active-backend ownership bug. After comparing both diffs with current main (`e4d4a9a7de5454cd8b854b21af723c2155a55a3d`), I am consolidating our report here. This PR also handles empty-batch cleanup, which #45483 deliberately left unchanged.

Additional AMD evidence from GLM-5.2-MXFP4 on MI355X, a v0.5.21-based deployment, same-node 1P1D (prefill TP4, decode TP4/DP4), EAGLE MTP 5/1/6, `index_share_for_mtp_iteration=true`, concurrency 56:

- In `dbge4a`, an IDLE batch reached the DSA k-only top-k with a stale length of 19,238 and a dummy-logits row width of 2,048; a GPU memory access fault followed one second later. The debug agent identified `coop_topk_kernel`. Other dumps recorded stale lengths up to 104,740.
- The idle draft step selected `draft_attn_backend.attn_backends[i]` through `ForwardContext`, while `_execute_idle` initialized the draft-extend backend held in `model_runner.attn_backend`. The DSA indexer consumed the former backend's stale metadata.
- In instrumented run `dbge6a`, initializing the active backend eliminated the out-of-row calls: 0 BAD calls and 0 memory access faults during the full A1 run. In `idxf1a`, the backend-initialization fix was the only change from faulting baseline `g4n1`; A1+A2 completed (approximately 170 minutes) with 0 faults.

These are historical results for our additive initialization patch, not a GPU rerun of this PR or current main. The workload used simulated MTP acceptance (`3.61`); it demonstrates the metadata/fault behavior and does not establish generated-text correctness. I am making no throughput claim. A separate October 27 stack-level bisect implicated PP communication overlap and decode custom all-reduce in output corruption; that does not invalidate this observed out-of-bounds metadata path or establish that fixing it resolves all corruption.

Source record: `yihou/glm52.v0521-idxshare-maf-fix.packup_20261021/README.md` in the Infera experiment archive (run IDs above). We will follow this PR rather than maintain a competing fix.
