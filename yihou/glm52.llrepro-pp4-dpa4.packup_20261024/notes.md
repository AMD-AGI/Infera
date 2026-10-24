# Notes

1. **Decode mem 0.95 is not a bad point (user, 10-24).** llr2 served more requests per second (+2.4 %) with lower TTFT
   (p90 -36 %) and TPOT; its score fell only because that run's mean input was 6 % shorter (closed-loop mix varies more than
   the ~3 % assumed). Keep it as a candidate when the decode is KV-capacity bound. Also possible (not checked): a kernel
   path that allocates speculatively and only takes the fast path when memory headroom exists.
2. Judge a change by qps, TTFT/TPOT and the stage histograms together with the score, and report the ISL change.
3. Kit vs llying's run: same kit and image recipe; deepseek_v2.py hunks applied at +3 lines offset (2987/3022 vs his
   2984/3019) - the rebuilt base differs slightly from his; image ID differs (rebuild). GPUs 4-7 on both nodes and all 8
   ionic rails as in his site; WORK_DIR on NFS; CTR yihou-llr (cache volumes reused across runs after removing the old
   exited containers - user-approved 10-23).
4. llr3 index_share on this image ran an hour with no decode error / Memory access fault.
5. llr5 (KF): kit keeps `SGLANG_OPT_USE_TOPK_V2=true`, which the KF indexer does not support; the KF indexer engaged only
   for some shapes (e.g. q rows 108/120 -> KF, 12/132/144+ -> orig). Sparse MLA and MoE: see the run's decode log.
6. Startup `post-warmup freeze_gc` ConnectionRefused tracebacks in the prefill log are harmless (before the bench).
