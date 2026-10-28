> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Environment (2026-10-25 02:5xZ .. 16:2xZ)

Same as `registry/glm52.llrepro-atomdyn-tune.packup_20261025/environment.md` (REDACTED_NODE prefill PP4 x TP1 GPU4-7, REDACTED_NODE decode
TP4/DP4 DPA GPU4-7, image `historical-sglang-image-b`, kit + registry hooks, KF overlay kernel files unchanged since
llr5; overlay `__init__.py` gained an additive flag-gated fused-indexer hook on 10-24 18:39). Kit change: prefill env
hook `PREFILL_ENV_EXTRA`. Runs llr21r..llr23: WORK_DIR on REDACTED_NODE local disk `REDACTED_LOCAL_PATH`.
Login via login-node-b (login-node-a ssh hangs).
