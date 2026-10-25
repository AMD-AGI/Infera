# Environment (2026-10-25 02:5xZ .. 16:2xZ)

Same as `yihou/glm52.llrepro-atomdyn-tune.packup_20261025/environment.md` (135 prefill PP4 x TP1 GPU4-7, 136 decode
TP4/DP4 DPA GPU4-7, image `infera-sglang:v0522-fdbafb60`, kit + yihou hooks, KF overlay kernel files unchanged since
llr5; overlay `__init__.py` gained an additive flag-gated fused-indexer hook on 10-24 18:39). Kit change: prefill env
hook `PREFILL_ENV_EXTRA`. Runs llr21r..llr23: WORK_DIR on 135 local disk `/mnt/m2m_nobackup/yihou_llrepro/work`.
Login via spur007 (spur006 ssh hangs).
