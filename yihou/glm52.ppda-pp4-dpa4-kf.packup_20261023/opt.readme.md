# What changed in ppda1 / ppda2 / ppda3 (knobs in scripts/harness/config.yihou.dcpx.sh)

| round | knob | what | effect measured |
|---|---|---|---|
| ppda1 | `DCPX_DCP=0` on the ppdc4 env | decode back to DPA4 (TP4/DP4, mem 0.95, max-running 96, graph 64, no DCP envs, no empty_cache patch); PP prefill forces router affinity 0 (one prefill DP) | 34,288 vs DCP4 31,076 @C80 (+10.3 %), TPOT 20.0 vs 23.5 ms |
| ppda2 | `DCPX_KF=1` | decode-only mount `<kf overlay dir> -> /yihou_overlay` + `PYTHONPATH=/yihou_overlay YIHOU_KF_SPARSE=1 YIHOU_KF_INDEXER=1 YIHOU_KF_MOE=1`; engages on non-DCP EP1 decode, TARGET_VERIFY <= 128 rows, needs `SGLANG_OPT_USE_TOPK_V2=false` | 40,780 (+18.9 %), TPOT 14.2 ms; e2e correctness unchecked |
| ppda3 | `DCPX_R2=on` | router env `INFERA_R2_DECODE_DEMAND=on` via new `ROUTER_EXTRA_ENV` in run.v3x | running |

Verify KF engaged: decode log `[yihou-kf] ... installed: ['INDEXER','MOE','SPARSE']`, `-> KF` per shape, no `RAISED`.
