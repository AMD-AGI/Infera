# REPRODUCE - ppdc1..4

Base (image, overlay, Mooncake patches, node prep, login notes): `../glm52.dcpx-pack-fix-best.packup_20261023/REPRODUCE.md`.

1. Patched files (on a host with /shared_nfs): `bash patches/install_pp.yihou.sh` builds deepseek_v2 / decode /
   metrics_reporter copies from overlay 93034eba91 with `pp-deepseek.patch`, `ec03.diff`, `memlog01.diff`, writes the
   mounts tsv (`mounts.yihou.dcpx-pp-prefill.tsv`, `mounts.yihou.dcpx-ec-decode.tsv`), adds the DCPX_PP / DCPX_SPF /
   DCPX_NOOFFLOAD / DCPX_DECODE_EC knobs and the agentx_env TP*PP fix (copies of the results in `scripts/harness/`).
2. Env + dryrun + token diff: `bash patches/mkenv_pp.yihou.sh` (ppdc1 vs dcpx11); `go_ppdc2/3/4.yihou.sh` for the others.
3. Run (on 135): C56 A1 only `bash scripts/tools/orch14.yihou.sh ppdc1`; a sweep on one stack
   `bash scripts/tools/orch15.yihou.sh ppdc2 64 72` (A1 @C64, A2 @C72); one point `orch16.yihou.sh ppdc4 80 80`.
4. Check the per-GPU denominator: `python3 -I patches/gpuchk.yihou.py <bench dir>` must show PREFILL_TP=4.
