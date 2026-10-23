# REPRODUCE - ppda1..3

Base as in `../glm52.ppdc-pp4-dcp4.packup_20261023/REPRODUCE.md` (image, overlay, PP install). Then:
- ppda1: `scripts/envs/env.ppda1-135x136.yihou.sh`, run `bash scripts/tools/orch16.yihou.sh ppda1 80 80` on 135.
- ppda2: `bash patches/prep_ppda2.yihou.sh` (KF knob + env + dryrun), then `orch16.yihou.sh ppda2 80 80`.
- ppda3: `bash patches/prep_ppda3.yihou.sh` (ROUTER_EXTRA_ENV in run.v3x + DCPX_R2 knob + env + dryrun).
- Old containers: `patches/rm_old.yihou.sh [dry]` (our prefixes, exited only, logs on NFS first, no -v; user-approved 10-23).
