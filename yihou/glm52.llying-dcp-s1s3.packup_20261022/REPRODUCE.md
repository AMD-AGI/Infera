# REPRODUCE — S1 / S2 / S3 (same-node P4D4, AgentX C80, 1800 s)

All paths under `/shared_nfs/...` are on the crusoe (spur) cluster NFS. Cluster access: `machine-pool` /
`spur-cluster` skills (login `spur`, compute via `spur exec <job>`; no ssh to compute nodes). Run heavy work on
compute nodes only; crusoe login nodes kill a user above 4 GB RSS.

## 0. Inputs (EXTERNAL, absolute paths)
| Input | Path |
|---|---|
| Model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` |
| Image tar | `/shared_nfs/yihou_glm52/llying_dcp_20261022/images/infera-sglang_v0522-d395c3b3-1020-mc4340.tar` (sha256 `7067b016125841acfcdebd2d511b51007133a9a36e8c9432e3271d8ca2523ade`) |
| sglang overlay (mounted ro over `/sgl-workspace/sglang/python/sglang` on both legs) | `/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-ab7a898b77/python/sglang` |
| InferenceX (AgentX client), pinned `918524ff94045b3f091115f1051c22a8588edf2b` | `/shared_nfs/yihou_glm52/dcp_pd_20261021/cache/InferenceX` (run_stage copies it per work dir) |
| Kit (identical to `scripts/kit/`) | `/shared_nfs/yihou_glm52/llying_dcp_20261022/kit/` |
| Analysis tools (identical to `scripts/pd-tools/`) | `/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/tools/` |

## 1. Image (only if the tar is gone)
Recipe in `scripts/image/`: `Dockerfile.sglang.v0522-d395c3b3-1020-mc4340.yihou` on base
`lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020`, Infera context = Infera commit `d395c3b3`
(`origin/feat/sglang-mix-glm5.2-tp4-v0522`) plus `build_mooncake_sglang.mc4340.yihou.sh` and
`mooncake-pr4340-e8062249.faae8dd4-port.yihou.diff` (Mooncake#4340 port onto Mooncake `faae8dd4`). The exact context
is `scripts/image/llimage-ctx.d395c3b3-mc4340.tar.gz`. All sglang source-patch steps are OFF: the overlay carries them.
```bash
# copy llimage-ctx.*.tar.gz and sglang-src-ab7a898b77.tar.gz (git archive of the sglang commit, below) to
# /shared_nfs/yihou_glm52/llying_dcp_20261022/build/inputs/, then on a held node:
spur exec <JOB> bash /shared_nfs/yihou_glm52/llying_dcp_20261022/<path>/build_on_node.yihou.sh
```
Expected image ID `sha256:d3d6a9c20ce060961ef13c69a6e281c09e0bd82b4411eb4fe26148d6f9fa854d`. A rebuild will not
reproduce the ID byte-for-byte (pip/cargo network fetches); use the tar for exact reproduction.

## 2. sglang code (overlay)
Branch `dev/yihou/glm52_dcp_pd` @ `ab7a898b77` (local only, not pushed). Rebuild it: apply
`../glm52.dcp4-pd.packup_20261022/patches/` (fda84f80b7 + PR #44322 merge + 31 commits -> `93034eba91`, steps in its
README), then `git am patches/0001-Fix-deepseek_v2-Read-hidden-size-from-config-on-non-.patch` (parent `93034eba91`;
PP>1 non-first ranks crashed at model construction without it). Overlay = `git archive --format=tar.gz -o sglang-src-ab7a898b77.tar.gz ab7a898b77 python/sglang` (sha256 in
`scripts/image/SHA256SUMS`), extracted at the path in §0 (`build_on_node.yihou.sh` step (a) does this).

## 3. Kit
`scripts/kit/` = `orig/` (Infera `d395c3b3` `examples/sglang_2p1d_glm5.2`) + `kit-v6-pp4.nodeepseek.diff`
(`patch -p2`) + `kitedit.yihou.py` -> `kit/`. Stage knobs in `stages/S{1,2,3}.env`; `render/` shows the resulting
docker argv per stage (`render/render.yihou.sh <s>` with stubbed docker/ip). Read `scripts/kit/README.yihou.md`.

## 4. Get a node and arm the container guard
One full MI355X node (8 GPUs) per stage, via the machine pool (`python3 pool.py request --cluster crusoe --kind oai ...`).
Before launching, start the job-bound guard (stops our containers ~90 s after the job leaves RUNNING):
```bash
spur exec <JOB> bash -c 'cd /tmp && setsid nohup bash /shared_nfs/yihou_glm52/dcp_pd_20261021/pd/tools/job_container_guard.yihou.sh start <JOB> <NK> > /dev/null 2>&1 < /dev/null &'
```
`<NK>` = node key (last digits of the hostname, e.g. 219); containers are named `yihou-dcp-pd-<NK>-s<stage>-*`.

## 5. Launch one stage (detached; ~25 min bring-up incl. image load + ~30 min warm-up + 30 min profiling)
```bash
# copy scripts/launch_stage.yihou.sh to /shared_nfs/yihou_glm52/llying_dcp_20261022/work/ once, then:
spur exec <JOB> bash -c 'cd /tmp && bash /shared_nfs/yihou_glm52/llying_dcp_20261022/work/launch_stage.yihou.sh <1|2|3> <NK> [EXTRA_ENV]'
# S3-aux (burst v3 node): EXTRA_ENV=HSA_ENABLE_SDMA=0  (note: S3-aux's engine-config.env shows it on PREFILL_ENV only)
tail -f /shared_nfs/yihou_glm52/llying_dcp_20261022/work/run_stage.S<s>-<NK>.log
```
`launch_stage` loads the tar if the image ID `d3d6a9c20ce0` is not present, then runs `kit/run_stage.yihou.sh <s>`:
preflight (GPUs idle, ports free) -> `engine/up.sh` -> `engine/agentx/agentx.sh` (C80, 1800 s, warm-up 10/lane,
`SGLANG_SIMULATE_ACC_LEN=3.61`) -> stack left up (set `DOWN=1` for `down.sh` = docker stop + log save).
It refuses to relaunch when the stage log already exists.

## 6. Analyze (on the compute node, not the login node)
```bash
spur exec <JOB> bash -c 'cd /tmp && bash /shared_nfs/yihou_glm52/llying_dcp_20261022/kit/analyze_ll.yihou.sh <RUN_DIR> yihou-dcp-pd-<NK>-s<s>'
```
Writes `<RUN_DIR>/analysis.txt` (dumps live container logs to `logs-live/`). **Per chip = `total_tput / 8`**; the
JSON `per_gpu` of these four runs is total/5 (see `notes.md` §1). The current `kit/engine/agentx/agentx.sh` already
has `PREFILL_TP=4`, so a rerun's `per_gpu` is total/8.

## 7. Stop
`spur exec <JOB> docker stop <each yihou-dcp-pd-<NK>-s<s>-* name>` (never `docker rm`), then disarm the guard
(`job_container_guard.yihou.sh disarm <JOB> <NK>`) and release the pool request.
