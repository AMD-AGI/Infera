# Reproduce (delta for 2026-11-20 05:30Z – 2026-11-24)

Staging, preflight, client selftest and the single-point chain are unchanged: see
`../dsv41_vllm_pd_20261119.packup_20261120/REPRODUCE.md` sections 0-2. This file adds the crusoe helpers and the PD points.

## 0. Assemble the kit
The kit = old kit + the changed/new files of this pack-up (later files win):
```bash
OLD=../dsv41_vllm_pd_20261119.packup_20261120; NEW=.          # run from this pack-up dir
mkdir -p /tmp/yihou_kit && rsync -a $OLD/scripts/kit/ /tmp/yihou_kit/ && rsync -a $NEW/scripts/kit/ /tmp/yihou_kit/
mkdir -p /tmp/yihou_cfg && rsync -a $OLD/configs/ /tmp/yihou_cfg/ && rsync -a $NEW/configs/ /tmp/yihou_cfg/
```
(cru_launch.sh / cru_pd.sh from the old kit are superseded by the cru_* helpers below.)

## 1. crusoe (spur): stage on NFS, then drive from the node via `spur exec`
The cru_* helpers hard-code `R=/shared_nfs/yihou/dsv41_vllm_pd_20261119` and expect the kit at `$R/ws/scripts/kit` and configs at
`$R/ws/configs`; logs go to `$R/launch/<tag>.{serve,point,stop}.log`, ledger `$R/results/ledger.csv`.
`cru_env.sh` (unchanged, old kit) exports DSV41_NFS_ROOT, DSV41_LOCAL_ROOT=/mnt/m2m_nobackup/yihou/…, DSV41_MODEL=/shared_nfs/models/
DeepSeek-V4.1-Flash, KIT_VENV_CTR, KIT_IX. Stage venv-agentx + hf_home on crusoe NFS BEFORE requesting a node (notes.md #1).
```bash
R=/shared_nfs/yihou/dsv41_vllm_pd_20261119
rsync -a /tmp/yihou_kit/ spur005:$R/ws/scripts/kit/ && rsync -a /tmp/yihou_cfg/ spur005:$R/ws/configs/
ssh spur005                                   # spur006 if spur005 is full (notes.md #6)
spur exec <job> -- bash -lc "source $R/ws/scripts/kit/cru_env.sh; bash \$K/pull_images.sh r1-dp2-c128"
```
One colocated official point (serve.sh --wait -> run_point.sh official -> `CHAIN_DONE rc=N` appended to the point log):
```bash
# cru_launch_point.sh <cfg> <gpus> <port-offset> <conc> <tag-prefix>   (prints "NEW <tag>")
spur exec <job> -- bash $R/ws/scripts/kit/cru_launch_point.sh r1-dp2-c128 0,1 0 256 e3-r1cru-c256
spur exec <job> -- bash $R/ws/scripts/kit/cru_launch_point.sh e4c-r1-dsmla-c128 2,3 10 256 e4c-cru-c256
spur exec <job> -- bash $R/ws/scripts/kit/cru_launch_point.sh e4b-r1-dsmla-nosparse-c128 4,5 20 256 e4b-cru-c256
```
Filling a node: start servers one at a time (next starts only after "ready after" in the previous serve log, max 20 min each):
```bash
spur exec <job> -- nohup bash $R/ws/scripts/kit/cru_seq_launch.sh \
  "r1-dp2-c128 0,1 0 256 e3-r1cru-c256" "e4c-r1-dsmla-c128 2,3 10 256 e4c-cru-c256" \
  "e4b-r1-dsmla-nosparse-c128 4,5 20 256 e4b-cru-c256" "r1-dp2-c128 6,7 30 288 e3-r1cru-c288" > $R/launch/seqlaunch-$(date -u +%H%M).log 2>&1 &
```
Keeping a node busy: queue "after <old-tag> finishes, stop it and launch the next" with >= MIN_GAP s (default 480) between launches:
```bash
# spec = "<old-tag> <cfg> <gpus> <off> <conc> <prefix>"; LAST_START=<epoch of the last launch> keeps the gap across chains
spur exec <job> -- bash -c "MIN_GAP=480 LAST_START=$(date +%s) nohup setsid bash $R/ws/scripts/kit/cru_seq_swap.sh \
  'e3-r1cru-c256-<MMDDHHMM> e4c-r1-dsmla-c128 0,1 0 288 e4c-cru-c288' \
  'e4c-cru-c256-<MMDDHHMM> e4b-r1-dsmla-nosparse-c128 2,3 10 288 e4b-cru-c288' > $R/launch/chainN.log 2>&1 < /dev/null & disown"
# single swap without queue: cru_swap_after.sh <old-tag> <cfg> <gpus> <off> <conc> <prefix>
```
Run the swap queue ON the node with nohup+setsid/disown: a chain started over a dropped ssh died (notes.md #4). Waiters wait up to
3 h (cru_seq_swap) / 3 h (cru_swap_after 1080x10 s). Stop a point: `spur exec <job> -- bash $R/ws/scripts/kit/stop.sh <tag>`.
Kill a chain by its PID (`kill <pid>`), never `pkill -f <script>` inside `spur exec bash -c` (notes.md #5).

PD points back to back on one crusoe node (fresh stack per point; GPU denominator 4):
```bash
# cru_pd_chain.sh <p-gpus> <d-gpus> <port-offset> <cfg:conc>...
spur exec <job> -- nohup bash $R/ws/scripts/kit/cru_pd_chain.sh 0,1 2,3 0 e7u-pd-tcp-hip:256 > $R/launch/pdchain.log 2>&1 &
```
Status on crusoe: e7th/e7tp (TCP + MC_DISABLE_HIP) move no KV; e7u (TCP, HIP allowed) fails with hipIpcOpenMemHandle 201; RDMA
container setup crashes with Triton Code 209. No crusoe PD config is known to work with this image (research/mooncake_samenode_transport.md).

## 2. openai (smci355): PD 1P1D points (on the node, kit + configs staged on /perf_apps, NOT $HOME NFS)
```bash
K=/perf_apps/yihou/dsv41_vllm_pd_20261119/stage/kit; T=e7-pd-c256-$(date -u +%m%d%H%M)
bash $K/serve_pd.sh e7-pd-mooncake-1p1d-dp2 0,1 2,3 $T --wait \
  && bash $K/run_point.sh $T http://127.0.0.1:8000 256 4 official; bash $K/stop.sh $T --prefix
#   e7h (MC_DISABLE_HIP=1): e7h-pd-mooncake-1p1d-dp2-nohip ; e7p (e7h + P mem 0.93 / long-prefill 8192 / max-num-seqs 64): e7p-pd-nohip-pb300
#   second stack on the same node: GPUs 4,5 6,7 and --port-offset 1000 (router then on :9000)
```
Colocated replicates (R2 c384/c512, e4b/e4c c128) use `launch_remote.sh` exactly as in the previous REPRODUCE.md, with conc 512 for e3-r2-c512.
PD smoke (correctness, not a score): `serve_pd.sh e7s-pd-mooncake-1p1d-dp2-smoke 0,1 2,3 <tag> --wait`, reference
`serve.sh e7r-colo-dsmla-ref-smoke 4,5 <ref> --port-offset 40 --wait`, then `pd_smoke.sh <tag> http://127.0.0.1:8040 --docker`
(old kit PD_README.md). Do not run two PD stacks on the same port set (e7so died on a port 6200 clash).

## Expected (same image; single runs, node shared with other servers)
openai: R2 c512 ≈ 425k, R2 c384 ≈ 407-414k, e7 PD c256 ≈ 254k, e4c c128 ≈ 244-248k, e4b c128 ≈ 221-227k; e7 c512 fails in warmup
(hipIpcOpenMemHandle 201) unless MC_DISABLE_HIP=1. crusoe: R1 c256 ≈ 348-373k, e4c c256 ≈ 274-315k, e4b c256 ≈ 265-276k (README table);
expect node stalls on v3-014 (notes.md #2).
