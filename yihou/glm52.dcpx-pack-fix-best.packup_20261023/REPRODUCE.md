# REPRODUCE - dcpx7..12, dpax1

Prerequisites, image build and node prep: `../glm52.dcpx-xnode-dcp4decode.packup_20261022/REPRODUCE.md` §0-1 (image
`infera-sglang:v0522-1020-yihou-dcp-mc4340-destpin`, ID 1b78f7cc on 135/136; overlay 93034eba91 on both legs).
Login: the spur login hosts may hang; `ssh -tt spur` (LB) works with commands piped in. Keep login-side work < 4 GB.

```bash
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; W=/shared_nfs/yihou_glm52/g4tune_20261019
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
```
1. Mount files: copy `patches/conn.yihou-dcpshard2*.py` and `patches/dcp_pack.yihou-nowait2.py` to `$H/mounts-v0522/`
   (or regenerate: `python3 -I patches/mk_shard2.yihou.py <overlay conn.py> <out>`; `mk_pack.yihou.py` then
   `mk_pack2.yihou.py` on the overlay `common/dcp_pack.py`; conn stats = conn + `xferstats2.tail.py`). Tsv files are in
   `scripts/harness/`.
2. Env + dryrun + token diff (one variable per round):
   ```bash
   source $W/env.dcpx11-135x136.yihou.sh; bash $H/run.v3x.yihou.sh dryrun > $W/dryrun.dcpx11-135x136.yihou.txt
   diff <(tr ' ' '\n' < $W/dryrun.dcpx10-135x136.yihou.txt | sed s/dcpx10/ROUND/g) <(tr ' ' '\n' < $W/dryrun.dcpx11-135x136.yihou.txt | sed s/dcpx11/ROUND/g)
   ```
3. Run A1 only (on 135): `bash scripts/tools/orch14.yihou.sh dcpx11` (launches chain_x, waits for `bench ok ... A1`,
   docker-stops both legs, checks 0 containers / 0 KFD on both nodes). dpax1: same with `env.dpax1`.
4. Analysis: `results/xfer_stats.txt` was produced by
   `python3 -I scripts/tools/xsum.yihou.py <prefill-0.log> <profiling start HH:MM:SS>` and `xsub.yihou.py` (pack
   sub-timers); `scripts/tools/summary2.yihou.sh <stage>` rebuilds `results/summary.tsv` (needs `$N/hist.yihou.py`).
