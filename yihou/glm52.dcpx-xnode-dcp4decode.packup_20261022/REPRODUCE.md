# REPRODUCE — dcpx rounds (DCP4 decode leg, cross-node 135 prefill -> 136 decode)

Run from a spur login host (`ssh spur006`, fallback 007/005); login hosts evict a user above ~4 GB, keep commands tiny.
135/136 are reached by plain ssh from the login host.

```bash
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs
W=/shared_nfs/yihou_glm52/g4tune_20261019
D=/shared_nfs/yihou_glm52/dcpx_20261022
```

## 0. Prerequisites
- Base image `infera-sglang:v0522-1020-yihou-dcp-mc4340` from
  `/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/images/infera-sglang_v0522-1020-yihou-dcp-mc4340.tar`;
  `docker load` on 135 and 136 gives ID `437497f5...` (used by dcpx1/dcpx2).
- sglang overlay `93034eba91` (v0.5.22 base, from the DCP kit), mounted on BOTH legs by
  `scripts/harness/mounts.yihou.dcpx-overlay.sglang-93034eba91.tsv` (source
  `/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-93034eba91`). All seven g3x1 prefill mounts are already contained in
  this overlay (code read, research 10-22), so no other mount is used.
- Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (read-only NFS).
- Node checks before every launch: `/` < 92 % (136 was lost for days to a full disk), docker root on the nvme LV,
  0 KFD processes, rails ACTIVE. Everything is written to NFS (`NODE_RW_ROOT=`).

## 1. Build the `-destpin` image (dcpx3 onward)
Mooncake faae8dd4 rebuilt inside the base image with the PR #4340 port + llying's dest-local-rail, ib-timeout and
dest-pinned-hca patches (`imgbuild/`):
```bash
ssh crsuse2-m2m-135 "cd $D/imgbuild && bash build.yihou.sh > build.yihou.log 2>&1"   # builds on 135, docker save | ssh 136 docker load
# result: infera-sglang:v0522-1020-yihou-dcp-mc4340-destpin, ID sha256:1b78f7cc... on both nodes
```
The aiter JIT cache is keyed by the local image ID: copy the seeded cache (`$R/../cache/13{5,6}/aiter/{prefill,decode}/437497f5...`)
under the `1b78f7cc...` key on both nodes, otherwise the bring-up JIT-compiles cold.

## 2. Env, dryrun, token diff
`scripts/envs/env.dcpx6-135x136.yihou.sh` is the run of record. `CONFIG=$H/config.yihou.dcpx.sh` layers the DCP decode on
top of the cross-node chain (xnode -> idlefix -> g4new -> xferfix2 -> offload -> tokw2 -> mc4340n19 -> E2-target).
Knobs of `config.yihou.dcpx.sh`: `DCPX_DCP`, `DCPX_IMAGE`, `DCPX_RAIL` (affinity | destlocal | hip), `DCPX_DECODE_RDMA`,
`DCPX_SPLIT`, `DCPX_SDMA0`, `DCPX_IB_TIMEOUT` (18), `DCPX_MC_MAX_MR` (1 GiB), `DCPX_{PREFILL,DECODE}_IMAGE_ID`.
```bash
source $W/env.dcpx6-135x136.yihou.sh
bash $H/run.v3x.yihou.sh dryrun > $W/dryrun.dcpx6-135x136.yihou.txt
diff <(tr ' ' '\n' < $W/dryrun.dcpx5-135x136.yihou.txt | sed s/dcpx5/ROUND/g) \
     <(tr ' ' '\n' < $W/dryrun.dcpx6-135x136.yihou.txt | sed s/dcpx6/ROUND/g)     # expect: no difference
```
Note: the header comment of `env.dcpx6` still describes dcpx1 (mem 0.85 / max_running 64 / index_share false);
`YIHOU_OVERRIDES` and the dryrun are authoritative (mem 0.85, max_running 128, graph bs 128, index_share true).

## 3. Launch (on 135), A1 + A2 at C56
```bash
ssh crsuse2-m2m-135 "cd /; setsid nohup bash $H/chain_x.yihou.sh $W/env.dcpx6-135x136.yihou.sh 56 56 > $R/dcpx6.chain.log 2>&1 < /dev/null &"
# stop: ssh crsuse2-m2m-135 "cd $H && source $W/env.dcpx6-135x136.yihou.sh && bash $H/run.v3x.yihou.sh stop"   (docker stop only)
```
chain_x = up -> census -> smoke -> audit -> A1 (AgentX 3600 s, 10 warm-up requests per lane, client on 135) -> flush -> A2.
Same-node variant (dcps1): `scripts/envs/env.dcps1-136.yihou.sh` (`DCPX_RAIL=hip`, decode GPU4-7).

## 4. Analysis
```bash
python3 -I scripts/tools/hist.yihou.py dcpx6 <server_metrics_export.csv>        # transfer-stage histograms
python3 -I scripts/tools/pdbal.yihou.py <metrics.A1.tsv> <t0_epoch> <t1_epoch>   # per-role/per-DP gauge means
python3 -I scripts/tools/decode_stats.dcp.yihou.py --bench <A1>/benchmark.log $R/dcpx6/server-logs/decode-0.a1.log
bash scripts/tools/rdmactr.yihou.sh <label>                                       # on a node: ionic hw counters
bash scripts/tools/gather.dcpx.yihou.sh <new-yihou-stage-dir>                     # rebuilds this kit's results/
```
