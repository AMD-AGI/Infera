> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# REPRODUCE

Same machinery as `../glm52.kf-decode-kernels-e2e.packup_20261023/REPRODUCE.md` (variables `$H $K $KH $R $W`, node
prep, image load, aiter JIT cache seeding, stop command). Only the differences are listed here.

## 0. Files onto cluster NFS
`overlay/` -> `$K/kf/e2e/overlay` (check `overlay/MD5SUMS`); `cluster/kf_e2e_20261023/harness/*` -> `$KH/`;
`cluster/kf_e2e_20261023/env.*` -> `$K/`; `tasks/<task>/{driver.py,graph_harness.py,<kernel>.py}` -> `$K/kf/tasks/<task>/`.
remote-launcher 0.15: `remote-launcher exec` runs as registry; use a node whose `/var/run/docker.sock` is 0666 (notes 3).

## 1. Kernel level in the target image (GPU 0, no server)
```bash
$W 'remote-launcher exec <job> bash -c "cd / && setsid nohup bash $K/kf/dcpk/run.sh >/dev/null 2>&1 </dev/null &"'   # DCP sparse+indexer -> $K/dcpk.log
$W 'remote-launcher exec <job> bash -c "cd / && setsid nohup bash $K/kf/fik/run.sh  >/dev/null 2>&1 </dev/null &"'   # fused indexer   -> $K/fik.log
```
Scripts: `cluster/kf_e2e_20261023/kf/{dcpk,fik}/run.sh`. All-rows sweeps: `scripts/dcp_{sparse,indexer}_allrows.registry.py`.

## 2. e2e rounds (env files carry node/IP/job; one variable per comparison)
| round | node | knobs (on top of the 10-23 DPA/DCP base) |
|---|---|---|
| kda1 | 169 | `DCPX_DCP=4 KF_ON=0` (DCP base) |
| kdc1 | 169 | `DCPX_DCP=4 KF_ON=1 KF_DCP=1`, DPA kernels off |
| kde1 | 169 | as kdc1 + `KF_DCP_SPARSE=0` (indexer only) |
| kdg1 | 192 | as kdc1 + `KF_DCP_INDEXER=0` (sparse only) |
| kdf1 | 046 | `DCPX_DCP=4 KF_ON=0 KF_DUMP=1` (capture, eager decode; not a perf run) |
| kfb3 | 192 | `DCPX_DCP=0 KF_ON=1` (DPA + 3 KF kernels) |
| kfc1 | 046 | as kfb3 + `KF_FUSED=1` |
(`env.kdb1-190` exists in `cluster/` but kdb1 is not a round of record.)
```bash
cd $KH && (source $K/env.<round>-<node>.registry.sh && bash $KH/run.v3.registry.sh dryrun > $K/dryrun.<round>-<node>.registry.txt)
$W 'remote-launcher exec <job> bash -c "cd / && setsid nohup bash $KH/chain_cap2.registry.sh $K/env.<round>-<node>.registry.sh 56 56 > $R/<round>.chain.log 2>&1 </dev/null &"'
```

## 3. Real-input check of the DCP indexer
After kdf1 has written `$R/kdf1/dcpdump/` (kvpool.L<layer>.pt + call.NNN.pt):
`cluster/kf_e2e_20261023/dcpreal_run.registry.sh` (runs `scripts/dcp_indexer_realdata.registry.py` in the image, GPU 0) -> `$K/dcpreal.log`.

## 4. KF campaigns (workspace-node / compute-cluster node, KernelForge f107746)
`tasks/<task>/campaign.env` + `node/campaign.registry.sh` in `REDACTED_LOCAL_PATH` (`FORCE_RESUME=1` to resume a
[Private orchestration details omitted.]

## 5. Analysis
`python3 ../glm52.kf-decode-kernels-e2e.packup_20261023/scripts/summary.registry.py cluster/v0521dbg_20261010/runs > results/summary.tsv`.
kde1 has no AgentX JSON: mean of `inter_token_latency` over profiling-phase records of `profile_export.jsonl` (notes 3).
