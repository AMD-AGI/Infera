# Reproduce llr5..llr7

Prerequisites are those of `yihou/glm52.llrepro-pp4-dpa4.packup_20261024` (image `infera-sglang:v0522-fdbafb60` built by
its `image/build_ll.yihou.sh` on BOTH nodes, model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`, nodes
crsuse2-m2m-135 (prefill, router, AgentX client) and crsuse2-m2m-136 (decode), GPU4-7 on each, ionic_0..7, passwordless
ssh 135 -> 136). Nothing else may run on GPU4-7.

```bash
D=/shared_nfs/yihou_glm52/llrepro_20261023          # any NFS dir visible from both nodes
mkdir -p $D && cp -a kit scripts/* $D/              # kit/ already contains both hooks (no need to re-run them)
mkdir -p /shared_nfs/yihou_glm52/kf_e2e_20261023/kf/e2e
cp -a kf_overlay /shared_nfs/yihou_glm52/kf_e2e_20261023/kf/e2e/overlay   # path used by DECODE_MOUNT_EXTRA
# edit WORK_DIR / IPs in the site files if the paths or nodes differ, then ON 135:
bash $D/run_llr.yihou.sh $D/llr5.site.env llr5-c80   # KF kernels                          ~95 min
bash $D/run_llr.yihou.sh $D/llr6.site.env llr6-c80   # + DECODE_CUSTOM_AR=1
bash $D/run_llr.yihou.sh $D/llr7.site.env llr7-c80   # + SGLANG_OPT_USE_TOPK_V2=false (decode only)
python3 -I $D/llsum.yihou.py $D/work/runs/llr7-c80   # summary row
python3 -I $D/bneck.yihou.py $D/work/runs/llr7-c80   # steady-state decode / prefill queue stats
```

`run_llr.yihou.sh` waits until neither node runs a container, removes only our exited `yihou-llr-*` containers (never
`-v`), runs `up.sh` + `agentx.sh`, copies the container logs to `$WORK_DIR/runs/<run>/logs/`, then `docker stop`s.

Check after start (decode, 136): `docker exec yihou-llr-decode env | grep TOPK_V2` -> `false` for llr7;
`docker logs yihou-llr-decode | grep yihou-kf` -> `installed` lines, `CHECK ... snr` lines, and per-shape
`indexer_logits/indexer_topk rows=N -> KF|orig` lines (see notes.md for the expected pattern);
`disable_custom_all_reduce': False` in the server-args line for llr6/llr7.

Probe (optional, during the profiling phase, on each node): `bash node_probe.yihou.sh <outprefix> 120`.
