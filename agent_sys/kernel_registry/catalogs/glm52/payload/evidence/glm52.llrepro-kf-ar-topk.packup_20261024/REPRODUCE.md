> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Reproduce llr5..llr7

Prerequisites are those of `registry/glm52.llrepro-pp4-dpa4.packup_20261024` (image `historical-sglang-image-b` built by
its `image/build_ll.registry.sh` on BOTH nodes, model `REDACTED_LOCAL_PATH`, nodes
compute-node-03 (prefill, router, AgentX client) and compute-node-04 (decode), GPU4-7 on each, ionic_0..7, passwordless
ssh REDACTED_NODE -> REDACTED_NODE). Nothing else may run on GPU4-7.

```bash
D=REDACTED_LOCAL_PATH          # any NFS dir visible from both nodes
mkdir -p $D && cp -a kit scripts/* $D/              # kit/ already contains both hooks (no need to re-run them)
mkdir -p REDACTED_LOCAL_PATH
cp -a kf_overlay REDACTED_LOCAL_PATH   # path used by DECODE_MOUNT_EXTRA
# edit WORK_DIR / IPs in the site files if the paths or nodes differ, then ON REDACTED_NODE:
bash $D/run_llr.registry.sh $D/llr5.site.env llr5-c80   # KF kernels                          ~95 min
bash $D/run_llr.registry.sh $D/llr6.site.env llr6-c80   # + DECODE_CUSTOM_AR=1
bash $D/run_llr.registry.sh $D/llr7.site.env llr7-c80   # + SGLANG_OPT_USE_TOPK_V2=false (decode only)
python3 -I $D/llsum.registry.py $D/work/runs/llr7-c80   # summary row
python3 -I $D/bneck.registry.py $D/work/runs/llr7-c80   # steady-state decode / prefill queue stats
```

`run_llr.registry.sh` waits until neither node runs a container, removes only our exited `historical-e2e-container-*` containers (never
`-v`), runs `up.sh` + `agentx.sh`, copies the container logs to `$WORK_DIR/runs/<run>/logs/`, then `docker stop`s.

Check after start (decode, REDACTED_NODE): `docker exec historical-e2e-container-decode env | grep TOPK_V2` -> `false` for llr7;
`docker logs historical-e2e-container-decode | grep infera-kernels` -> `installed` lines, `CHECK ... snr` lines, and per-shape
`indexer_logits/indexer_topk rows=N -> KF|orig` lines (see notes.md for the expected pattern);
`disable_custom_all_reduce': False` in the server-args line for llr6/llr7.

Probe (optional, during the profiling phase, on each node): `bash node_probe.registry.sh <outprefix> 120`.
