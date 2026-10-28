> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# REPRODUCE

Paths on cluster NFS (run login-side commands on login-node-a/login-node-b; keep them < 4 GB):
```bash
H=REDACTED_LOCAL_PATH          # shared harness (configs, bench-harness)
K=REDACTED_LOCAL_PATH KH=$K/harness       # this work: overlay, envs, private harness copy
R=REDACTED_LOCAL_PATH
W=REDACTED_LOCAL_PATH         # from workspace-node: login failover + remote-launcher exec
```

## 0. Files onto cluster NFS
`overlay/` -> `$K/kf/e2e/overlay`; `scripts/kernel/*` -> `$K/kf/e2e/`; KF task drivers + data
(`REDACTED_LOCAL_PATH`) ->
`$K/kf/tasks/`. In `$K/kf/tasks/sparse_mla/sparse_mla_kernel.py` make the `q8kv8_topk_length_from_indices` import fall
back to `sglang.kernels.ops.attention.metadata` (0.5.22 moved it). Upload from workspace-node: `ssh login-node-a 'cat > <file>' < <local>`.
`cluster/kf_e2e_20261023/harness/*` -> `$KH/`, `cluster/kf_e2e_20261023/env.*` -> `$K/`.

## 1. Node prep (per node, root via remote-launcher exec)
```bash
$W 'remote-launcher exec <job> bash -c "docker load -i REDACTED_LOCAL_PATH"'
# expect ID sha256:5b6565ef...; seed the aiter JIT cache under that key (see environment.md), e.g. for node key 190:
C=REDACTED_LOCAL_PATH for r in prefill decode; do cp -a $C/REDACTED_NODE/aiter/$r/437497f5b1f73203228a5c388ba1004ecce194d1a2050d8b49cfa47280923221 $C/190/aiter/$r/5b6565ef3e4e7dcca02423d509f842a74cf7416b18ed94edbf12e2be100f7376; done
```
A new node also needs its name in `$KH/config.registry.mc4340n19.kf.sh` (case list) and an env file with its IP/job.

## 2. Kernel-level A/B in the target stack (no server; ~10 min, GPU 0)
```bash
$W 'remote-launcher exec 234200 bash -c "cd / && setsid nohup bash $K/kbench_run.registry.sh kb2 sparse,indexer,moe 0 >/dev/null 2>&1 </dev/null &"'
# -> $K/kbench.kb2.log ; prints prod vs KF ms per call (CUDA graph) + correctness per case
```

## 3. End-to-end rounds (one variable: KF_ON)
```bash
# dryrun + token diff (must differ only by the overlay mount + 4 env vars)
cd $KH && (source $K/env.kfa1-190.registry.sh && bash $KH/run.v3.registry.sh dryrun > $K/dryrun.kfa1-190.registry.txt)
cd $KH && (source $K/env.kfb1-169.registry.sh && bash $KH/run.v3.registry.sh dryrun > $K/dryrun.kfb1-169.registry.txt)
# a round = up -> census -> smoke -> audit -> A1 (fresh) -> flush both legs -> A2 (warm), stack left running
$W 'remote-launcher exec 234200 bash -c "cd / && setsid nohup bash $KH/chain_cap2.registry.sh $K/env.kfa1-190.registry.sh 56 56 > $R/kfa1.chain.log 2>&1 </dev/null &"'
# stop a round's stack (on the node; local ssh shim first on PATH)
$W 'remote-launcher exec 234200 bash -c "export PATH=REDACTED_LOCAL_PATH cd $KH && source $K/env.kfa1-190.registry.sh && bash $KH/run.v3.registry.sh stop"'
```
Rounds of record: kfa1 (190, KF off), kfb1 (169, KF on) in parallel; then crossed: kfb2 (190, on), kfa2 (169, off).
kfa1 ran its benches through `chain_cont.registry.sh` (bench-only continuation) after the remote-launcherious audit stop (notes 2).
Round names must be new (stopped containers keep their names).

## 4. Analysis
```bash
python3 scripts/summary.registry.py cluster/v0521dbg_20261010/runs > results/summary.tsv
python3 $K/bsdist.registry.py $R/<round>/server-logs/decode-0.a1.log HH:MM:SS HH:MM:SS     # per-rank decode bs
grep -a -o "\[infera-kernels\] pid=[0-9]* [a-z_]* \(rows\|M\)=[0-9]*[^>]*-> [A-Za-z]*" <decode log> | ... | sort | uniq -c   # engagement
```

## compute-cluster cluster (v0.5.21, single server, no PD) — bring-up only
`scripts/compute-cluster/serve.registry.sh` (TAG, SLOT a|b, KF=0|1, SIM) + `agentx.registry.sh`; image
`historical-sglang-image-d`, model `REDACTED_LOCAL_PATH`. Start the second
server only after the first is up (notes 5). The TARGET_VERIFY gate (notes 3) was added after these runs.
