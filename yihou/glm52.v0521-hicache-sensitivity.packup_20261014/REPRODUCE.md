# Reproduce round hc50 (e2g recipe, prefill HiCache 50 GB/rank, AgentX C40 A1 fresh -> flush -> A2 scored)

All paths are the live session paths (absolute paths are baked into the harness). `scripts/harness/` is a verbatim
copy taken 2026-10-14 ~02:00 UTC; run from the original location `H` or copy the tree and `sed` the WS root as in the
baseline kit's REPRODUCE §2.

```bash
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness      # harness root (= scripts/harness/ here)
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs         # run outputs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013             # session workspace (shim, metrics TSVs, tools)
WS=/shared_nfs/yihou_glm52/v0521dbg_20261010             # caches live in $WS/cache/<node>
PK=/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.v0521-hicache-sensitivity.packup_20261014   # this packup
```

Harness-external files the scripts call by absolute path (not under `$H`): `$N/shimbin/ssh` (on-node shim, copied as
`scripts/harness/shimbin/ssh`), `/shared_nfs/yihou_glm52/v0521cap_20261012/tools/metrics_sampler.yihou.py` (copied as
`scripts/tools/metrics_sampler.yihou.py`), and `/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin/ssh` (5,461 B;
`pipeline.v3.yihou.sh:17`, used only when a spur-job pipeline runs OFF the node; not used by hc50; not copied).
`run.v3.yihou.sh:306` calls `$H/tools/pidcensus.yihou.sh` (census step) — copied as `scripts/harness/tools/pidcensus.yihou.sh`
at verification (2026-10-14).

## 0. Access

- Login host `spur006` (`crsuse2-slog-006`), cluster SSH from the workstation (`ssh spur006`). No other credential is
  needed: images are local/NFS tarballs, the model is on NFS, the AgentX dataset is cached under
  `$WS/cache/<node>/agentx` (`HF_HOME`); if that cache is missing, the aiperf run downloads the public dataset
  `semianalysisai/cc-traces-weka-062126` (no token used in our runs — `runtime.env` has no token variable).
- 136 was used by plain ssh (user-approved, no slurm job). For a spur node, hold it first:
  `sbatch -p amd-oai -A amd-oai -q amd-oai-qos --gres=gpu:8 -c 160 ...` (see the `spur-cluster` skill / earlier kits).

## 1. Verify the node (read-only checks)

```bash
ssh crsuse2-m2m-136 'rocm-smi --showpids; rocm-smi --showmemuse; docker ps --format "{{.Names}} {{.Image}} {{.Status}}"'
ssh crsuse2-m2m-136 'rocm-smi --showrasinfo all 2>/dev/null | grep -iE "correctable|deferred" | sort | uniq -c'   # ECC (026 had 13.2M on GPU3 -> released)
```
Want: 0 KFD processes, VRAM idle (run.v3 `gpu_idle_gate`: max used VRAM < 400 MiB on 2 checks >= 60 s apart; hc50
passed at 283 MiB), no foreign container driving the GPUs. The ECC command above is [inferred syntax, not run here]. At hc50 time the foreign container `dev_xb` (lmsysorg/sglang:v0.5.20-rocm724-mi35x, up
since ~2026-10-09) was present on 136. run.v3.yihou.sh:137-138 calls it "a foreign looping job (136 dev_xb) [with]
short gaps"; the chain's KFD check (0 processes) and the GPU-idle gate passed at start, but whether dev_xb used the
GPUs or CPUs during the 3 h of hc50 was **not measured**.
**Before re-running on 136: the hc50 stack itself was still running at 2026-10-14 01:57 UTC**
(`glm52-v0521-yihou-136-hc50-{etcd,prefill-0,decode-0,router}`); stop it with
`cd $H && source env.hc50-136.yihou.sh && bash run.v3.yihou.sh stop` (stops, never removes) and wait 28-35 min for VRAM
to drain after a HiCache stack (ledger note).

## 2. Image

- 136: its own build, `infera-sglang:v0521-yihou-1010-mc4340` ID `sha256:9872bbdabfcb8f6676b04378d8ea75c922dfc8db6a8c5c87acd801e205b3b575`
  (verified 2026-10-14 01:57 in `env/collect_env.136.20261014.txt`). `config.yihou.mc4340n.sh` pins this ID for 136.
- Any other node: load the 102 build from NFS:
  `ssh <node> 'zstd -dc /shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst | docker load'`
  then `docker image inspect --format '{{.Id}}' infera-sglang:v0521-yihou-1010-mc4340`. 026/085/274/206/089 got
  9d755dd6 and are pinned in `config.yihou.mc4340n4.sh`; 137 re-IDs to e8e2838f. If the node is not pinned, write a
  NEW config copy with a `<node>) EXPECTED_IMAGE_ID=<id> ;;` line (never edit a config another round is using).
  For a like-for-like comparison with e2g use a 9d755dd6 node (CONFIG=config.yihou.mc4340n4.sh).

## 3. JIT caches (saves ~30 min of AITER/triton JIT)

Plain-ssh nodes whose `/shared_nfs` is read-only use `NODE_RW_ROOT=/mnt/m2m_nobackup/yihou_v0521dbg` (default in
`config.yihou.v0521-base.sh:166-167` whenever `YIHOU_JOB` is empty; caches at `/mnt/m2m_nobackup/yihou_v0521dbg/cache/136/{aiter,triton}`, already populated on
136). New spur node: `cp -a $WS/cache/102/aiter $WS/cache/102/triton $WS/cache/<node>/` (keyed by image ID; the 102
cache matches 9d755dd6 only).

## 4. Env file

`scripts/harness/env.hc50-136.yihou.sh` (copy it to a new name for a new node/round and edit YIHOU_NODE/YIHOU_IP/
YIHOU_JOB/YIHOU_ROUND only):
```bash
export CONFIG=$H/config.yihou.mc4340n.sh
export YIHOU_OVERRIDES="MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh CONC=40 PREFILL_HICACHE_SIZE=50 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000"
```
The e2g reference env is `env.e2g-102.yihou.sh` (same overrides without `PREFILL_HICACHE_SIZE=50`; base default 450).
Note: e2g used `CONFIG=config.yihou.mc4340.sh`, hc50 `config.yihou.mc4340n.sh`; `diff` of the two (checked at
verification) is only a header comment plus `EXPECTED_IMAGE_ID` lines for nodes 174/029/137, so for 102/136 they are
identical. e2g's A2 was driven by a separate post-chain step (`results/reference-e2g/e2g.A2.log`: flush :29357 then
:29101 at 07:16:29Z, A2 done 08:35:29Z), i.e. the same flush order as `chain_cap2l`, which did not exist yet.
Config chain: `config.yihou.mc4340n.sh` -> `config.yihou.E2-target.sh` -> `config.yihou.v0521-base.sh`; mounts
`mounts.yihou.v2-regbase.tsv` (3 files in `patches/mounts-v0521/src/`).

## 5. Dryrun + token diff against the reference

```bash
cd $H && ( source env.hc50-136.yihou.sh && bash run.v3.yihou.sh dryrun ) > /tmp/yihou/dryrun.hc50.mine.txt 2>&1
diff $PK/results/dryrun/dryrun.hc50-136.yihou.txt /tmp/yihou/dryrun.hc50.mine.txt          # expect: nothing but host name
bash $H/tokdiff.yihou.sh $PK/results/dryrun/dryrun.e2g-102.yihou.txt /tmp/yihou/dryrun.hc50.mine.txt
```
Expected token diff vs e2g (`results/dryrun/tokdiff.e2g-102_vs_hc50-136.txt`): `--hicache-size 450 -> 50` on the
prefill leg, plus node IPs and the read-only NFS / node-local cache mounts. Nothing else.

## 6. Launch (driver detached)

- Plain-ssh node (136), driver on the login host:
  ```bash
  ssh spur006 "cd /; setsid nohup bash $H/chain_cap2l.yihou.sh $H/env.hc50-136.yihou.sh 40 40 > $R/hc50.chain.log 2>&1 < /dev/null &"
  ```
- Held spur node (`YIHOU_JOB` set), driver on the node with the local ssh shim (`$N/shimbin/ssh`, = `scripts/harness/shimbin/ssh`):
  ```bash
  spur exec <job> bash -c "cd /; setsid nohup bash $H/chain_cap2.yihou.sh $H/env.<round>.yihou.sh 40 40 > $R/<round>.chain.log 2>&1 < /dev/null &"
  ```
Chain = pipeline.v3 (gate -> etcd -> prefill -> decode -> router -> census -> smoke -> audit v3 -> A1 C40 fresh) ->
`POST :29357/flush_cache` then `:29101/flush_cache` -> A2 C40 (scored). A `/metrics` sampler (5 s) per bench writes
`$N/runs/<round>/metrics.A{1,2}.tsv`. hc50 timing: prefill up 15 min, decode 14 min, each bench ~79-84 min; total
~3 h 12 min. The stack is left running at the end.

Check live: `tail $R/<round>.chain.log $R/<round>/status.tsv`; census must show
`Allocating kv hierarchical KV host pool: 1112896 tokens, 50.00 GB host memory` x4 (`results/hc50/round/census.txt`).

## 7. Read the score

```bash
ssh spur006 "python3 -B $N/tools_cmp.yihou.py 'hc50 A2=hc50/agentx-c40-A2/agentx_conc40.json' 'e2g A2=e2g/agentx-c40-A2/agentx_conc40.json'"
ssh spur006 "python3 -B $N/tools_summ.yihou.py"                       # full session table incl. GPU/CPU hit
ssh spur006 "cd $PK && python3 -B scripts/tools/extract_results.yihou.py > /tmp/yihou/results.csv"
```
Expected (this run): A1 17,733, A2 20,112 tok/s/chip; GPU hit 0.930, CPU hit 0.000.

## 8. Stop

`cd $H && source env.<round>.yihou.sh && bash run.v3.yihou.sh stop` (docker stop only). Server logs on a plain-ssh node
stay on the node mirror `/mnt/m2m_nobackup/yihou_v0521dbg/runs/<round>/server-logs/`; the login-side copy under
`$R/<round>/server-logs/` is a snapshot from bring-up time (see `logs/LOG_INDEX.md`).
