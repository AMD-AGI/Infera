# Reproduce — pc48b (prefill_complete, C48) and its control pc48e (early)

Wall time per round on a node with the image loaded and JIT caches seeded: VRAM drain of a previous stack (10-30 min) +
bring-up (pc48b 13 min, pc48e 32 min) + A1 (≈84-90 min incl. warmup) + flush + A2 (≈84 min). Expect (single replicates on
274): pc48b A1 ≈18.2k / A2 ≈20.0k; pc48e A1 ≈19.3k / A2 ≈19.2k tok/s/chip.

Conventions used below (all absolute; these directories still exist on the shared NFS):
```bash
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness      # harness (copied verbatim to scripts/harness/)
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs         # round outputs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013             # session workspace (shim, mounts/pcomplete, metrics, dryruns)
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012            # metrics sampler, image tar, extracted sglang source
WS=/shared_nfs/yihou_glm52/v0521dbg_20261010             # $WS/cache/<node-key>/{aiter,triton} JIT caches
```
The scripts have these paths baked in. To run from a copy of this kit instead, copy `scripts/harness/*` to a new
directory and `sed` `H`/`WS` in the env files and configs (same procedure as the baseline kit REPRODUCE §2).

## 0. Prerequisites

- Cluster login to spur (login host `spur006` = `crsuse2-slog-006`), account/QoS `amd-oai` / `amd-oai-qos`. **Secrets:** none
  beyond the cluster login (model on NFS, image as a tar on NFS, AgentX dataset public on HF; no tokens in any env file).
- Model: `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (unmodified, 256 experts), served as `glm5.2-mxfp4`.
- Image tar: `$CAP/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst` (23.5 GB). Rebuild recipe: baseline kit
  `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/REPRODUCE.md` §1.

## 1. Hold a node (from the login host)

```bash
ssh spur006
sbatch -A amd-oai -p amd-spur -q amd-oai-qos -N 1 --gres=gpu:8 -c 160 -t 24:00:00 -J yihou-oai \
       <baseline-kit>/scripts/alloc/hold_inf.yihou.sh          # exec sleep infinity; see baseline kit REPRODUCE §0
squeue -u $USER                                              # note JOBID and NODE; data IP = the node's ens3 address
```
Used here: job 211171 on `crsuse2-m2m-274` (10.245.157.34).

## 2. Verify the node (via `spur exec <job>`; spur nodes reject plain ssh)

```bash
J=<jobid>
spur exec $J bash -c 'hostname; rocm-smi --showpids; rocm-smi --showmeminfo vram | grep Used; docker ps -a'
spur exec $J bash -c 'amd-smi metric --ecc'      # all GPUs must show 0 uncorrectable/deferred; node 026 was rejected for
                                                 # GPU3 UMC correctable 13,198,148 + 2 deferred (crashed a run)
```
Require: 0 KFD processes, all 8 GPUs < 400 MiB used, no foreign containers holding GPUs. Foreign containers are **not ours to
stop** without the owner's / user's approval (274 had `crusoe-sgl-q36`, docker-stopped — not removed — with user approval).

## 3. Load the image and seed JIT caches

```bash
spur exec $J bash -c 'zstd -dc /shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst | docker load'
spur exec $J bash -c 'docker image inspect -f {{.Id}} infera-sglang:v0521-yihou-1010-mc4340'
#   274 / 085 / 026 / 206 / 089 kept sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2 (137 re-IDed to e8e2838f).
```
If the node is not in `scripts/harness/config.yihou.mc4340n2.sh` (`case "$YIHOU_NODE"`), add a **new** config copy with
`<node>) EXPECTED_IMAGE_ID=<id>` (the session did this as `mc4340n3` / `mc4340n4`; never edit a config in use).
Seed the JIT caches from node 102 (saves ≈30 min of AITER/Triton compiles) [inferred command; the session log only
records `SEED_OK`]:
```bash
K=<last 3 digits of node>     # e.g. 274; the harness uses $WS/cache/$K
mkdir -p $WS/cache/$K && cp -a $WS/cache/102/aiter $WS/cache/102/triton $WS/cache/$K/
```

## 4. Write the env file

Copy `scripts/session/env.pc48-NODE.template` → `$H/env.pc48b-<K>.yihou.sh`, replace `__NODE__ __IP__ __JOB__`, set
`YIHOU_ROUND=pc48b`, and append the hook mounts to `YIHOU_OVERRIDES` exactly as in `scripts/harness/env.pc48b-274.yihou.sh`:
```
PREFILL_MOUNTS_ADD=$H/mounts.yihou.pcomplete-attndp.tsv DECODE_MOUNTS_ADD=$H/mounts.yihou.pcomplete-attndp.tsv
```
(the TSV mounts `$N/mounts/pcomplete/pd_disaggregation_hook.yihou-attndp.py` = `patches/pd_disaggregation_hook.yihou-attndp.py`
over `/sgl-workspace/sglang/python/sglang/srt/arg_groups/pd_disaggregation_hook.py`, read-only).
Control round: `scripts/harness/env.pc48e-274.yihou.sh` (CONFIG `config.yihou.mc4340n2.sh`, `PREFILL_HICACHE_WRITE_POLICY=write_back`,
no policy flag, no hook mount).

Effective knobs (from the env + config chain): `CONFIG=config.yihou.pcomplete.sh` (adds
`--disaggregation-decode-allocation-policy prefill_complete` to both legs, forces `PREFILL_HICACHE_WRITE_POLICY=write_back`) →
`PCOMPLETE_BASE_CONFIG=config.yihou.mc4340n2.sh` (image + REGBASE mounts) → `MC4340_BASE_CONFIG=config.yihou.E2-target.sh` →
`config.yihou.v0521-base.sh`; overrides `CONC=48 PREFILL_HICACHE_SIZE=370 DECODE_MAX_RUNNING=96 PREFILL_MEM_FRACTION=0.88
DECODE_MEM_FRACTION=0.96 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000`.

## 5. Dryrun and token-diff against the reference (login host or node)

```bash
cd $H && source env.pc48b-<K>.yihou.sh && bash run.v3.yihou.sh dryrun > $N/dryrun.pc48b-<K>.yihou.txt
bash $H/tokdiff.yihou.sh $H/dryrun.q1-137.yihou.txt $N/dryrun.pc48b-<K>.yihou.txt
```
Expected diff (see `results/dryruns/tokdiff.yihou.txt`): node paths/IP only, plus prefill `write_through→write_back`,
`450→370`, decode `0.95→0.96`, the policy flag on both legs, and the hook mount on both legs. Anything else = stop.
pc48b vs pc48e must differ only by the hook mount + policy flag on each leg.

## 6. Launch (detached ON the node; the local ssh shim makes the harness' `ssh <node> cmd` run locally)

```bash
spur exec $J bash -c "cd /; setsid nohup bash $H/chain_cap2.yihou.sh $H/env.pc48b-<K>.yihou.sh 48 48 \
    > $R/pc48b.chain.log 2>&1 < /dev/null &"
```
`chain_cap2` = `pipeline.v3` (gpuidle → etcd → prefill → decode → router → census → smoke → audit v3 → A1 fresh C48) →
flush decode `:29357` then prefill `:29101` `/flush_cache` → A2 C48 (scored). One `/metrics` sampler (5 s) per bench into
`$N/runs/<round>/metrics.<A1|A2>.tsv`. Stack is left running.

**Known audit false negative (happened in pc48b):** the same TSV in both `PREFILL_MOUNTS_ADD` and `DECODE_MOUNTS_ADD` trips the
audit's role-only rule → `AUDIT FAIL (2)` and `STOP at audit` (93 other checks PASS). Either put both-leg mounts in
`YIHOU_MOUNTS` (via a new config) — the fix the session recommended but did not test — or, as was done, verify the live stack
and continue the benches on it:
```bash
python3 -c 'import json;d=json.load(open("'$R'/pc48b/audit/prefill-server-info.json"));print(d["disaggregation_decode_allocation_policy"],d["optimistic_prefill_attempts"],d["hicache_write_policy"],d["hicache_size"])'
# expect: prefill_complete 1 write_back 370   (decode-server-info: prefill_complete, mem 0.96, max_total_num_tokens 2816896)
spur exec $J bash -c "cd /; setsid nohup bash $H/cont_bench.yihou.sh $H/env.pc48b-<K>.yihou.sh 48 48 \
    >> $R/pc48b.chain.log 2>&1 < /dev/null &"
```
Control: after pc48b, stop it and start pc48e on the same node:
```bash
spur exec $J bash -c "cd /; setsid nohup bash $H/stop_then_chain.yihou.sh $H/env.pc48b-<K>.yihou.sh $H/env.pc48e-<K>.yihou.sh 48 48 \
    > $R/pc48e.handover.log 2>&1 < /dev/null &"
```

## 7. Read the scores

```bash
ssh spur006 "python3 -B $N/tools_cmp.yihou.py 'pc48b A1=pc48b/agentx-c48-A1/agentx_conc48.json' 'pc48b A2=pc48b/agentx-c48-A2/agentx_conc48.json' \
                                           'pc48e A1=pc48e/agentx-c48-A1/agentx_conc48.json' 'pc48e A2=pc48e/agentx-c48-A2/agentx_conc48.json'"
```
Score = `request_metrics.throughput.per_gpu.total_tput_tps`; A2 is the scored number. Full table incl. GPU/CPU hit:
`python3 -B $N/tools_summ.yihou.py` (or `scripts/tools/results_csv.yihou.py`).

## 8. Log statistics (optional)

```bash
awk -v W1S=<A1 start HH:MM:SS> -v W1E=<A1 end> -v W2S=<A2 start> -v W2E=<A2 end> -v TAG=pc48b \
    -f scripts/tools/logstats.yihou.awk $R/pc48b/server-logs/prefill-0.log     # and decode-0.a1.log
```
Windows = the `bench start` / `bench ok` timestamps in `$R/<round>/status.tsv`.

## 9. Stop

```bash
spur exec $J bash -c "cd $H && source env.pc48b-<K>.yihou.sh && PATH=$N/shimbin:\$PATH bash run.v3.yihou.sh stop"
```
VRAM release after a HiCache-370 stack takes ≈10 min (pc48e gpuidle wait); the next pipeline gates on it.
