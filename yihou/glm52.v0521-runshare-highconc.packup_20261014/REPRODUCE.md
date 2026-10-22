# Reproduce tb56 (runshare v2) and tb56n (no patch) — AgentX C56 A1 -> C56 A2 -> C64 A3 on one warm stack

Conventions (absolute paths as used at run time; the harness has them baked in):
```
H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness     # harness (copy: scripts/harness/)
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs        # round outputs
N=/shared_nfs/yihou_glm52/v0521cap2_20261013            # session workspace (shim, metrics, dryruns; copy: scripts/session/)
CAP=/shared_nfs/yihou_glm52/v0521cap_20261012           # image tar, metrics sampler, extracted image sglang source
WS=/shared_nfs/yihou_glm52/v0521dbg_20261010            # $WS/cache/<node>/{aiter,triton,...} JIT caches
```
If those NFS paths are gone, copy `scripts/harness/` to `$H`, `scripts/session/shimbin/ssh` to `$N/shimbin/ssh`,
`scripts/session/metrics_sampler.yihou.py` to `$CAP/tools/` (or sed the paths in `chain_cap2.yihou.sh`).
Drivers run from a spur login host (`ssh spur006`); keep login-host commands light (serial, no heavy python).

## 1. Hold a node (spur, amd-oai)

```bash
# from spur006; 24 h, 8x MI355X, 160 cores (the hold used for this packup: job 211169 -> crsuse2-m2m-085).
# Form per the spur-cluster skill (the exact 2026-10-13 submit line was not recorded); ~/spur-hold/hold_node.sh sleeps $HOLD_SECONDS (verbatim copy: scripts/session/hold_node.sh).
HOLD_SECONDS=86400 sbatch --parsable --export=ALL -A amd-oai -p amd-spur -q amd-oai-qos -N1 --gres=gpu:8 -c 160 -t 24:00:00 -J yihou-oai-cap2 ~/spur-hold/hold_node.sh
squeue -j $JOB -h -o "%T %N %R"           # wait for RUNNING; then prove GPUs:
spur exec $JOB bash -c 'hostname; rocm-smi --showproductname | grep -c "Card Series"'   # 8
```
`amd-oai-qos` has a group node limit (QOSGrpNodeLimit pending is normal). Burst QoS (`amd-burst-qos`, PreemptMode
cancel) killed 3 h rounds at 20 min on 2026-10-12 — do not use it for this.

## 2. Verify the node before use (all via `spur exec <JOB> ...`)

```bash
spur exec $JOB bash -c 'rocm-smi --showpids; rocm-smi --showmeminfo vram | grep Used; docker ps -a'
spur exec $JOB bash -c 'amd-smi metric --ecc | grep -iE "gpu|correctable|uncorrectable|deferred"'   # all 0 expected
```
- Require: 0 KFD processes, no foreign GPU containers running. On 2026-10-13 085 and 274 carried another user's orphan
  container `crusoe-sgl-q36` (a loop that can relaunch sglang on GPU0-5); it was **docker stop**-ped (not removed)
  with the user's approval. Never stop a foreign container without asking.
- ECC: node 026 (same day) had GPU3 HBM 13,198,148 correctable + 2 deferred and crashed a round with "Memory access
  fault ... DRAM ECC failure" — check before trusting any number.

## 3. Load the image

```bash
spur exec $JOB bash -c 'zstd -dc /shared_nfs/yihou_glm52/v0521cap_20261012/images/infera-sglang.v0521-yihou-1010-mc4340.9d755dd6.yihou.tar.zst | docker load'
spur exec $JOB bash -c 'docker image inspect -f "{{.Id}}" infera-sglang:v0521-yihou-1010-mc4340'
# expected on 085/274/026/206/089: sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2
```
On 085 the load took 05:10 -> 05:14 UTC [M `$N/logs/imgload.crsuse2-m2m-085.yihou.log`]. If the ID differs (137 got
`e8e2838f…` from the same tar), add a `case` line for the node to a NEW copy of `config.yihou.mc4340n2.sh` — the config
refuses unknown nodes, and `run.v3` refuses a mismatching image ID.

## 4. Seed the JIT caches (saves ~30 min of AITER/Triton compile)

```bash
# [inferred: exact command not logged; $N/logs/seed.085.yihou.log only records SEED_OK 2026-10-13T06:26:33Z;
#  mission ledger debug/working_process.md:162 says "JIT caches seeded from cache/102 (aiter 9d755dd6 + triton)"]
spur exec $JOB bash -c 'mkdir -p /shared_nfs/yihou_glm52/v0521dbg_20261010/cache/085 && cp -a /shared_nfs/yihou_glm52/v0521dbg_20261010/cache/102/aiter /shared_nfs/yihou_glm52/v0521dbg_20261010/cache/102/triton /shared_nfs/yihou_glm52/v0521dbg_20261010/cache/085/'
```
Per-role caches are then used as `cache/<node>/aiter/<role>/<hash>` and `cache/<node>/triton/<role>` (see dryrun).

## 5. Env files (one per round)

Use `scripts/harness/env.tb56-085.yihou.sh` and `env.tb56n-085.yihou.sh` verbatim, or retarget
`YIHOU_NODE / YIHOU_IP / YIHOU_JOB / YIHOU_ROUND` for another node (new file; never edit a file a running driver reads).
- tb56: `CONFIG=$H/config.yihou.runshare2.sh` with `RUNSHARE_BASE_CONFIG=$H/config.yihou.mc4340n2.sh`,
  `MC4340_BASE_CONFIG=$H/config.yihou.E2-target.sh`, `RUNNING_KV_SHARE=0.5`.
- tb56n: `CONFIG=$H/config.yihou.mc4340n2.sh` directly.
- Common overrides: `CONC=56 PREFILL_HICACHE_SIZE=0 PREFILL_HICACHE_RATIO=1.5 DECODE_MAX_RUNNING=96
  PREFILL_MEM_FRACTION=0.88 DECODE_MEM_FRACTION=0.95 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000`.

Also needed per node: `$H/topology.yihou.<node suffix>.tsv` (copy: `scripts/harness/topology.yihou.085.tsv` — two
rows, prefill and decode, same node + IP) and the AgentX client checkout `$WS/cache/InferenceX` (a symlink to
`/shared_nfs/yihou_glm52/agentx44_20261009/cache/InferenceX`, git HEAD `918524ff94045b3f091115f1051c22a8588edf2b`).

Config chain: `runshare2` -> `mc4340n2` (image + per-node ID + YIHOU_MOUNTS=`mounts.yihou.v2-regbase.tsv`) ->
`E2-target` (H1 HiCache chunk 512, index_share false, decode mem 0.95) -> `v0521-base` -> `bench-harness/config.sh`.

## 6. Dryrun and token diff (must show ONLY the intended delta)

```bash
cd $H && ( source env.tb56-085.yihou.sh && bash run.v3.yihou.sh dryrun ) > /tmp/yihou/dryrun.tb56.txt
cd $H && ( source env.tb56n-085.yihou.sh && bash run.v3.yihou.sh dryrun ) > /tmp/yihou/dryrun.tb56n.txt
bash $H/tokdiff.yihou.sh /tmp/yihou/dryrun.tb56.txt /tmp/yihou/dryrun.tb56n.txt
```
Reference renderings: `scripts/session/dryrun.tb56-085.yihou.txt`, `dryrun.tb56n-085.yihou.txt`. Their diff [M] =
the decode mount `decode.yihou-runshare2.py:/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py:ro` and
decode env `YIHOU_DECODE_RUNNING_KV_SHARE=0.5 YIHOU_RUNSHARE_MAX_WAIT_S=5` (+ round name in paths/container names).
vs q1 (`results/reference/q1-137/dryrun.q1-137.yihou.txt`): no `--hicache-size` (ratio 1.5) + node paths/IP.

## 7. Launch the chain ON the node (detached; survives login-host loss)

```bash
# tb56 (stack fresh):
spur exec 211169 bash -c "cd /; setsid nohup bash $H/chain_cap2.yihou.sh $H/env.tb56-085.yihou.sh 56 56 64 > $R/tb56.chain.log 2>&1 < /dev/null &"
# tb56n (stops the finished tb56 stack, waits for 0 KFD procs, then starts the new chain):
spur exec 211169 bash -c "cd /; setsid nohup bash $H/stop_then_chain.yihou.sh $H/env.tb56-085.yihou.sh $H/env.tb56n-085.yihou.sh 56 56 64 > $R/tb56n.handover.log 2>&1 < /dev/null &"
```
`chain_cap2` puts `$N/shimbin/ssh` first on PATH (the harness' `ssh <node> cmd` runs locally, other hosts refused),
refuses to start with any KFD process present, then: `pipeline.v3` (up -> census -> smoke -> audit v3 -> A1 bench at
C1) -> flush decode `:29357/flush_cache` then prefill `:29101/flush_cache` -> A2 at C2 -> flush -> A3 at C3. One
/metrics sampler (5 s, 5400 s) per bench into `$N/runs/<round>/metrics.<A1|A2|A3>.tsv`. AgentX: 3600 s, warmup 10/lane.
Wall time per round ~4 h 50 min (bring-up ~17 min, each bench 86-101 min).

Watch: `$R/<round>.chain.log`, `$R/<round>.pipeline.log`, `$R/<round>/status.tsv`; faults:
`grep -a -cE "Traceback|Memory access fault" $R/<round>/server-logs/*.log`.
Verify the patch is live (tb56 only): `grep -a -c "YIHOU-RUNSHARE block" $R/tb56/server-logs/decode-0.a1.log` > 0;
`grep -a -o "pre-allocated usage: [0-9.]*" ... | sort -t: -k2 -n | tail -1` <= 0.50.

## 8. Read the score

```bash
python3 -B $N/tools_summ.yihou.py            # session-wide table (tb56/tb56n rows)
python3 -B $N/tools_cmp.yihou.py "tb56 A2=tb56/agentx-c56-A2/agentx_conc56.json" "tb56n A2=tb56n/agentx-c56-A2/agentx_conc56.json"
python3 -I scripts/make_results_csv.yihou.py <this packup dir>   # rebuilds results/results.csv
```
Expected (this packup): tb56 A1/A2/A3 = 17,545 / 17,611 / 15,070; tb56n = 17,791 / 18,016 / 15,108 tok/s/chip.

## 9. Stop

```bash
spur exec $JOB bash -c "cd $H && source env.tb56n-085.yihou.sh && PATH=$N/shimbin:\$PATH bash run.v3.yihou.sh stop"
```
`stop` removes only this round's containers (`glm52-v0521-yihou-085-<round>-*`). VRAM drain after a stack stop can take
tens of minutes; `stop_then_chain` waits for 0 KFD processes (up to 30 min).

## Secrets / access needed (no values in this packup)

- spur cluster login (ssh to `spur006` / `crs-spur.crusoe.amd.com`) with the `amd-oai` account/QoS.
- No registry login (image from the NFS tar), no HF token (model on NFS), no API keys (server-info shows `api_key: null`).
