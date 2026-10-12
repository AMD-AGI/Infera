# Reproduction kit — v0.5.21 target config, AgentX full C40, warm-stack protocol (round e2g A2, 22,114 tok/s/chip)

Goal: reproduce **e2g A2** — `AgentX passed`, CONC 40, 3600 s profiling, ≈22k tok/s/chip — on one MI355X node.
Wall time with the image present and warm JIT caches: ≈3 h 15 min =
VRAM drain after a previous HiCache-450 stack (10-35 min) + bring-up ≈14 min + **A1** (warmup 15-18 min + 60 min
profiling + ≈4 min processing) + flush (≈1 min) + **A2** (warmup ≈13 min + 60 min + ≈4 min).
Expect A1 ≈ 19.3-20.4k (not scored) and A2 ≈ 21.7-22.1k (scored). Single-run spread of the warm protocol: ±1 %.

## 0. Prerequisites

- **Machine:** one 8× MI355X node in partition `amd-spur`. Proven node: `crsuse2-m2m-102` (data IP `10.245.147.130`).
  Hold it with a script that `exec sleep infinity`s (so `scontrol update TimeLimit=` really extends it — a
  `sleep 86400` hold ends at 24 h regardless):
  ```bash
  sbatch -A amd-oai -p amd-spur -q amd-oai-qos -N 1 -w crsuse2-m2m-102 --gres=gpu:8 -c 160 -t 24:00:00 \
         -J yihou-oai-go3 scripts/alloc/hold_inf.yihou.sh
  scontrol update JobId=<jobid> TimeLimit=2-00:00:00      # extend later if needed
  ```
  To hand a node over to a new job without losing it: submit the new hold with `--dependency=afterany:<old>` first.
- **Node access:** spur nodes reject plain ssh. Use the spur exec shim `scripts/alloc/shimbin-ssh` (deployed at
  `/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin/ssh`) with `SPUR_SSH_JOBS=<node>:<jobid>`; the harness sets
  this itself from `YIHOU_NODE`/`YIHOU_JOB`.
- **Docker policy:** spur-authz denies `--pid=host`; allows `--cpuset-cpus`, `--cap-add SYS_PTRACE` + `--pid container:`.
- **Secrets:** none beyond your cluster login. Base image, InferenceX repo and HF dataset are public; model on NFS.
- **External absolute paths (not in this kit):**
  - model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (256 experts, unmodified), served as `glm5.2-mxfp4`;
  - workspace `WS=/shared_nfs/yihou_glm52/v0521dbg_20261010` (original harness, runs, JIT caches — still on disk);
  - image build context: clone `/shared_nfs/yihou_glm52/agentx44_20261009/infera-src.yihou`, branch
    `yihou.agentx44.20261009` (see `environment.md`);
  - InferenceX cache `/shared_nfs/yihou_glm52/agentx44_20261009/cache/InferenceX` (@ `918524ff`).
- **Node state:** all 8 GPUs idle (< 400 MiB each, two checks 60 s apart — enforced by the harness).

## 1. Image `infera-sglang:v0521-yihou-1010-mc4340`

On 102 it exists with ID `sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2`.
To rebuild (CPU only, ~3 min with the base pulled, needs network for the Mooncake clone):
```bash
# recreate the build context: infera.glm52.pd (git@github.com:AMD-AGI/Infera.git, branch dev/pd_opt/glm_5.2_agentx)
git clone git@github.com:AMD-AGI/Infera.git infera-src && cd infera-src && git checkout 28121cb4
git am <kit>/patches/image-recipe/000{1,2,3,4}-*.patch          # 0005 (mc4340prio) is optional, not used for 22k
#   or: git fetch <kit>/patches/image-recipe/infera-src.yihou.28121cb4..702af10f.bundle yihou.agentx44.20261009
docker build -f deploy/docker/Dockerfile.sglang.v0521-mc4340.yihou \
  --build-arg BUILD_MOONCAKE=1 --build-arg REQUIRE_MOONCAKE_CROSS_HOST_ROUTING=1 --progress=plain \
  -t infera-sglang:v0521-yihou-1010-mc4340 .
```
Base `lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008` = digest `sha256:6c2c0b0e2afed9924d6f3bd621c554e83e6ffd37a1bf559f900cfbda7eb763ef`.
Verify (both print `fail=0`):
```bash
docker run --rm --network none -v <kit>:/kit:ro --entrypoint bash infera-sglang:v0521-yihou-1010-mc4340 \
  -c 'bash /kit/scripts/verify/verify_v0521.sh; bash /kit/scripts/verify/verify_mc4340.sh'
```
A rebuilt image gets a new ID: add it to the `case` in `scripts/harness/config.yihou.mc4340.sh` (EXPECTED_IMAGE_ID).
Build logs: `logs/build/`.

## 2. Harness

`scripts/harness/` is a verbatim copy of `$WS/harness` at pack-up time. Absolute paths baked in:
`config.yihou.v0521-base.sh` (`WS=/shared_nfs/yihou_glm52/v0521dbg_20261010`, shim dir), `pipeline.yihou.sh`, the
`env.*.yihou.sh` files (`H=$WS/harness`), `a2_e2g.yihou.sh`. Easiest: run from the original `$WS/harness`. For a fresh copy:
```bash
NEWWS=/shared_nfs/<you>/v0521_repro_yihou && mkdir -p $NEWWS/{runs,cache,logs}
cp -a <kit>/scripts/harness $NEWWS/harness
sed -i "s#/shared_nfs/yihou_glm52/v0521dbg_20261010#$NEWWS#g" $NEWWS/harness/*.sh $NEWWS/harness/*.tsv
ln -s /shared_nfs/yihou_glm52/agentx44_20261009/cache/InferenceX $NEWWS/cache/InferenceX
```
Another node: add `topology.yihou.<node>.tsv` and edit `YIHOU_NODE/YIHOU_IP/YIHOU_JOB` in `env.e2g-102.yihou.sh`.

The round env `env.e2g-102.yihou.sh` resolves to (dryrun: `results/e2g/dryrun.e2g-102.yihou.txt`):
```
CONFIG = config.yihou.mc4340.sh                 image mc4340 + mounts v2-regbase (noblock kv-events, absorb-bmm off, REGBASE log)
  MC4340_BASE_CONFIG = config.yihou.E2-target.sh  prefill --hicache-size 450 + SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512,
                                                  index_share_for_mtp_iteration=false, fused DSA indexer ON (v0521 default),
                                                  decode --mem-fraction-static 0.95, triton DSA prefill+decode, EAGLE 5/1/6,
                                                  SGLANG_SIMULATE_ACC_LEN=3.61
    -> config.yihou.v0521-base.sh
YIHOU_OVERRIDES: CONC=40 ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000
```
Check offline: `source env.e2g-102.yihou.sh && bash run.v2.yihou.sh dryrun` must match the saved dryrun except paths/job.

## 3. A1 — bring-up + first (warm-up) run

From the login host in `$WS/harness` (run detached so a login-host loss cannot kill the driver):
```bash
source env.e2g-102.yihou.sh
# if another stack is up on the node: export PRE_STOP_ENV=$H/env.<old>.yihou.sh
setsid nohup bash pipeline.yihou.sh > ../runs/e2g.pipeline.log 2>&1 < /dev/null &
```
Stages (`runs/e2g.pipeline.log`, `runs/e2g/status.tsv`): `gpuidle` → `etcd` → `prefill ok` (~7 min) → `decode-a1 ok`
(~6 min; retries decode only on RCCL init failure) → `router` → `census` (hard-stops on MAF/OOM/Traceback or
overlapping rank PIDs) → `smoke` (garbled text is EXPECTED: simulated acceptance) → `audit` (`AUDIT PASS`) →
`bench` (AgentX full C40 → `runs/e2g/agentx-c40/`). End: `bench ok rc=0 AgentX passed …` then `PIPELINE DONE`.
**Do not stop the stack.**

## 4. A2 — flush + scored run (the 22k)

```bash
setsid nohup bash a2_e2g.yihou.sh >> ../runs/e2g.A2.log 2>&1 < /dev/null &
```
It POSTs `/flush_cache` to decode `:29357` then prefill `:29101` (prefill answers 400 while requests are pending →
retries every 10 s until 200), then `BENCH_TAG=A2 bash run.v2.yihou.sh bench` → `runs/e2g/agentx-c40-A2/`.
Confirm cold: the first A2 prefill batches must show `#cached-token: 0`:
`cd $WS && strings runs/e2g/server-logs/prefill-0.log | grep -a "Prefill batch" | awk '$0 > "<A2 start ISO>"' | head`.
(`chain_e2g.yihou.sh` automates A1 → A2 incl. the host sampler; optional.)

## 5. Read the result

```bash
cd $WS        # /shared_nfs/yihou_glm52/v0521dbg_20261010 (or your NEWWS)
python3 - <<'EOF'
import json; j=json.load(open('runs/e2g/agentx-c40-A2/agentx_conc40.json')); m=j['request_metrics']
print(j['num_requests_successful'], j['request_accounting']['records_error_dropped'],
      round(m['throughput']['per_gpu']['total_tput_tps']), round(m['latency']['ttft']['p95'],2), round(m['latency']['tpot']['mean']*1000,1))
EOF
for l in runs/e2g/server-logs/*.log; do echo $l $(strings $l | grep -ac "Memory access fault"); done      # want 0
python3 harness/tools/decode_stats.yihou.py --label A2 runs/e2g/server-logs runs/e2g/agentx-c40-A2/benchmark.log
```

## Expected output (`results/e2g/A2-scored/agentx_conc40.json`)

| metric | e2g A2 | replicates (same protocol) |
|---|---|---|
| verdict | `AgentX passed`, rc=0 | pass, pass |
| ok / error-dropped | 4286 / 1 | 4258 / 1 (e2i R2), 4237 / 3 (e2c C40) |
| tok/s/chip (total) | **22,114** (176,909) | 21,921, 21,696 |
| TTFT mean / p50 / p95 | 5.30 / 3.33 / 19.21 s | |
| TPOT mean / p95 | 12.1 / 17.9 ms | |
| decode ms/step/DP, running-req | 40.9, 3.57 | 41.8 / 41.9 |
| A1 (same stack, first run) | 19,640 — expected and not scored | |

If A2 lands ≈19.5-20.5k, it was effectively a first run (stack restarted in between, or the warm-up run never reached
decode load) — see `notes.md` §1.

## If it doesn't reproduce

See `notes.md` (warm-stack penalty, slow VRAM drain, login-host loss, PID-collision family, ports, InferenceX symlink)
and the phase-2 pack-up's notes for bring-up failures.
