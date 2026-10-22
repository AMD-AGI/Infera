# Reproduction kit — GLM-5.2 same-node 1P1D P4D4 on sglang v0.5.21, AgentX full C44 (round e2c)

Goal: reproduce round **e2c** (the mission target) — `AgentX passed`, CONC 44, 3600 s profiling,
0 GPU memory-access faults — on one MI355X node.
Wall time on a node that already has the image and warm JIT caches: ~1 h 50 min
(≈25 min VRAM drain after a previous HiCache-450 round, ≈6 min prefill + ≈6 min decode bring-up,
≈23 min AgentX warmup, 60 min profiling, ≈4 min result processing). Cold JIT caches add ~25 min;
an image build adds ~3 min (r2 build on 102: 06:33:23 → 06:36:06, base already pulled).

## 0. Prerequisites

- **Machine:** one 8× MI355X node in `amd-spur`. Proven node: `crsuse2-m2m-102` (ens3 10.245.147.130).
  Allocation used: account `amd-oai`, QOS `amd-oai-qos`, 24 h, holder `scripts/hold_inf.yihou.sh`:
  ```bash
  sbatch -p amd-spur -A amd-oai -q amd-oai-qos -w crsuse2-m2m-102 -N1 -t 1-00:00:00 \
         --job-name yihou-oai-go2 scripts/hold_inf.yihou.sh
  ```
  (`-N1` gave all 8 GPUs here. On burst QOS `-N1` alone once gave cpu=1 and 0 GPUs; there
  `--gres=gpu:mi355x:8 -c 160` was needed. Burst jobs were preempted after ~20 min — do not use them for 3600 s benches.)
- **Node access:** sshd on spur nodes rejects normal users. The harness reaches the node through
  `scripts/shimbin/ssh` (spur exec shim). It needs `SPUR_SSH_JOBS=<node>:<jobid>`. The harness
  sets this itself when `YIHOU_JOB` is non-empty.
- **Docker policy on spur nodes:** `spur-authz` denies `--pid=host`. Do not use it. The harness
  refuses `ENGINE_PID_HOST=1` when `YIHOU_JOB` is set.
- **Secrets:** none are required beyond your cluster login. The base image and the InferenceX repo
  are public, the HF dataset `semianalysisai/cc-traces-weka-062126` is public, and the model is on shared NFS.
- **External paths (absolute, not in this kit):**
  - model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (256 experts, unmodified);
  - original workspace `WS=/shared_nfs/yihou_glm52/v0521dbg_20261010` (harness, runs, caches; still on disk);
  - build context: clone `/shared_nfs/yihou_glm52/agentx44_20261009/infera-src.yihou` @ `68f06e29`.
- **Node prerequisites:** before launching, all 8 GPUs must be idle (< 400 MiB each, two checks 60 s
  apart, which the harness enforces). Fixed ports must be < 30000 because the ephemeral range is
  32768-60999; the harness already does this.

## 1. Image `infera-sglang:v0521-yihou-1010-mc4340`

If the image already exists on the node, check its ID. On 102 it should be
`sha256:9d755dd6a7bf2f3705a45dcc7877b9ec3625efb8370ab3b144751799e31300d2`; a rebuild elsewhere gets a
new ID, which you set in `config.yihou.mc4340.sh`, as described in §2.

To build it, use the exact command that produced the 102 image, run from the login host:
```bash
export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH   # or ./scripts/shimbin
export SPUR_SSH_JOBS=crsuse2-m2m-102:<jobid>
ssh crsuse2-m2m-102 "cd /shared_nfs/yihou_glm52/agentx44_20261009/infera-src.yihou && \
  setsid nohup bash -c 'date -u; docker build -f deploy/docker/Dockerfile.sglang.v0521-mc4340.yihou \
    --build-arg BUILD_MOONCAKE=1 --build-arg REQUIRE_MOONCAKE_CROSS_HOST_ROUTING=1 --progress=plain \
    -t infera-sglang:v0521-yihou-1010-mc4340 . ; echo BUILD_EXIT=\$?; date -u' \
  > /mnt/m2m_nobackup/yihou_v0521dbg/build/build.mc4340.r2.yihou.log 2>&1 < /dev/null &"
```
- The build context is the clone at `68f06e29` with a clean tree. To rebuild from scratch, apply
  `patches/image-recipe/01..04` in order on `infera.glm52.pd@28121cb4`.
- The defaults used are `SGLANG_BASE_IMAGE=lmsysorg/sglang-rocm:v0.5.21-rocm720-mi35x-20261008` (digest
  `sha256:6c2c0b0e…63ef`) and `MOONCAKE_GIT_REF=faae8dd4a6309c3ecd47e0721a83b0250d686fa2`. The build
  needs network access for the Mooncake clone.
- Expected log lines: `[mc-build] applied … sha256=6d73b2f55a0962ad` and `#4340 HIP IPC offset path present`.
  The 102 log is in `logs/build/build.mc4340.r2.102.yihou.log.gz`.
- Verify the image with the following. Both scripts should print `fail=0`:
  ```bash
  docker run --rm --network none -v /shared_nfs/yihou_glm52:/shared_nfs/yihou_glm52:ro --entrypoint bash \
    infera-sglang:v0521-yihou-1010-mc4340 -c 'bash <kit>/scripts/verify_v0521.sh; bash <kit>/scripts/verify_mc4340.sh'
  ```
  The loaded engine must hash to `mooncake/engine…so` sha256 `3c96489a…b289a`.

## 2. Harness

The harness is `scripts/harness/` (verbatim). It has absolute paths baked in:
- `config.yihou.v0521-base.sh:24` sets `WS=/shared_nfs/yihou_glm52/v0521dbg_20261010`;
- `:38` and `pipeline.yihou.sh:16` reference the shim dir `/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin`;
- the `env.*.yihou.sh` files set `H=$WS/harness`.

The original WS is still on disk, so the simplest path is to run from it. For a fresh copy:
```bash
NEWWS=/shared_nfs/<you>/v0521_repro_yihou && mkdir -p $NEWWS/{runs,cache,logs}
cp -a <kit>/scripts/harness $NEWWS/harness
sed -i "s#/shared_nfs/yihou_glm52/v0521dbg_20261010#$NEWWS#" $NEWWS/harness/*.sh $NEWWS/harness/*.tsv
ln -s /shared_nfs/yihou_glm52/agentx44_20261009/cache/InferenceX $NEWWS/cache/InferenceX   # or let agentx_bench clone 918524ff
```
- **Another node:** add `topology.yihou.<node>.tsv` (role/node/data_ip) and add your image ID to
  the `case` in `config.yihou.mc4340.sh`.
- **Another job:** set `YIHOU_JOB` in the env file.

The env file for the target round is `scripts/harness/env.e2c-102.yihou.sh`. Its header still
says "prepared, NOT launched" from when it was written; this is the file that ran. It resolves to:
```
CONFIG=config.yihou.mc4340.sh  (image mc4340 + REGBASE mount)
  MC4340_BASE_CONFIG=config.yihou.E2-target.sh  (hicache-size 450, CHUNK_GB=512, index_share false,
                                                  fused indexer default=ON, decode mem 0.95, CONC 44)
    -> config.yihou.v0521-base.sh
YIHOU_OVERRIDES: ENGINE_PID_HOST=0 IPC_MODE_LEGACY=0 DEBUG_AGENT=0 DECODE_PID_OFFSET=30000
```
Check the rendering offline: `source env.e2c-102.yihou.sh && bash run.v2.yihou.sh dryrun`. The
output must match `results/e2c/dryrun.e2c-102.yihou.txt`, apart from paths, node, and job.

## 3. Run the round, end to end

From the login host, inside `$WS/harness`:
```bash
source env.e2c-102.yihou.sh
# if a previous round is still up on the node:  PRE_STOP_ENV=$H/env.<old>.yihou.sh
nohup bash pipeline.yihou.sh > ../runs/e2c.pipeline.log 2>&1 &
```
Stages are logged to `runs/e2c.pipeline.log` and `runs/e2c/status.tsv`:
1. `gpuidle`: all GPUs < 400 MiB twice, 60 s apart. After a HiCache-450 round this takes 10-30 min (see notes).
2. `etcd`, then `prefill ok`, then `decode-a1 ok`. Decode retries up to 3× on an RCCL init failure, without
   restarting prefill. In the PID-offset era it came up on the first try in all 3 passes.
3. `router`, then `census`. The pipeline hard-stops (`pipeline.yihou.sh:46-50`) on either of these:
   - any Memory access fault, `kernel_code_entry`, OOM, or non-health-check Traceback;
   - prefill and decode scheduler rank PIDs that overlap, or that cannot be found.

   Check these census lines by eye, because they are not gated:
   - 4× `Found 0 HCAs` and HIP installed ×4 per leg;
   - `fused idx ON` = 4 on both legs for e2c;
   - `REGBASE interior=0`;
   - `host pool … 450.00 GB` ×4.
4. `smoke`: a garbled completion is EXPECTED, because simulated acceptance is on. It still proves the
   KV moved prefill → decode.
5. `audit`: the live server_args and env are checked against the config (`AUDIT PASS`).
6. `bench`: AgentX full, CONC 44, 3600 s, warmup 10/lane, output in `runs/e2c/agentx-c44/`.

The result appears as the final pipeline line:
`bench ok rc=0 AgentX passed: …/runs/e2c/agentx-c44/agentx_conc44.json`, then `PIPELINE DONE`.

Manual equivalents: `bash run.v2.yihou.sh {up|census|smoke|audit|bench|stop}` with the same env.
Stop with `bash run.v2.yihou.sh stop` (stop only, never rm).

## 4. Read the result

```bash
python3 - <<'EOF'
import json; d=json.load(open('runs/e2c/agentx-c44/agentx_conc44.json')); m=d['request_metrics']
print(d['num_requests_successful'], d['request_accounting']['records_error_dropped'],
      round(m['throughput']['total']['tokens_per_second']), m['latency']['ttft']['p50'], m['latency']['itl']['p50'])
EOF
for l in runs/e2c/server-logs/*.log; do echo $l $(strings $l | grep -c "Memory access fault"); done   # want 0
```

## Expected output (the e2c numbers, `results/e2c/agentx_conc44.json`)

| metric | e2c |
|---|---|
| verdict | `AgentX passed`, rc=0 |
| profiled ok / error-dropped | 4179 / 2 (InvalidInferenceResultError) |
| total / input / output tok/s | 164,194 / 163,011 / 1,182 (20,524 total per GPU) |
| TTFT p50 / p90 / p95 | 4.63 / 19.95 / 28.65 s |
| ITL p50 / p95 | 14.7 / 21.6 ms |
| server logs, both legs | Memory access fault 0, Traceback 0, KV send/get failures 0, `Error code: 201` 0 |
| host pool | cpu_kv_usage reaches ~100 % (crosses the H1 boundary at ~61 %) without a fault |

Run-to-run variance has not been measured, since there is one run per config. See `notes.md`.

## Relaxed baselines (same chain)

- **e1c**: `env.e1c-102.yihou.sh`. E1 relaxed, CONC 40, hicache ratio 1.5, decode mem 0.85, fused indexer OFF.
- **e2b4**: `env.e2b4-102.yihou.sh`. Target minus the fused indexer.

## If it doesn't reproduce

See `notes.md`, in particular:
- the PID-collision family (201 / `hipIpcGetMemHandle invalid argument`);
- `--pid host` being denied on spur nodes;
- the ephemeral-port collision;
- the slow VRAM drain;
- the InferenceX symlink;
- another user's job looping on a shared node.
