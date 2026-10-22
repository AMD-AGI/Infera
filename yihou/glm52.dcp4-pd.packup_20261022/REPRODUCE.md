# REPRODUCE — GLM-5.2-MXFP4 + DCP on MI355X (PD via Infera, mixed, decode microbenchmarks)

Ordered, copy-pasteable steps for:

- (a) the best PD run, **pdBeperf-r4 A2** (18,321 tok/s/chip), and the current main line, **pdBeperf-r5b** (score pending);
- (b) the mixed DCP4 c40 run, **full1-dcp4-c40** (26,277 tok/s/GPU on 4 GPUs);
- (c) the decode microbenchmarks (fixed-batch waves on cached long prompts).

Path conventions:

- `KIT` = this directory, `/home/yihou/dev/git/infera.glm52.pd/yihou/glm52.dcp4-pd.packup_20261022`.
- **EXTERNAL** = an absolute path outside the pack-up. These are cluster paths on `/shared_nfs` (mounted on the spur login
  node and on the compute nodes) and are listed in `environment.md` "External dependencies":
  - `CL=/shared_nfs/yihou_glm52/dcp_pd_20261021` (EXTERNAL): the cluster workspace.
  - `PD=$CL/pd`, `MIX=$CL/mixrun`, `B=$CL/bench` (EXTERNAL).
  - Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` (EXTERNAL, read-only).
  - `WS=/home/yihou/dev/git.16-19/sglang.pd.pp.dcp/yihou_ws/glm52_dcp_pd_20261021` (EXTERNAL, local workstation, not on the
    cluster): ledgers (`WS/ledger.*.md`), `WS/poll.md`, `WS/RESULTS.md`, cited as evidence only; no step needs it.
  - InferenceX checkout `$CL/cache/InferenceX` (PD) and `/shared_nfs/yihou_glm52/agentx44_20261009/cache/InferenceX`
    (mixed `agentx.sh`), commit `918524ff94045b3f091115f1051c22a8588edf2b` (EXTERNAL).

**The scripts hard-code `CL`** (e.g. `PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd` in `pd_seq.yihou.sh`,
`launch_pd.yihou.sh`, `mk_env.yihou.sh`, `run.v2.yihou.sh` via `$WS`; `M=.../mixrun` in `scripts/mixed/*.sh`). To
reproduce elsewhere, copy the scripts to the same relative layout under a new `yihou` directory and change those
variables. The mapping is (from `scripts/README.md`):

| Pack-up path | Must live at (as run) |
|---|---|
| `scripts/harness/*` (incl. `bench-harness/`, `tools/`, `mounts-v0522/`) | `$PD/harness/` |
| `scripts/harness/pd-tools/*` | `$PD/tools/` |
| `scripts/mixed/*` | `$MIX/` (and `$B/` for the bench copies of `serve.sh`, `decodebench.py`, `check.py`) |
| `scripts/bench/*` | `$B/` (plus `scripts/mixed/decodebench.py` copied to `$B/decodebench.py`) |

Rules used throughout this task (`spec/mission.md`): containers named `yihou-dcp-*`; stop with `docker stop`, never
`docker rm`; never delete outside a `yihou` path. Scores with `SGLANG_SIMULATE_ACC_LEN` set are **performance only**;
correctness comes only from the sim-off gates below.

---

## 0. Prerequisites (all runs)

1. **Spur access** (the only secret; `environment.md` "Required secrets"): ssh to the spur login node
   `crs-spur.crusoe.amd.com` with your own key. Commands on a compute node run as `spur exec <JOB> bash -c "..."`.
   `spur exec` prints spurious "transport error" lines and can drop a long-attached driver; clients keep running
   (`notes.md` §4).
2. **Hold one 8 x MI355X node** (spur job). Nodes used: `crsuse2-m2m-054` (job 228279, amdgpu 6.14.14) for PD r1-r4;
   `crsuse2-m2m-v3-045` (229517) for mixed; `crsuse2-m2m-v3-060` (229524) for PD r5/r5b and benchrun; `crsuse2-m2m-v3-070`
   (230266) for E2 (`environment.md` "Hardware"). **v3 (054 vs v3) results are not comparable** (different driver,
   `HSA_ENABLE_SDMA=0` required on v3).
3. **Node prep on v3 nodes (amdgpu 6.16.13):** every engine container needs `HSA_ENABLE_SDMA=0`, otherwise weight load
   hangs in HIP `memcpy_and_sync` (`notes.md` §1, `WS/ledger.mixrun.md` R5-R8). On v3 also use `DEBUG_AGENT=0`
   (the ROCm debug agent aborted a prefill scheduler in r5, `notes.md` §6). 054 (amdgpu 6.14.14) ran with SDMA on
   and `DEBUG_AGENT=1`.
4. **sglang source tree** (overlay, mounted read-only over `/sgl-workspace/sglang/python/sglang`). Rebuild the branch
   from the patches (`patches/README.md`):

   ```bash
   # in an sglang clone (origin = sgl-project/sglang)
   git checkout --detach fda84f80b7
   git apply --index $KIT/patches/sglang-branch/0000-merge-pr44322-onto-fda84f80b7.diff && git commit -m "merge PR #44322"
   git am $KIT/patches/sglang-branch/00*.patch          # tree == 93034eba91 (checked 2026-10-22)
   ```

   Then export the python package to `$CL/src/sglang-<sha>/python/sglang` (EXTERNAL). The runs used
   "`git archive python/` of the branch" (`environment.md` "sglang source actually run"); the exact command line was
   not captured. An equivalent:

   ```bash
   SHA=ae529c96bf      # r4 + mixed full1-dcp4-c40;  93034eba91 for r5/r5b/E2
   mkdir -p $CL/src/sglang-$SHA && git archive $SHA python | tar -x -C $CL/src/sglang-$SHA
   ```

   `ae529c96bf` = patches 0000 + 0001..0012 applied (`patches/README.md` commit table). Always `cd /tmp` before
   `import sglang` inside a container (`environment.md`).

---

## (a) PD: best run pdBeperf-r4 A2 (and r5b, the current main line)

Layout: same-node 1P1D P4D4. Prefill GPUs 0-3 = the 22k e2g leg (TP4 + DPA4, no DCP, HiCache 450). Decode GPUs 4-7 =
TP4/EP4 DP1 + `--dcp-size 4`, EAGLE 5/1/6, CUDA graph, custom AR on. KV transfer Mooncake on the same node. The full
knob table is in `README.md`; the exact rendered commands are `scripts/harness/dryrun.pdBeperf-r4-054.yihou.txt` and
`scripts/harness/dryrun.pdBeperf-r5b-060.yihou.txt`.

### a.1 Image `infera-sglang:v0522-1020-yihou-dcp-mc4340`

Expected ID `sha256:5b6565ef3e4e7dcca02423d509f842a74cf7416b18ed94edbf12e2be100f7376` (`environment.md` "Docker images";
`IMAGE_IDS` in `results/pd/pdBeperf-r4/agentx-c40-A2/runtime.env`).

Option 1 (what r5/r5b/E2 did, `WS/ledger.pdrun.md` P016): `launch_pd.yihou.sh` loads it automatically if missing, from
`$PD/images/infera-sglang_v0522-1020-yihou-dcp-mc4340.tar` (EXTERNAL, 23 GB). Manually:

```bash
spur exec $JOB bash -c "cd /tmp && docker load -i $PD/images/infera-sglang_v0522-1020-yihou-dcp-mc4340.tar && \
  docker image inspect --format '{{.Id}}' infera-sglang:v0522-1020-yihou-dcp-mc4340"
```

Option 2 (rebuild, what 054 did, `WS/ledger.pdrun.md` P003): needs the Infera clone `$PD/build/infera-src.yihou`
(EXTERNAL, branch `yihou.dcp.v0522.20261021` @ `80458804`) and network:

```bash
spur exec $JOB bash -c "bash $PD/tools/build_image_on_node.yihou.sh"   # = scripts/harness/pd-tools/build_image_on_node.yihou.sh
```

It pulls `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020@sha256:3a75c913...a464`, builds with
`Dockerfile.sglang.v0522-1020-mc4340.yihou`, runs `pd-tools/verify/verify_v0521.sh` + `verify_mc4340.sh`, and appends
`node<TAB>image<TAB>id` to `$PD/harness/image_ids.dcp.tsv` (read by `config.yihou.dcp.sh`; a node without a row only
gets a warning, and `run.v2 up` refuses it).

### a.2 Node topology file

`scripts/harness/topology.yihou.<NK>.tsv` (role, node, data IP). Existing: `054`, `060`, `070`, `v3-070`, `268`. For a new
node, copy one and change host/IP (both rows point at the same node).

### a.3 Correctness gate (sim off) — `pd_seq.yihou.sh` phase G

`pd_seq.yihou.sh` first launches variant **Be** (same layout, `DECODE_SIMULATE_ACC_LEN=` empty = real draft) as round
`pdBe-<TAG>g`, runs T4a (`t4.yihou.sh a` -> `pd-tools/t4/check.py`: math, passkey 8K / 34K / 96K) and stops unless
`math_nothink`, `passkey_8k`, `passkey_34k` are `PASS`. Then it stops the gate stack.

Recorded gates:
- r4 code (ae529c96bf): `pdBe-r1` on 054, **eager**, sim off: math 29 x2, passkey 8K/34K/96K PASS
  (`results/pd/pdBe-r1/t4/`, `WS/ledger.pdrun.md` P010). No graph-mode sim-off gate on ae529c96bf is recorded.
- r5/r5b code (93034eba91): `pdBe-r5g` on v3-060, **graph**, sim off: math 29, passkey 8K/34K/96K PASS
  (`WS/ledger.pdrun.md` P016).
- **Pending:** a 40-concurrent sim-off check on the DCP4 PD decode in graph mode (only 5-request checks exist,
  `results/RESULT.md` §6). To run it against a live sim-off stack (router port 28100):

  ```bash
  # on the node; scripts/mixed/run_check40_pd.sh (writes $MIX/results/check{,40}.<TAG>.*)
  bash $MIX/run_check40_pd.sh dcp4pd-graph http://<IP>:28100 glm5.2-mxfp4
  ```

### a.4 Perf round + AgentX protocol (A1 fresh -> flush -> A2 scored)

Protocol (same as the 22k kit; `results/RESULT.md` §1): AgentX C40, 3600 s profiling, warm-up 10 requests/lane
(444 total), scenario `inferencex-agentx-mvp`, dataset `semianalysis_cc_traces_weka_062126`, `--random-seed 42`
(`results/pd/pdBeperf-r4/agentx-c40-A2/benchmark_command.txt`).

1. Bring up variant **Beperf** (decode `SGLANG_SIMULATE_ACC_LEN=3.61`, `match-expected`, `real-draft-token`).
2. **A1**: AgentX C40 on the fresh stack (not the score).
3. `POST /flush_cache` on decode (:29357) then prefill (:29101); both must return 200.
4. **A2**: AgentX C40 again on the same stack -> **scored**.

#### r4 (054, ae529c96bf, SDMA on, DEBUG_AGENT=1) — as run

r4 was driven by `scripts/harness/pd_r4.yihou.sh` (stop r2, launch r4, A1), whose overrides are:

```bash
# login node, inside $PD/harness ; JOB = spur job holding the node
export PD_JOB=$JOB PD_CONC=40
export PD_EXTRA="PREFILL_HICACHE_SIZE=450 DECODE_CUSTOM_AR=1 DECODE_EXTRA_ENV_BASE=MC_LOG_LEVEL=INFO"
# PD_OVERLAY unset -> mk_env default $CL/src/sglang-ae529c96bf/python/sglang
setsid nohup bash $PD/harness/launch_pd.yihou.sh 054 $JOB Beperf r4 > $PD/runs/launch.pdBeperf-r4.log 2>&1 < /dev/null &
```

`launch_pd.yihou.sh` = image check/load -> `pd-tools/mk_env.yihou.sh 054 pdBeperf-r4 Beperf` -> `run.v2.yihou.sh dryrun`
(-> `dryrun.pdBeperf-r4-054.yihou.txt`) -> `up` -> `census` / `audit` / `smoke` -> `pdcheck.yihou.sh`. Outputs:
`$PD/runs/pdBeperf-r4/{status.tsv,census.txt,audit.txt,smoke.txt,pdcheck.txt}` (copies in `results/pd/pdBeperf-r4/`).

Check before benching:
- `audit.txt` has exactly one FAIL, `decode disable_custom_all_reduce=False (want True)`: the audit expectation is
  stale; `False` is the intended custom-AR-on state (r5g audit shows the same, `WS/ledger.pdrun.md` P016).
- Diff your rendered `dryrun` against `scripts/harness/dryrun.pdBeperf-r4-054.yihou.txt` (only node/IP/cache paths may differ).

Then (as run; the A1 driver was lost to a spur exec error, so the fresh run that counts is A1b, `working_process.md` Phase 5):

```bash
cd $PD/harness
# same shell as above: PD_JOB and PD_EXTRA must still be exported (env.<round>-<NK>.yihou.sh reads PD_JOB, PD_EXTRA
# and PD_OVERLAY when it is sourced; it aborts if PD_JOB is unset)
bash bench_retry.yihou.sh 054 pdBeperf-r4 A1b      # fresh run   -> runs/pdBeperf-r4/agentx-c40-A1b/   (17,473/chip)
bash flush_bench.yihou.sh pdBeperf-r4 A2           # flush D+P, then bench -> runs/pdBeperf-r4/agentx-c40-A2/ (18,321/chip)
```

`bench_retry.yihou.sh` and `flush_bench.yihou.sh` hard-code `JOB=228279` and the 054 IP `10.245.144.125`; edit them for
another node. On a new node the one-shot equivalent is `pd_seq.yihou.sh` below with `SKIP_GATE=1` and the same `PD_EXTRA`.

#### r5b (v3-060, 93034eba91, SDMA=0, DEBUG_AGENT=0, kv_b bf16 + split triton indexer) — current main line

Driver `pd_seq.yihou.sh` (gate already passed as `pdBe-r5g` on the same code -> `SKIP_GATE=1`). The exact `PD_EXTRA`
string of r5b was not captured; the one below is reconstructed from `WS/ledger.pdrun.md` P016 ("r5 = ... BOTH_ENV_LIST=
HSA_ENABLE_SDMA=0 ... DECODE_ENV_LIST=..."; "r5b = r5 + DEBUG_AGENT=0") and must render the recorded dryrun:

```bash
export PD_OVERLAY=$CL/src/sglang-93034eba91/python/sglang
export PD_EXTRA="PREFILL_HICACHE_SIZE=450 DECODE_CUSTOM_AR=1 DECODE_EXTRA_ENV_BASE=MC_LOG_LEVEL=INFO DEBUG_AGENT=0 \
BOTH_ENV_LIST=HSA_ENABLE_SDMA=0 \
DECODE_ENV_LIST=SGLANG_DSA_DCP_KEEP_KV_B_BF16=1,SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1,SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1,SGLANG_LOG_DECODE_SEQ_LENS=1"
cd $PD && SKIP_GATE=1 setsid nohup bash $PD/harness/pd_seq.yihou.sh 060 $JOB r5b 40 > runs/pd_seq.r5b.log 2>&1 < /dev/null &
# pd_seq: launch Beperf round pdBeperf-r5b -> bench A1 -> flush (29357, 29101) -> bench A2 (stack left up)
```

As run, the r5b driver also stopped the previous r5 stack first (first line of `$PD/runs/pd_seq.r5b.log`,
EXTERNAL: `pdBeperf-r5 stopped`), i.e. `OLD_ROUND=pdBeperf-r5` was set; on a clean node leave `OLD_ROUND` unset.

Verify the reconstruction before trusting a score: the decode env of your `dryrun.pdBeperf-r5b-<NK>.yihou.txt` must add,
relative to r4, exactly `HSA_ENABLE_SDMA=0`, `SGLANG_DSA_DCP_KEEP_KV_B_BF16=1`, `SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1`,
`SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1`, `SGLANG_LOG_DECODE_SEQ_LENS=1`, mount `sglang-93034eba91`, and drop
`HSA_TOOLS_LIB` / `HSA_ENABLE_DEBUG` / `ROCM_DEBUG_AGENT_OPTIONS` (that is the r4 -> r5b difference in the recorded
dryruns `scripts/harness/dryrun.pdBeperf-{r4-054,r5b-060}.yihou.txt`; prefill differs only by `HSA_ENABLE_SDMA=0`, the
mount and the debug-agent env). `results/pd/pdBeperf-r5b/audit.txt` shows the same env as PASS lines.
If the driver dies while A1 runs, continue with `bash $PD/harness/pd_cont.yihou.sh 060 $JOB pdBeperf-r5b 40`
(wait A1 client -> flush -> A2; `WS/ledger.pdrun.md` 02:08Z).

r5b A1/A2 scores: **pending** at pack-up time (`results/RESULT.md` §2).

### a.5 Score and decode statistics

The score is `request_metrics.throughput.per_gpu.total_tput_tps` of the AgentX JSON = (input + output tok/s) /
(`num_prefill_gpu` 4 + `num_decode_gpu` 4) (`results/RESULT.md` §1):

```bash
python3 -I -c "import json,sys; j=json.load(open(sys.argv[1])); t=j['request_metrics']['throughput']['per_gpu']; \
print(round(t['total_tput_tps']), j['num_prefill_gpu'], j['num_decode_gpu'])" \
  $KIT/results/pd/pdBeperf-r4/agentx-c40-A2/agentx_conc40.json      # -> 18321 4 4
```

Note: the JSON field `decode_dcp_size` reads 1 even for DCP4; the DCP setting is in `audit.txt` / `census.txt` / dryrun.

Decode step time from the server logs (EXTERNAL, `$PD/runs/<round>/server-logs/`, not copied):

```bash
python3 -I $KIT/scripts/harness/pd-tools/decode_stats.dcp.yihou.py --label r4A2 \
  --bench $PD/runs/pdBeperf-r4/agentx-c40-A2.log \
  $PD/runs/pdBeperf-r4/server-logs/decode-0*.log --prefill $PD/runs/pdBeperf-r4/server-logs/prefill-0*.log
```

(`--bench` takes the AgentX benchmark log that bounds the profiling window, or use `--start/--end`. The recorded r4 A2
line in `results/pd/pdBeperf-r4/decode_stats.r4.txt` used the window `10-21 21:45:19 + 3640 s`, i.e.
`--start "2026-10-21 21:45:19" --end "2026-10-21 22:45:59"`; the exact command line was not recorded.)

### a.6 Stop

```bash
# needs the same PD_JOB / PD_EXTRA (/ PD_OVERLAY for r5b) exports as the launch
cd $PD/harness && source env.pdBeperf-r4-054.yihou.sh && bash run.v2.yihou.sh stop     # docker stop only
```

---

## (b) Mixed DCP4 c40: full1-dcp4-c40 (26,277 tok/s/GPU, one 4-GPU server)

As run on `crsuse2-m2m-v3-045` (amdgpu 6.16.13 -> `HSA_ENABLE_SDMA=0`), image
`lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020` used directly, sglang overlay `ae529c96bf`
(`WS/ledger.mixrun.md` R4-R9 and "Full AgentX c40"). A no-DCP reference server (`full1-ref-c40`) ran at the same time on
GPUs 4-7 (NUMA-split, `CPUSET`); start it too if you want the same node load.

```bash
# on the node (spur exec $JOB bash -c "..."), scripts copied to $MIX
docker pull lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020          # digest sha256:3a75c913...a464 (environment.md)

# 1) correctness first: same server, sim OFF (SIM_ACC ''), graph on -> check.py (math 29 x2, passkey 8K/34K/96K)
SHA=ae529c96bf NODE=v3-045 CPUSET=0-117 PID_OFFSET=30000 EXTRA_ENV=HSA_ENABLE_SDMA=0 \
  bash $MIX/serve.sh yihou-dcp-mix-dcp4-graph 0,1,2,3 30000 4 1 '' 80 64
bash $MIX/run_check.sh dcp4-graph-v3045 30000          # -> $MIX/results/check.dcp4-graph-v3045.txt : all PASS
docker stop yihou-dcp-mix-dcp4-graph

# 2) perf server: sim 3.61, graph, max-running 80, graph bs 64
SHA=ae529c96bf NODE=v3-045 CPUSET=0-117 PID_OFFSET=30000 EXTRA_ENV=HSA_ENABLE_SDMA=0 \
  bash $MIX/serve.sh yihou-dcp-mix-dcp4-sim 0,1,2,3 30000 4 1 3.61 80 64
#    wait for "The server is fired up and ready to roll" in $MIX/logs/mix-dcp4-sim.<node>.<ts>.log

# 3) smoke (300 s, warm-up 1/lane, --unsafe-override: NOT a valid score)
NODE=v3-045 WARMUP=1 bash $MIX/agentx.sh smoke-dcp4-c40 30000 40 300 4

# 4) flush, then the full run (3600 s, warm-up 10/lane) -> $MIX/agentx/full1-dcp4-c40/agentx_conc40.json
curl -s -X POST http://127.0.0.1:30000/flush_cache
NODE=v3-045 bash $MIX/agentx.sh full1-dcp4-c40 30000 40 3600 4
```

The rendered server env/args of the recorded run are the first three lines of
`$MIX/logs/mix-dcp4-sim.v3-045.20261021T142655Z.log` (EXTERNAL; read 10-22): env adds `SGLANG_OPT_USE_TOPK_V2=true`,
`HSA_ENABLE_SDMA=0`, sim 3.61 / `match-expected` / `real-draft-token`; args `--tp 4 --ep 4 --kv-cache-dtype fp8_e4m3
--dsa-prefill-backend triton --dsa-decode-backend triton --speculative-algorithm EAGLE 5/1/6 --chunked-prefill-size 32768
--mem-fraction-static 0.85 --max-running-requests 80 --cuda-graph-max-bs-decode 64 --dcp-size 4` (no HiCache, custom AR
on by default, index_share from the checkpoint config = true per `working_process.md` Phase 5).

Score: `per_gpu.total_tput_tps` of the JSON = total / 4 GPUs (InferenceX non-multinode, `IS_MULTINODE=false`;
`scripts/mixed/agentx.sh` header). **Not comparable to the /8 PD numbers**; compare totals (105,108 vs 146,569 tok/s,
`results/RESULT.md` §0).

---

## (c) Decode microbenchmarks (mixed server, cached long prompts, waves of N)

Method (`WS/ledger.benchrun.md` "Direct decode microbench", T10, T11, T15, T16; `results/RESULT.md` §5):

- Two sim-3.61 graph servers, one per GPU half (A arm GPUs 0-3 `CPUSET=0-117 PID_OFFSET=30000`, B arm GPUs 4-7
  `CPUSET=118-235`), `HSA_ENABLE_SDMA=0`, max-running 80, graph bs 64; then the **pair is repeated with halves swapped**.
- `decodebench.py BASE_URL OUT_JSON [PROMPT_TOKENS] [OUT_TOKENS] [WAVES]` primes 32 distinct prompts (max_tokens 1), then
  for each N runs the first N prompts concurrently with `ignore_eos` (`scripts/mixed/decodebench.py` docstring).
  Context points: 34K = `30000 2000 8,16,24,32`; 150K = `132000 1500 8,16,24`; ~200K = `175000 1200 16,24,32`.
- Metric: median `step_ms = 1000 * running * accept / gen_tput` over TP0 decode lines with `running == N` inside each
  wave window (13-14 lines per cell), computed by `wavesteps.py LABEL LOG DECODEBENCH_JSON` (`scripts/bench/wavesteps.py`;
  runs at `$B/wavesteps.py`).

Pair drivers `scripts/bench/runpair.sh` (34K + 150K) and `scripts/bench/runpair200.sh` (~200K); they must live at
`$B/` together with `scripts/bench/serve.sh` and `$B/decodebench.py` (= `scripts/mixed/decodebench.py`). Each takes
`TAG0 ENV0 TAG1 ENV1` (ENV = space-separated `KEY=VALUE` for that arm) and the sha in `$SHA`:

```bash
# e.g. T15/T16 style, radix-merge split indexer vs reference, both with kv_b bf16 (sha 93034eba91)
SHA=93034eba91 bash $B/runpair.sh rdx "SGLANG_DSA_DCP_KEEP_KV_B_BF16=1 SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1 SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1" \
                                   ref "SGLANG_DSA_DCP_KEEP_KV_B_BF16=1"
# stop both servers (docker stop yihou-dcp-bench-<tag>-sim), then swap halves:
SHA=93034eba91 bash $B/runpair.sh ref2 "SGLANG_DSA_DCP_KEEP_KV_B_BF16=1" \
                                   rdx2 "SGLANG_DSA_DCP_KEEP_KV_B_BF16=1 SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1 SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1"
python3 -I $B/wavesteps.py rdx  $B/logs/bench-rdx-sim.*.log  $B/results/decodebench.rdx-34k.json
```

(The tag names above are illustrative; the recorded tags per A/B are in `WS/ledger.benchrun.md` and the file names under
`results/decodebench/`.)

With only the kit's copies (`scripts/mixed/`), the same measurement is: start two servers with `serve.sh` as in (b)
step 2 (sim 3.61, `EXTRA_ENV="<arm env> HSA_ENABLE_SDMA=0"`), then
`bash $MIX/run_decodebench.sh <TAG> <PORT> 30000 2000 8,16,24,32` per server (writes
`$MIX/results/decodebench.<TAG>.{json,txt}`). The custom-AR ON/OFF A/B was run this way on v3-045
(`WS/ledger.mixrun.md` "Custom-AR decode microbench"; AR OFF = `--disable-custom-all-reduce` as serve.sh extra arg).
DP2 x DCP2 variant: `run_decodebench_dp.sh` / `decodebench_dp.py`.

Before using any arm for perf, gate it sim off: `check.py` all PASS and 40-concurrent `check40.py` 40/40
(`bash $MIX/run_check40.sh <TAG> <PORT>`), as done for every arm in T10-T16 (`WS/ledger.benchrun.md`).

---

## Where each recorded number came from

| Number | File |
|---|---|
| PD r4 A2 18,321/chip | `results/pd/pdBeperf-r4/agentx-c40-A2/agentx_conc40.json` |
| PD r4 A1b 17,473/chip | `results/pd/pdBeperf-r4/agentx-c40-A1b/agentx_conc40.json` |
| PD r4 decode 61.2 ms/step @ 18.75 running | `results/pd/pdBeperf-r4/decode_stats.r4.txt` |
| Mixed DCP4 c40 26,277/GPU | `results/mixed/full1-dcp4-c40/agentx_conc40.json` |
| Microbench deltas | `results/RESULT.md` §5 (sources per row) |
