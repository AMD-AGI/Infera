# Independent review of the DSv4.1 vLLM launcher + AgentX kit (2026-11-19, dev host a16-19, static only)

Scope: `scripts/kit/*`, `configs/*.env`, `configs/renders/*`. Nothing ran on a GPU node (no docker, GPU, sbatch, ssh).
Review artefacts (diff tool, CI records, extracted entry scripts, pre-review config copies): `debug/review/`.
Tag legend: [FH] = checked first-hand in this review; [UNVERIFIED] = needs a node.

## 1. Rendered argv + env vs the CI launch records (token diff)

Tool: `debug/review/tokdiff.py <render> <reproduce.json>` (shlex-tokenises the rendered `env ... vllm serve ...`, router and
UMBP lines and compares them with `.logs.launch.steps[]`; JSON-valued flags compared as parsed JSON; flag order checked too).

CI records used:
- r2-c384: `notes/ix/reproduce_448242.json` (already present).
- r2-c256: `notes/ix/reproduce_448241.json` and r1: `notes/ix/reproduce_448240.json` -- **downloaded in this review** from
  `https://inferencex.semianalysis.com/api/v1/reproduce?id=...` (the builder had inferred these two from 448242).
- r4: the real CI launch record `/perf_apps/chaox/DeepSeek_V4.1_Flash/conc32/official_ci/server_logs/x/logs/sweep_48308.log`
  lines 119-120 (`Env:` + `Command:`), cross-checked with the engine "non-default args" line of `mia1-p01-g16_agg_w0.out`.

| Config | Result after review [FH] | Remaining differences (all intentional, documented in the config header) |
|---|---|---|
| r2-dp2-umbp-c384 vs 448242 | env, vLLM argv (same order), router argv, UMBP argv + env identical | model positional `deepseek-ai/DeepSeek-V4.1-Flash` -> `/models/DeepSeek-V4.1-Flash`; node IP in `HEAD_NODE_IP`, `VLLM_NIXL_SIDE_CHANNEL_HOST`, router `--worker-urls` |
| r2-dp2-umbp-c256 vs 448241 | identical (same as above) | same two |
| r1-dp2-c128 vs 448240 | identical; no UMBP service, no `--kv-transfer-config` (matches the row's own record) | same two |
| r4-tp2-c32 vs CI job 114309765517 (sweep_48308) | **fixed in review** -> identical, same flag order | same two |
| r4-tp2-c32-s8 | = r4-tp2-c32 + `--max-num-batched-tokens 8192 --max-num-seqs 8` | hybrid, never run in CI on this image (header says so) |

All CI rows 448240/1/2 use the same image digest `sha256:67a3a0fb...` and run 43861172833 [FH].

### Fix R1 (kit was wrong): r4 configs were not the CI command
Pre-review r4 (copies: `debug/review/r4-tp2-c32*.env.pre-review`) followed chaox `launch_server.sh`, not the CI launch:
- GPU selection `HIP_VISIBLE_DEVICES=<list>` instead of CI's `--device-ids 0,1`;
- missing srtctl-injected worker env `DYN_VLLM_KV_EVENT_PORT=5200`, `HEAD_NODE_IP`, `VLLM_NIXL_SIDE_CHANNEL_HOST`,
  `VLLM_NIXL_SIDE_CHANNEL_PORT=5400`, `VLLM_PORT=20000` (r1/r2 configs already carried these; r4's KV event port is 5200,
  not 3000).
The headers claimed "CI-exact ... verbatim", so this was a defect, not a documented choice. Now: `GPU_SELECT=device-ids`,
`--device-ids @DEVICE_IDS@`, the five env vars, CI flag order. Throughput effect of the change: unknown (chaox's
HIP_VISIBLE_DEVICES repro got 107,583 vs CI 107,989; not attributable).

### Fix R2 (kit was wrong): `--port-offset` broke run_point/watchdog
`serve.sh` copied the raw config to `runs/<tag>/config.env`; `run_point.sh` and `watchdog.sh` read `PORT`/`ROUTER_PORT` from
that file. With `--port-offset N`, run_point would have scraped the **un-offset** engine port (`:6100`, i.e. another server's
`/metrics`) for the freshness gate and `--server-metrics`. serve.sh now appends the effective `PORT`, `ROUTER_PORT`,
`ROUTER_PROM_PORT`, `VLLM_PORT`, `NIXL_PORT`, `KV_EVENT_PORT`, `KIT_PORT_OFFSET`, `KIT_GPU_LIST`, `KIT_NODE_IP` to the copy.
Verified only by code read + `bash -n` + shellcheck (the real-launch branch needs docker).

### Fix R3: provenance headers
r1 / r2-c256 headers said "UNVERIFIED: reproduce_44824x.json not downloaded"; replaced with the verified statement.

### Not a kit error, decision for E1: r4 vs r4-s8
Builder A is right [FH]: CI job 114309765517 used `max-num-batched-tokens 16384` and no `max-num-seqs`
(sweep_48308.log:120 and the engine non-default args). The 8/8192 pair exists only in the later agentic.yaml
`override_tp2_c32` (lines 261-273). SYNTHESIS s3.1 row R4 mislabels the 107,989 recipe. **Use `r4-tp2-c32` for E1**
(only it has a reference number); `-s8` is an experiment, not a reproduction.

## 2. run_point.sh contract vs InferenceX official

| Item | Official (source) | Kit | Status |
|---|---|---|---|
| Duration | 3600 s (CI config `DURATION: '3600'`; CI argv `--benchmark-duration 3600`) | `DURATION=3600` official; fast = harness `AIPERF_EXPERIMENTAL_FAST=1` (1200 s) | OK [FH] |
| Warmup | 10 req/lane, grace 1800 s (agentx preset `aiperf/common/scenario/inferencex_agentx_mvp.py:55-56`) | preset default (no override in official) | OK [FH] |
| Seed | 42 (same preset, line 53) | preset default | OK [FH] |
| Metric | `total_tput_tps / num_gpus` (`infx/results/agentic/__init__.py:237-239`) | same code path (`python -m infx.bench agentic`) | OK [FH] |
| Denominator | CI env `TP: '2'` for TP2 and DP2 rows | `TP=<gpus arg>`, refused if != config `BENCH_TP` (=2 for all five) | OK [FH, dry-run tested: gpus=4 on r4 refused] |
| Failed-request threshold | `FAILED_REQUEST_THRESHOLD = "0.10"` hard-coded (`infx/bench/agentic/run.py:31`) | same code | OK [FH] |
| Server metrics | CI: engine `:6100/metrics` **and** `:9405/metrics` | engine port only (`KIT_METRICS_URLS` to add more) | minor; :9405 owner unidentified |
| Required metric prefix | `vllm:` | `AIPERF_REQUIRED_SERVER_METRIC_PREFIX='vllm:'` | OK |
| Dataset | `semianalysis_cc_traces_weka_062126` | same, pinned to 23f152f6 offline | OK (pin stricter than CI) |
| Power | CI `ENABLE_AGENTX_POWER: '1'`, `REQUIRE_POWER: '0'` | `false`/`false` | intentional (no power exporter); does not enter tok/s |
| Fresh server | one server per point | official refuses `vllm:request_success > 0` | OK |
| Harness code version | CI head 94b81ec4 | our IX checkout db3fc796; **94b81ec4 is not in our clone** | [UNVERIFIED] harness-code equality |

Dry runs (`KIT_DRY_RUN=1`, sandbox roots under `debug/review/{nfs,local}`) for r4/r1/r2-c384 produced the expected env:
TP=2, CONC, DURATION=3600, DP_ATTENTION false/true/true, KV_OFFLOADING none/none/dram, TOTAL_CPU_DRAM_GB 0/0/449,
metrics `:8000` (r4, no router) / `:6100` (DP2) [FH].

## 3. Safety audit (grep of rm/rmdir/rmtree/docker rm/docker stop/kill/pkill over scripts/kit/*)

| Site | Target | Verdict |
|---|---|---|
| serve.sh entry `kill -TERM/-KILL "$p"` | PIDs the entry script itself spawned (inside its own container) | OK |
| stop.sh `docker stop` | exact `yihou-dsv41-<tag>` (or prefix with `--prefix`), re-checked `yihou-dsv41-*` | OK; never `docker rm` |
| run_point.sh `docker rm "$CNAME"` | only if name matches `yihou-dsv41-*-client` and state `exited` | OK |
| run_point.sh `docker stop` on SIGTERM | `$CNAME` guarded by `yihou-dsv41-*` | OK |
| client_selftest.sh `res.rmdir()` | empty dir from `tempfile.mkdtemp(prefix="yihou-selftest-")`, non-recursive | OK |
| all `docker run --rm --name` | names `yihou-dsv41-*` | OK |
| pkill / killall / rm -rf / variable recursive delete | none | OK |

`serve.sh` `chmod a+rwx` touches only `$RUN`, `$LOC`, `$LOC/logs`, all under roots checked to contain `/yihou/`.
Builder A's `/tmp/yihou_entry_check.sh` slip: file no longer exists [FH]. This review deleted one path:
`scripts/kit/__pycache__` (created by its own `py_compile`, inside the workspace).

## 4. Static checks [FH]
- `bash -n`: all 9 kit scripts, 5 configs, 6 rendered entry scripts: pass.
- `shellcheck -S warning` (shellcheck-py via uvx): kit scripts clean; rendered entry scripts only SC2206 on
  `PIDS+=($PID)` (numeric PIDs; harmless).
- `python3 -m py_compile scripts/kit/summarize.py`: pass.
- Renders regenerated with `scripts/kit/render_all.sh` after the fixes.

## 5. Open risks (ranked)
1. **R4 image is gone from Docker Hub** [FH]: `hub.docker.com/v2/repositories/vllm/vllm-openai-rocm/tags/nightly-rocm100-7d0b4e57...`
   -> 404; name filter `7d0b4e57` -> no tags. `pull_images.sh` will fail for r4/r4-s8. E1-R4 needs the image from
   somewhere else (a node's local docker cache, a colleague's `docker save`), or must be dropped / re-based on the 60bd image
   (then it is no longer a reproduction). The 60bd image (`sha256:67a3a0fb...`, 9.1 GB) exists [FH].
2. **Containers outlive a requeue** (openai.md pitfall 2026-10-20): the kit does not bind containers to the slurm job.
   After a preemption the old node keeps `yihou-dsv41-*` running with GPUs and up to 449 GB host DRAM, and ssh is lost.
   Not built here; needs a job-guard (stop own containers when `squeue -j $JOB -h -o %N` != host).
3. **Host DRAM for UMBP**: 449 GB per r2 server; Muse saw SIGKILL above ~1 TiB host tier per node (different cluster).
   Run at most 2 r2 servers per node until E0 measures free DRAM. [UNVERIFIED]
4. Harness code: CI used IX 94b81ec4, we use db3fc796 (contract constants above re-read in db3fc796 only).
5. Model weights: tokenizer files sha-identical to HF 314113f8 (builder B); weight files vs the CI HF revision not checked.
6. Router `/health` (used by `--wait` and run_point), NFS root-squash, PyPI reachability for `vllm-router==0.1.14`,
   `/root` overlay hiding image content (Muse did the same without issue), `--port-offset` sufficiency, UMBP in one
   container vs CI's separate steps -- all [UNVERIFIED] until a node.
7. GPU index: with `--device-ids` all GPUs are visible and indices are HIP indices; HIP vs rocm-smi mapping not probed
   (preflight writes the manual command).
