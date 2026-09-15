# Reproduce the completed GLM-5.2 mix experiment

This is a **single-node aggregated server**, not PD/disaggregated serving. Reproduce real correctness first, then restart for **one simulated-acceptance TP8/EP1/DP1 concurrency-1 performance point**. Commands below are for a future authorized rerun; packaging did not execute Docker, download dependencies, allocate a node, or rerun inference.

Historical budget: first launch reported 1710 s, GSM8K 2917 s, warm launch reported 540 s, performance window 3600 s; setup and a reported approximately 28-minute drain delay were additional. Reserve several hours; these are observations, not deadlines or guarantees.

## 0. Prerequisites and scope

- An exclusive 8-GPU MI355X/gfx950 node on crsuse2-m2m / amd-spur, enough host RAM for the configured HiCache, working GPU driver and Docker access. Historical measured node: crsuse2-m2m-153. Do not reuse historical job IDs 120922/120923.
- Run the following node-side commands through your **already authorized** allocation, e.g. `spur exec "$JOB_ID" bash`; arrange the current account/QOS/allocation procedure with the cluster administrator. The historical sleeper is `scripts/hold_node.sh`; this kit does not automatically submit/cancel jobs.
- Available tools: Bash, Python 3, Git, tar/gzip, zstd, Docker, rocm-smi, jq, curl, and coreutils. Do not modify a shared host to install them without authorization.
- External model: `/shared_nfs/models/GLM-5.2-MXFP4`, including tokenizer/config/custom code, mounted read-only. Exact original weight revision/hashes were not captured; directory naming alone is not identity proof.
- External image archive, **not included**: `/shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst`. The pinned build alternative below removes dependence on that path but needs public downloads.
- Code/scripts and raw evidence are packaged. Dataset contents and runtime wheels are not; pinned dataset acquisition is specified below. Cold builds/evaluation require access to Docker Hub, GitHub, PyPI/uv and Hugging Face. These are declared network dependencies, not hidden scratch-workspace paths.
- Credentials: authorized cluster identity and Docker/NFS permissions; registry credentials only if your registry policy requires them, from your approved provider. Both datasets were public. The local eval uses literal placeholder `OPENAI_API_KEY=EMPTY`; **never substitute a real secret into this shell-traced helper**. No credentials or credential stores are included.
- See `environment.md` for missing host driver/fabric, model identity and wheel-lock provenance. Never infer Docker host backing-store space from a spur namespace's `df`, or global process absence from its `/proc`/`pgrep`.

## 1. Verify package and restore source without downloads

Choose a new dedicated rerun directory (absolute path, no whitespace). Do not run in `scripts/repo`, the archived source, or the included `audit-work` tree. Commands assume the package remains at the delivered path; change `KIT` if you relocate it.

```bash
set -euo pipefail
export KIT=/shared_nfs/yihou/packups/glm52_mix_repro.packup_20260909-100729
export RUN_ROOT=/shared_nfs/yihou/playground/glm52_mix_rerun_$(date -u +%Y%m%d-%H%M%S)
export MODEL_PATH=/shared_nfs/models/GLM-5.2-MXFP4
(cd "$KIT" && sha256sum -c MANIFEST.sha256)
test ! -e "$RUN_ROOT"
mkdir -p "$RUN_ROOT/logs"
python3 "$KIT/scripts/bootstrap_repo.py" "$RUN_ROOT/repo" \
  | tee "$RUN_ROOT/logs/bootstrap.json"
export REPO="$RUN_ROOT/repo"
cd "$REPO"
```

Bootstrap uses only included snapshots and `provenance/git-snapshots.json`. It checks every blob, file mode, symlink, tree and original commit, then restores authentic **shallow tips** (parents are not bundled). It never clones/fetches, runs submodule update, or reads the old workspace. Top-level branch `main` is at `e6fea39083d05d184732ff8f7e2891067b30e640`; nested repositories are detached at InferenceX `6d6d296c6d323540e566b005e68c6522c816ed23` and aiperf `754356e9a39acc6cc6afb242d123bb57c3fb6f75`. Do not run `git submodule update`: the nested working trees are already complete.

### Explicit packaging-only adaptation

The original Dockerfile used a floating base tag, and the dataset loaders requested `main`. For a new run, apply the included input-pinning patch. This is **not a historical experiment fix**; it pins the base digest and the two observed dataset revisions without changing server/replay flags. It changes files in all three restored repositories; preserve the diff with new results.

```bash
cd "$REPO"
patch --dry-run -p1 < "$KIT/patches/pin-external-inputs.patch"
patch -p1 < "$KIT/patches/pin-external-inputs.patch"
for d in . third_party/InferenceX third_party/InferenceX/utils/aiperf; do
  printf '\nREPOSITORY %s\n' "$d"
  git -C "$d" rev-parse HEAD
  git -C "$d" diff
 done > "$RUN_ROOT/logs/reproduction-adaptations.diff"
```

The patch pins AgentX CLI prefetch **and the Python loader**, and GSM8K `dataset_kwargs.revision`. Prefetch alone would not constrain a subsequent `load_dataset(..., revision=main)`. Pinned download commands actually run later inside the original helpers. An untouched historical recipe is also reconstructable by omitting this patch, but remote base/dataset drift then remains possible; do not call that pinned reproduction.

## 2. Prepare the image — choose one route

### A. Load the existing archive (closest binary identity)

Confirm available space on the **actual Docker daemon backing filesystem** with an authorized host view first. There is no experimentally established 92-GB free-space requirement. Do not prune other users' images, change DockerRootDir, restart the daemon, or copy the archive to root disk.

```bash
export ARCHIVE=/shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst
printf '%s  %s\n' a6d047de9e3791b3c819f9035b0c3a4f632663b5cf8f85c3147b4f5b8204f9a2 "$ARCHIVE" \
  | sha256sum -c -
zstd -t "$ARCHIVE"
zstd -dc "$ARCHIVE" | docker load | tee "$RUN_ROOT/logs/image_load.log"
export IMAGE=sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d
test "$(docker image inspect -f '{{.Id}}' "$IMAGE")" = "$IMAGE"
```

Historical 09-08 save and zstd test passed. A **later independent validation on 2026-09-09** loaded this archive and recovered the exact image and SGLang/AITER identities; see `provenance/later-image-validation.md`. That is not a rerun of the archived mix measurement.

### B. Build from the included Dockerfile (not bit-identical)

The pinning patch must be applied first. The Dockerfile fetches xiaobochen-amd SGLang and AITER at fixed commits; AITER recursively fetches its submodules. It modifies only the build image, not the host. Original build workaround: writable Docker client config, legacy builder, **no `--progress`**.

```bash
cd "$REPO"
export DOCKER_CONFIG="$RUN_ROOT/docker-client-config"
mkdir -p "$DOCKER_CONFIG"
export DOCKER_BUILDKIT=0
export IMAGE=glm52-mix-repro:$(date -u +%Y%m%d-%H%M%S)
docker build \
  --build-arg SGLANG_SHA=402df1e1e453e1e85ec0f5ac4052d36598cc691a \
  --build-arg AITER_SHA=2c71811b32c8ce2e1266aedaec199df7d90f597d \
  -t "$IMAGE" . 2>&1 | tee "$RUN_ROOT/logs/docker_build.log"
docker image inspect -f '{{.Id}}' "$IMAGE" > "$RUN_ROOT/logs/rebuilt-image-id.txt"
```

Pinned base: `lmsysorg/sglang@sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72`. Different image IDs after a rebuild are possible. Dependency wheels and host runtime are not fully locked; inspect/record them, do not claim binary equivalence.

## 3. Capture environment and start real correctness

Inspect `scripts/collect_env.sh` before use; run it on the allocated node only after authorization, saving to the new workspace. It is a current snapshot, not missing historical evidence. Collect the host Docker backing storage and driver/fabric details with the administrator if the namespace cannot see them.

Use a unique container name and a free port. The original launcher **stops and force-removes a pre-existing container with the same name**, so a unique name is mandatory. Its drain gate can wait up to two hours and checks reported HBM usage; it does not prove host-global process absence. Avoid inherited overrides, especially `AITER_JIT_CACHE` (automatically keyed by image identity), `HIP_VISIBLE_DEVICES` and eval/runtime flags.

```bash
cd "$REPO"
export NAME=glm52-mix-repro-$(date -u +%Y%m%d-%H%M%S)
export PORT=8888
unset ACC_LEN AITER_JIT_CACHE HIP_VISIBLE_DEVICES HF_TOKEN EVAL_LIMIT
unset EVAL_MAX_MODEL_LEN EVAL_CONCURRENT_REQUESTS MODEL_NAME
unset AIPERF_EXPERIMENTAL_FAST AIPERF_UNSAFE_OVERRIDE AIPERF_EXTRA_INPUTS
unset AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES
bash "$KIT/scripts/collect_env.sh" "$IMAGE" > "$RUN_ROOT/logs/environment-new-node.txt" 2>&1
./glm52/launch_sglang.sh \
  "MODEL_PATH=$MODEL_PATH" "IMAGE=$IMAGE" "NAME=$NAME" "PORT=$PORT" \
  TP=8 DP=1 EP=1 CONC=1 ENABLE_HICACHE=1 \
  HICACHE_RATIO=1.5 HICACHE_IO_BACKEND=kernel HICACHE_MEM_LAYOUT=page_first \
  SPEC_STEPS=5 SPEC_DRAFT=6 ACC_LEN= \
  "SERVER_LOG=$RUN_ROOT/logs/server_correctness.log" \
  2>&1 | tee "$RUN_ROOT/logs/launch_correctness.log"
# Consume the entire input (no grep -q/SIGPIPE false negative); emit no values.
docker inspect -f '{{json .Config.Env}}' "$NAME" | python3 -c '
import json, sys
assert not any(v.startswith("SGLANG_SIMULATE_") for v in json.load(sys.stdin)), "Correctness simulation must be OFF"
'
curl -fsS "http://localhost:$PORT/server_info" > "$RUN_ROOT/logs/server_info_correctness.json"
```

Original effective server flags are all in `scripts/repo/glm52/launch_sglang.sh`: FP8 e4m3 KV; FlyDSL DSA prefill/decode; aiter top-k; chunked prefill 32768; static memory fraction 0.85; EAGLE steps5/draft6/top-k1; max-running-requests and CUDA graph max batch size1; fused allreduce and QK norm/rope; hierarchical cache ratio1.5/write-through/kernel/page-first; watchdog1800; metrics enabled. Environment uses `SGLANG_OPT_USE_TOPK_V2=false`, keep-alive900, `PYTHONNOUSERSITE=1`, FlyDSL MoE sorting and JIT grouped top-k enabled. `ACC_LEN=` suppresses all three simulation environment variables; omitting it enables the default3.61 simulation.

## 4. Correctness probes and full GSM8K

```bash
cd "$REPO"
./eval/curl.sh > "$RUN_ROOT/logs/curl.json"
# Packaging-only reconstructed inputs: original input text was not saved.
printf '%s\n' 'What is the capital of France?' 'How many people live there?' \
  'Name one famous museum there.' | ./eval/chat.sh > "$RUN_ROOT/logs/chat.txt"
# OUT_DIR is RELATIVE to REPO in the original helper. No EVAL_LIMIT.
OUT_DIR=results/correctness/eval-gsm8k EVAL_LIMIT= ./eval/eval.sh gsm8k \
  2>&1 | tee "$RUN_ROOT/logs/eval_gsm8k.console.log"
```

Expect Jupiter in `curl.json`, coherent context-dependent chat, and full `n-samples.gsm8k.effective=1319`. Original chat inputs are unknown, so that probe is semantic, not exact textual replay. Original full evaluation: five shots; local-chat-completions; apply-chat-template; sample logs enabled; client concurrent requests64 **despite server CONC=1**; max_length16384, max_tokens12288, temperature0/top_p1, timeout1800, retries5; seeds0/1234/1234/1234, limit null. The included task YAML and benchmark helper generate these exact defaults. Original accuracy is0.9681576952236542 (1277/1319), stderr0.004836348558260912, both strict-match and flexible-extract. The task specified “sane accuracy”, not a numerical pass threshold.

Check logs after real traffic: eight ranks should each report `FlyDSL sparse MLA decode engaged`, `gfx950 fused DSA indexer enabled`, and eventually `FlyDSL sparse MLA prefill engaged`. The first two occur at readiness; prefill engagement requires traffic. An HTTP-ready server or plausible number is insufficient proof that requested kernels engaged. Inspect the saved logs, including any `!! did not engage:` warning.

Stop your correctness server gracefully **before** killing its launcher/client shell. If stop times out or HBM remains high, investigate with an authorized host view and wait; do not force-kill or claim the drain symptom is a fixed kernel deadlock.

```bash
docker stop -t 300 "$NAME" | tee "$RUN_ROOT/logs/stop_correctness.log"
```

## 5. ONE simulated-acceptance performance point

After the prior server is stopped and the drain gate is satisfied, launch the original batch wrapper with a single concurrency value. `OUT_DIR` must remain **relative**, since the wrapper prefixes `$PWD`/`$REPO`. The wrapper installs AIPerf in an isolated venv, acquires the public dataset, launches the benchmark and stops its own container on exit. It does not expand to any other point.

```bash
cd "$REPO"
./run_server_agentx.sh \
  "MODEL_PATH=$MODEL_PATH" "IMAGE=$IMAGE" "NAME=$NAME" "PORT=$PORT" \
  TP=8 EP=1 DP=1 CONC=1 ENABLE_HICACHE=1 \
  HICACHE_RATIO=1.5 HICACHE_IO_BACKEND=kernel HICACHE_MEM_LAYOUT=page_first \
  SPEC_STEPS=5 SPEC_DRAFT=6 ACC_LEN=3.61 \
  DURATION=3600 TOTAL_CPU_DRAM_GB=3023 \
  OUT_DIR=results/perf/glm52_mxfp4_tp8dp1ep1 \
  2>&1 | tee "$RUN_ROOT/logs/agentx_c1.log"
python3 utils/collect.py results/perf
```

The run injects `SGLANG_SIMULATE_ACC_LEN=3.61`, `SGLANG_SIMULATE_ACC_METHOD=match-expected`, `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`. **Do not run correctness tests on this container or interpret its throughput as correctness-valid/production throughput.** Live responses are discarded for future prompt construction; original prerecorded assistant responses are replayed.

Exact client arguments, with only output paths relocated, are preserved in `results/perf/glm52_mxfp4_tp8dp1ep1/c1/benchmark_command.txt`: scenario inferencex-agentx-mvp; chat endpoint with streaming; c1, duration3600, stats30s, seed42, failed-request-threshold0.10, trajectory start ratio0.25–0.75, warmup-requests-per-lane10, trace idle gap cap300, warmup grace1800, server token counts, no GPU telemetry, tokenizer trust-remote-code, dataset entries393, timeslices1s, server metrics URL, and public-dataset `semianalysis_cc_traces_weka_062126`. Do not run that recorded line verbatim: its output path belongs to the old source workspace.

Data revisions after the pinning patch:
- AgentX `semianalysisai/cc-traces-weka-062126` @ `23f152f6f0f9399a85901b89a6458def0ef16729`, train,393 entries. `resolve_trace_source` executes `hf download --repo-type dataset ... --revision <pin>` and the AIPerf loader uses the same revision.
- GSM8K `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866`, main, test1319/train fewshot. Eval acquires it via the packaged YAML `dataset_kwargs.revision`.
- AIPerf source is pinned, but original installers resolve floating wheels and download uv/Python as needed. Observed versions are recorded in `dependencies/aiperf-observed-constraints.txt`, **not a wheel-hash lock**; no new dependency resolution was run during packaging.

## 6. Read and compare results

The new CSV is `$REPO/results/perf/results.csv`; raw JSON, command, logs and AIPerf metrics are under its `glm52_mxfp4_tp8dp1ep1/c1/` directory. The original measured CSV and the baseline CSV are already packaged separately. Never overwrite them with new results.

Original result vs2026-09-07 baseline: total(input+output)tokens/s/chip2184.42030125 vs2172.7358 (+0.538%); output16.41346625 vs16.3256; P90 interactivity261.7801047 vs261.0966057 (+0.262%); median ITL0.00363 vs0.00368s; median TTFT0.85016 vs0.85274s;249 profiled/260 total/0 errors. P90 interactivity is `1 / round(p90 full_response_itl, 5)` (not tokenwise median throughput). Eight GPUs divide total throughput. Recorded DRAM3023 is metadata, not a measured host memory amount.

No tolerance or repeatability threshold was specified. These are n=1 vs n=1 observations, not an equivalence/significance test. Compare each metric, count, effective configuration, dataset revision, image/source identity and backend engagement before describing reproduction quality.

See `notes.md` for failures/limitations, `patches/README.md` for inherited fixes and adaptation details, and `audit.md` for the **offline** checks actually performed on this kit. Once all your work is complete, release only your own allocation according to cluster policy; no cleanup of shared images, models, historical artifacts or other users' processes is part of this recipe.
