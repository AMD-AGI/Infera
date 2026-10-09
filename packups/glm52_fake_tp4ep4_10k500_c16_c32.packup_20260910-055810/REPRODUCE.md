# Reproduce the two-point fake-decode measurement

This is a synthetic-KV/simulated-acceptance benchmark, not genuine PD or semantic correctness. **Offline commands do not use GPUs. Replay commands below are for a future authorized allocation; none were executed during packaging.** Never reuse expired historical job130891 or run original scripts against the old workspace.

## 1. Offline audit, no cluster access

```bash
KIT=/shared_nfs/yihou/packups/glm52_fake_tp4ep4_10k500_c16_c32.packup_20260910-055810
export PYTHONDONTWRITEBYTECODE=1
python3 "$KIT/scripts/audit_package.py"
python3 "$KIT/scripts/recompute_metrics.py"
(cd "$KIT" && sha256sum --check MANIFEST.sha256)
```

The kit can be moved: set KIT to its new absolute path. Python3.10+, GNU patch and bash are required. The audit temporarily writes only under kit/provenance and cleans its own temporary files; it requires a writable kit directory. It verifies lossless provenance, patch bytes, AST/shell syntax, red/green guard tests, fresh-workspace isolation, metric consistency and checksum coverage. Original experiment paths are not needed unless explicitly adding `--verify-sources`.

## 2. Prepare a NEW runtime workspace on shared storage

```bash
RUN=/shared_nfs/yihou/playground/glm52_fake_short_replay_$(date -u +%Y%m%dT%H%M%S)
NAME=glm52-fake-short-replay-$(date -u +%H%M%S)
PORT=31816
python3 "$KIT/scripts/prepare_workspace.py" "$RUN" --name "$NAME" --port "$PORT"
```

The parent directory must already exist; RUN itself must not. The helper rejects destinations inside this kit or either original experiment, including symlink-resolved paths. It copies only needed source/input/test/build files and preserves scripts under `scripts/original/`. Generated `scripts/launch.sh`, `bench.sh`, `wait_ready.py` differ **only** in workspace path, container name and optional port. `state/preparation.json` records substitutions and hashes. This helper is **new packaging code**, never part of the measured run. No old caches, logs or rounds are copied into the runtime workspace; caches start empty. Never execute `scripts/original/` directly.

## 3. Arrange external resources and exact image

Use your own currently authorized allocation on an idle eight-MI355X host with host Docker access, enough shared disk, the local model/tokenizer at `/shared_nfs/models/GLM-5.2-MXFP4`, and sufficient walltime. Only GPUs0–3 are used by the server, but the original safety probe requires all eight VRAM allocations <=2%. This does not grant permission to stop another workload or to use peer056. Arrange availability rather than removing the safety gate.

Set JOB to that allocation's ID. In examples, **LOGIN** means the login shell with Spur; **HOST** means its authorized host Docker context opened using `spur exec "$JOB" bash`. Reassign RUN, NAME and PORT in the host shell; these shell variables do not automatically propagate. Read [environment.md](environment.md) for access, pinned versions and capture gaps.

```bash
# LOGIN: JOB must already be assigned to your currently authorized allocation.
: "${JOB:?Set JOB to an authorized current allocation}"
spur exec "$JOB" bash
```

```bash
# HOST: assign the same RUN, NAME and PORT used in step2.
IMAGE=sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d
# If already resident, the first command is sufficient.
docker image inspect "$IMAGE" --format '{{.Id}}'
# If missing, load the external archive (not bundled):
ARCHIVE=/shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst
sha256sum "$ARCHIVE"
# Compare with a6d047de9e3791b3c819f9035b0c3a4f632663b5cf8f85c3147b4f5b8204f9a2.
set -o pipefail
zstd -dc "$ARCHIVE" | docker load
docker image inspect "$IMAGE" --format '{{.Id}}'
```

The archive hash is inherited from the prior package, not newly hashed here. Require the exact image ID. No registry base-image digest was captured. If the archive is unavailable, `reference/Dockerfile` provides the original SHA-pinned build recipe; its mutable base tag/transitive packages mean rebuilding is not binary-equivalent. Build only within the new workspace and explicitly record any generated launcher's changed IMAGE value; never label that an exact-image reproduction or modify kit evidence. Image/model/caches are excluded and no full model shard manifest is available.

## 4. CPU regression tests (no model load)

```bash
# LOGIN
python3 "$RUN/scripts/tests/test_index_elision.py" "$RUN/source/kv_cache_configurator_fake_fix.py"
python3 "$RUN/scripts/tests/test_fake_topk_domain.py" "$RUN/source/dsa_utils_fake_fix.py"
# HOST: optional original proposal test with the exact image dependencies.
# No GPU device options; imports and test tensors run on CPU.
docker run --rm -w / -v "$RUN:$RUN:ro" -e PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint python3 "$IMAGE" \
  "$RUN/scripts/tests/test_fake_proposal.py" "$RUN/source/eagle_disaggregation_fake_fix.py"
```

Expected guards:72/48 subcases pass; proposal:4 tests pass. Packaging reran the two dependency-free guard tests, not the proposal test (local Torch absent, no Docker permitted). Do not confuse the test dependency gap with a failed runtime test.

## 5. C16 server, readiness, benchmark and safe restart

Confirm the selected PORT is unused and NAME is fresh. Keep original optimization flags, four selected GPUs, TP4/EP4/DP1, memory0.85, no HiCache, FP8 KV, EAGLE5/6/topk1, fake decode and simulation3.61.

```bash
# HOST
bash "$RUN/scripts/launch.sh" sim c16 0.85 16
# LOGIN (same RUN; initialization can take >30min without bundled JIT caches)
python3 "$RUN/scripts/wait_ready.py" "$JOB" c16 --seconds 3600 \
  > "$RUN/rounds/c16/readiness-monitor.log" 2>&1
# HOST, after readiness succeeds
set -o pipefail
bash "$RUN/scripts/bench.sh" c16 128 16 2>&1 | tee "$RUN/rounds/c16/benchmark-console.log"
docker logs "$NAME" > "$RUN/rounds/c16/server-full.log" 2>&1
curl -fsS "http://127.0.0.1:$PORT/server_info" > "$RUN/rounds/c16/server-info-after.json"
docker stop --timeout 300 "$NAME" > "$RUN/rounds/c16/stop.log" 2>&1
docker inspect "$NAME" --format '{{.State.Running}} {{.State.ExitCode}}' \
  > "$RUN/rounds/c16/exit-state.txt"
# Inspect exit-state: require false and a normal exit before continuing.
# Preserve stopped container; use a unique suffix if necessary.
docker rename "$NAME" "${NAME}-c16-stopped"
```

Stop if exit confirmation fails; do not force-kill unknown processes or reset the host. The next launch repeats the eight-GPU idle gate. A readiness timeout leaves the server intact for diagnosis; inspect earliest worker error, not just PID1 status. Ensure enough allocation time before extending readiness.

## 6. C32: only max-running/graph-max and client concurrency change

```bash
# HOST
bash "$RUN/scripts/launch.sh" sim c32 0.85 32
# LOGIN
python3 "$RUN/scripts/wait_ready.py" "$JOB" c32 --seconds 3600 \
  > "$RUN/rounds/c32/readiness-monitor.log" 2>&1
# HOST
set -o pipefail
bash "$RUN/scripts/bench.sh" c32 128 32 2>&1 | tee "$RUN/rounds/c32/benchmark-console.log"
docker logs "$NAME" > "$RUN/rounds/c32/server-full.log" 2>&1
curl -fsS "http://127.0.0.1:$PORT/server_info" > "$RUN/rounds/c32/server-info-after.json"
docker stop --timeout 300 "$NAME" > "$RUN/rounds/c32/stop.log" 2>&1
docker inspect "$NAME" --format '{{.State.Running}} {{.State.ExitCode}}' \
  > "$RUN/rounds/c32/exit-state.txt"
docker rename "$NAME" "${NAME}-c32-stopped"
# Task-owned GPU probe; no monitor/collector modification.
docker run --rm --device=/dev/kfd --device=/dev/dri --entrypoint bash "$IMAGE" \
  -c 'rocm-smi --showmemuse; rocm-smi --showpids' > "$RUN/state/final-gpu.txt" 2>&1
```

Inspect exit state and GPU release. KFD listing entries with zero GPU count/bytes/occupancy do not mean an active GPU workload. Preserve collectors/exporters. Release only allocations you created or were explicitly asked to release; do not replay historical stop/cancel commands. Containers above are stopped/preserved; no image or stopped-container deletion is required.

## 7. Expected evidence and interpretation

Each point has16 warmups,128 measured successes, zero errors, all actual input lengths10000 and output lengths500,64,000 generated output tokens. Temperature0 and ignoreEOStrue are native client defaults (metric/client implementation in `source/baseline/serving.py`). JSONL preserves reported total/input throughput too, but **never report fake input tokens as real served-prefill throughput**. Authoritative output token counts come from server completion IDs, not retokenized generated text.

C16 observed1803.145482 output tok/s, P50 TPOT8.775409ms; C32 observed2569.097920 and11.632978ms. These are reference observations, not acceptance thresholds or guarantees. Check sampled `#running-req`, `#retracted-req` and graph state. C32's192-row verification must be reported as above the FlyDSL96 gate with fallback; do not lower batch/MTP or suppress fallback just to make it appear entirely optimized.

The packaged recomputation tool audits **historical packaged** rounds, not arbitrary reruns; it asserts their exact scheduler sample counts. For a new measurement inspect its new JSONL/logs separately and retain different observed rates/counts honestly. Exact per-request latency was not serialized by the native client: ITL-sum reconstruction closely estimates TPOT but cannot reconstruct its last-response overhead exactly. Acceptance is cross-checked with embedded server_info, not missing raw acceptance counters.
