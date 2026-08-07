# Reproduction kit — GLM-5.2 MIX fixlen sweep

Goal: reproduce the 12-round fixlen throughput/latency grid in
`results/fixlen_summary.md` from a clean crsuse node with cluster access.
Estimated time: ~30–40 min image build (first time) + ~10 min cold start + the
sweep itself (p99/c24 alone runs many minutes; whole grid ≈ 1–2 h dominated by the
long-OSL rounds).

All commands run **on the node** via `spur exec <job> …`, or from a shell already on
`crsuse2-m2m-036`. Docker here is docker-out-of-docker: the daemon is the host's,
containers are siblings and survive the `spur exec` namespace teardown — but **never
background a long docker client inside `spur exec`** (the teardown kills it); use
`docker exec -d` / `nohup` inside the container as the scripts do.

## 0. Prerequisites (arrange before you start)

- **Node:** one 8×MI355X crsuse spur node. This run used `crsuse2-m2m-036`
  (job 44901), data-plane IP `10.245.148.191`, NIC `ens3`. Hold it per
  `spur-cluster-usage` / `spur-interactive-debug`.
- **Secrets** (values NOT here — source them yourself):
  - Docker registry login → written to `DOCKER_CONFIG=/var/tmp/dockercfg_yihou`.
  - Cluster access → spur allocation.
- **External dependencies (absolute paths):**
  - Weights: `/shared_nfs/GLM-5.2-MXFP4` (bind-mounted into the container).
- **Repo state:** branch `dev.yihou.glm52.mix.experiment` @ `d1a97b2`.
- **Image:** `infera/engine-sglang:final-pr` (base `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`
  — see `environment.md`).

```bash
export DOCKER_CONFIG=/var/tmp/dockercfg_yihou && mkdir -p "$DOCKER_CONFIG"
export MY_IP=10.245.148.191            # this node's data-plane IP
export IMAGE=infera/engine-sglang:final-pr
export MODEL=/shared_nfs/GLM-5.2-MXFP4 # host path; also the in-container path (bind-mount)
export SERVED=glm5.2-mxfp4
```

## 1. Build the engine image on the node

From the repo checkout at `d1a97b2` on the node:

```bash
export DOCKER_CONFIG=/var/tmp/dockercfg_yihou
docker build -f deploy/docker/Dockerfile.sglang -t infera/engine-sglang:final-pr .
```

Confirm the DSA P1V3 patch reached the **bytecode** (not just source) — the marker
is an identifier, `_p1v2_rows`:

```bash
docker run --rm "$IMAGE" bash -lc \
  'python3 - <<PY
import importlib.util, dis
spec = importlib.util.find_spec("sglang.srt.layers.attention.nsa.nsa_indexer")
print("module:", spec.origin)
src = open(spec.origin).read()
print("_p1v2_rows in source:", "_p1v2_rows" in src)
PY'
```
(Adjust the module path to whatever the DSA patch script targets; the point is to see
`_p1v2_rows` present. See `deploy/docker/patches/sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py`.)

## 2. Bring up the MIX deployment (etcd → kvd → worker → router)

`scripts/mix_up.sh` does teardown → fresh container → etcd → kvd daemon (L2 32 GiB +
L3 file 64 GiB) → mix worker → kv-aware router, and waits on `/health` at each step.

```bash
env MY_IP=$MY_IP IMAGE=$IMAGE MODEL=$MODEL bash scripts/mix_up.sh
```

Endpoint when done: `http://$MY_IP:8100` (router). Worker log inside the container:
`/tmp/glm52_mix.log`; router log `/tmp/router.log`; kvd log `/tmp/kvd.log`.
Cold start is minutes (weights load + CUDA-graph capture — wait, don't kill).

The worker recipe (TP8, DPA off, EP8, MTP EAGLE, kvd, fp8 KV, ctx 262144, DSA env)
lives in `scripts/mix_worker.sh`, staged into the container by `mix_up.sh`.

## 3. Prove every feature is really live (do this BEFORE the sweep)

```bash
env MY_IP=$MY_IP SERVED=$SERVED bash scripts/mix_smoke.sh
```

Read the four blocks, not just the exit code. Expect: exactly **1 worker**
(`disagg_mode=mixed`); a **coherent** answer (garbage ⇒ DSA env not live); MTP
**accept-len median ~3** (4.00 = degeneration); router policy `kv-aware` +
tokenizer loaded; **kvd adapters ~one per rank** + entries>0 after traffic.
This run saw: 1 worker mixed, coherent answer, accept-len median 3.12, 8 kvd
adapters, kv-aware tokenizer loaded.

## 4. Run the 12-round fixlen sweep

InferenceX-aligned `sglang.bench_serving --backend sglang-oai-chat` against the
router :8100. Shapes ISL = Case-A percentile × 10% = 7400 / 15500 / 23500, paired
OSL = 320 / 3300 / 17000, × concurrency {1,8,16,24}. One frozen server; num-prompts
= 10×C, warmup = 2×C. Flags `--random-range-ratio 1.0 --temperature 1.0 --top-p 0.95
--cache-report`.

```bash
env MY_IP=$MY_IP MODEL=$MODEL SERVED=$SERVED OUT=/tmp/mix_fixlen bash scripts/fixlen_bench.sh
```

Raw per-round jsonl lands at `glm52_mix:/tmp/mix_fixlen/*.jsonl`. Copy them out:

```bash
docker cp glm52_mix:/tmp/mix_fixlen ./fixlen_jsonl
```

## 5. Read the result

Each jsonl round reports `output_throughput` and `total_token_throughput`; divide by
8 for per-GPU. The computed grid (req/s, out tok/s, out/GPU, TTFT/TPOT/ITL/E2E
percentiles) is `results/fixlen_summary.md`; the raw console with the full
bench_serving blocks is `logs/fixlen.console.log`.

## Expected output

12 jsonl files (one per shape×conc) + a summary grid matching
`results/fixlen_summary.md`. Success = all 12 rounds complete on **one frozen
server** with all features verified in §3. This is a characterization sweep, so the
bar is completeness + verified features, not a threshold number.

## If it doesn't reproduce

See `notes.md` — DSA garbage output, MTP degeneration, `/tmp` not writable, etcd
entrypoint, and the cache-hit-is-residue caveat are all covered there.
