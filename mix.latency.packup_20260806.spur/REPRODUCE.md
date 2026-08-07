# Reproduction kit — GLM-5.2 MIX conc=1 agentic latency (Task 2)

Goal: reproduce the three-shape conc=1 latency table in `results/lat_summary.md`
(TTFT / E2E / TPOT at P50/P90/P99 Case-A shapes, 100% cache-hit) from a clean crsuse
node with cluster access.

Estimated time: ~30–40 min image build (first time) + ~10 min cold start + the driver
itself (p99 alone runs ~10 × ~100 s ≈ 20 min because OSL=17000; whole run ≈ 30–40 min
dominated by the p99 shape).

All commands run **on the node** via `spur exec <job> …`, or from a shell on
`crsuse2-m2m-036`. Docker here is docker-out-of-docker: containers are siblings and
survive the `spur exec` namespace teardown — but **never background a long docker
client inside `spur exec`** (teardown kills it); the latency driver is launched with
`docker exec` and its output redirected to a file, not backgrounded across the exec.

## 0. Prerequisites (arrange before you start)

- **Node:** one 8×MI355X crsuse spur node. This run used `crsuse2-m2m-036`
  (job 44901), data-plane IP `10.245.148.191`, NIC `ens3`.
- **Secrets** (values NOT here — source them yourself):
  - Docker registry login → written to `DOCKER_CONFIG=/var/tmp/dockercfg_yihou`.
  - Cluster access → spur allocation.
- **External dependencies (absolute paths):**
  - Weights: `/shared_nfs/GLM-5.2-MXFP4` (bind-mounted into the container; also the tokenizer).
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

Confirm the DSA P1V3 patch reached the **bytecode** (marker is the identifier
`_p1v2_rows`):

```bash
docker run --rm "$IMAGE" bash -lc \
  'python3 - <<PY
import importlib.util
spec = importlib.util.find_spec("sglang.srt.layers.attention.nsa.nsa_indexer")
print("module:", spec.origin)
print("_p1v2_rows in source:", "_p1v2_rows" in open(spec.origin).read())
PY'
```

## 2. Bring up the MIX deployment (etcd → kvd → worker → router)

`scripts/mix_up.sh` does teardown → fresh container (`glm52_mix`) → etcd → kvd daemon
(L2 32 GiB + L3 file 64 GiB) → mix worker → kv-aware router, waiting on `/health`
at each step. **This is the same frozen server both Task 1 and Task 2 measured** — if
it is already up from Task 1, skip to §4.

```bash
env MY_IP=$MY_IP IMAGE=$IMAGE MODEL=$MODEL bash scripts/mix_up.sh
```

Endpoint: `http://$MY_IP:8100` (router). In-container logs: worker `/tmp/glm52_mix.log`,
router `/tmp/router.log`, kvd `/tmp/kvd.log`. Cold start is minutes (weights load +
CUDA-graph capture — wait, don't kill). The worker recipe (TP8, DPA off, EP8, MTP
EAGLE, kvd, fp8 KV, ctx 262144, DSA env) is in `scripts/mix_worker.sh`.

## 3. Prove every feature is really live (before measuring)

```bash
env MY_IP=$MY_IP SERVED=$SERVED bash scripts/mix_smoke.sh
```

Expect: exactly **1 worker** (`disagg_mode=mixed`); a **coherent** answer (garbage ⇒
DSA env not live); MTP **accept-len median ~3** (4.00 = degeneration); router policy
`kv-aware` + tokenizer loaded; **kvd adapters ~one per rank** + entries>0 after traffic.

## 4. Run the conc=1 latency driver INSIDE the container

`scripts/lat_conc1.py` is the driver. Copy it into the container and run it with
`docker exec`, pointing it at the in-container router. It streams for a real TTFT.

```bash
docker cp scripts/lat_conc1.py glm52_mix:/tmp/lat_conc1.py
docker exec \
  -e URL=http://127.0.0.1:8100 \
  -e SERVED=glm5.2-mxfp4 \
  -e TOK=/shared_nfs/GLM-5.2-MXFP4 \
  -e REPEATS=10 \
  -e CACHE_HIT=0.89 \
  -e OUT=/tmp/mix_lat \
  glm52_mix python3 /tmp/lat_conc1.py 2>&1 | tee lat.console.log
```

What it does per shape `(p50,74000,320) (p90,155000,3300) (p99,235000,17000)`:
1. builds a **fixed cacheable prefix** of `round(0.89 * ISL)` tokens (the Case-A
   cache model — a shared prefix, variable fresh suffix);
2. **WARMS** that prefix once (fresh tail + 8-token gen) so it is resident — not measured;
3. fires **10 sequential** requests (conc=1, no think time): each = the same prefix
   (cache hit) + a **distinct** fresh suffix reaching ISL, with
   `max_tokens = min_tokens = OSL` and `ignore_eos: true` to force the exact fixed
   output shape; `temperature 1.0 / top_p 0.95` (checkpoint's generation_config).
   Records per-request TTFT (first streamed token), E2E, derived TPOT.

To run one shape only: add `-e SHAPES="p50"` (space-separated subset).

## 5. Copy results out and read the table

```bash
docker cp glm52_mix:/tmp/mix_lat ./lat_jsonl        # lat_p*.jsonl + lat_summary.json
python3 scripts/lat_report.py                        # reads /tmp/mix_lat/lat_*.jsonl
```

`lat_report.py` reads the per-request jsonl and prints the P50/P90 percentile table
(TTFT / E2E / TPOT + cache-hit). The captured table is `results/lat_summary.md`; the
raw per-request records are `results/lat_jsonl/lat_p{50,90,99}.jsonl` and the mean
rollup is `results/lat_jsonl/lat_summary.json`.
> Note: `lat_report.py` reads from `/tmp/mix_lat/lat_%s.jsonl`; point it at the copied
> `./lat_jsonl` (or set that as `/tmp/mix_lat`) when reading from a different machine.

## Expected output

Three jsonl files (one per shape, 10 records each) + a summary matching
`results/lat_summary.md`: **cache-hit 100%**, TPOT steady ~6 ms, E2E scaling with OSL
(≈2.5 s / 20 s / 104 s for p50 / p90 / p99), TTFT 0.4–0.9 s. Success = all three shapes
measured at conc=1 on **one frozen server** with cache-hit honored and all features
verified in §3.

## If it doesn't reproduce

See `notes.md` — the cache model, the `ignore_eos`/`min_tokens` output forcing, the
"cache-hit is warmed-prefix residence" caveat, DSA garbage, MTP degeneration,
`/tmp` not writable, and etcd entrypoint are all covered there.
