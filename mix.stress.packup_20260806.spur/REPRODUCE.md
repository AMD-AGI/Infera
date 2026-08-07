# Reproduction kit — GLM-5.2 MIX Task 3 agentic stress (Case-A, closed-loop)

Goal: reproduce the sustain-phase saturation numbers in `results/stress_summary.md`
(TTFT/TPOT, cache-hit 89%, live sessions ~24, in-flight ~16) from a clean crsuse
node with cluster access.

Estimated time: ~30–40 min image build (first time) + ~10 min cold start + the
stress run itself = **ramp 400 s + sustain 3600 s + ~5 s drain ≈ 67 min**.

Bring-up runs **on the node** via `spur exec <job> …`; the stress driver runs on
the **node host** (not inside the container — the image has no `agent` module).
Docker here is docker-out-of-docker: containers are siblings and survive the
`spur exec` teardown — but **never background a long docker client inside
`spur exec`** (teardown kills it).

## 0. Prerequisites (arrange before you start)

- **Node:** one 8×MI355X crsuse spur node. This run used `crsuse2-m2m-036`
  (job 44901), data-plane IP `10.245.148.191`, NIC `ens3`.
- **Secrets** (values NOT here — source them yourself):
  - Docker registry login → written to `DOCKER_CONFIG=/var/tmp/dockercfg_yihou`.
  - Cluster access → spur allocation.
- **External dependencies (absolute paths):**
  - Weights: `/shared_nfs/GLM-5.2-MXFP4` (bind-mounted into the container; also the tokenizer).
  - Agent venv: `/shared_nfs/yihou_agentbench/venv` (holds the `agent` module + deps).
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

`scripts/mix_up.sh` does teardown → fresh container (`glm52_mix`) → etcd → kvd
daemon (L2 32 GiB + L3 file 64 GiB) → mix worker → kv-aware router, waiting on
`/health` at each step. **This is the same frozen server Tasks 1 and 2 measured**
— if it is already up, skip to §4.

```bash
env MY_IP=$MY_IP IMAGE=$IMAGE MODEL=$MODEL bash scripts/mix_up.sh
```

Endpoint: `http://$MY_IP:8100` (router). In-container logs: worker
`/tmp/glm52_mix.log`, router `/tmp/router.log`, kvd `/tmp/kvd.log`. Cold start is
minutes (weights load + CUDA-graph capture — wait, don't kill). The worker recipe
(TP8, DPA off, EP8, MTP EAGLE, kvd, fp8 KV, ctx 262144, DSA env) is in
`scripts/mix_worker.sh`.

## 3. Prove every feature is really live (before measuring)

```bash
env MY_IP=$MY_IP SERVED=$SERVED bash scripts/mix_smoke.sh
```

Expect: exactly **1 worker** (`disagg_mode=mixed`); a **coherent** answer (garbage
⇒ DSA env not live); MTP **accept-len median ~3** (4.00 = degeneration); router
policy `kv-aware` + tokenizer loaded; **kvd adapters ~one per rank** + entries>0
after traffic.

## 4. Run the closed-loop agentic stress FROM THE HOST venv

The driver is the INFERA `agent.agent_throughput` (NOT the customer/AgentX bench —
mission rule 5). It runs on the **node host** via the staged venv and talks to the
router on `$MY_IP:8100`. `scripts/run_stress.sh` wraps the invocation; the
workload is `scripts/stress_caseA.yaml`.

```bash
# stage the workload where run_stress.sh expects it (or override YAML=)
mkdir -p /shared_nfs/yihou_final_pr/mix/scripts
cp scripts/stress_caseA.yaml /shared_nfs/yihou_final_pr/mix/scripts/stress_caseA.yaml

env MY_IP=$MY_IP \
    V=/shared_nfs/yihou_agentbench/venv/bin/python \
    YAML=/shared_nfs/yihou_final_pr/mix/scripts/stress_caseA.yaml \
    NAME=mix_stress_caseA \
    DATADIR=/shared_nfs/yihou_final_pr/mix/results/stress \
    bash scripts/run_stress.sh 2>&1 | tee stress.console.log
```

The exact command `run_stress.sh` issues:

```bash
/shared_nfs/yihou_agentbench/venv/bin/python -m agent.agent_throughput \
  --workload-config /shared_nfs/yihou_final_pr/mix/scripts/stress_caseA.yaml \
  --server http://10.245.148.191:8100 --model glm5.2-mxfp4 \
  --tokenizer /shared_nfs/GLM-5.2-MXFP4 \
  --name mix_stress_caseA --data-dir /shared_nfs/yihou_final_pr/mix/results/stress \
  --dashboard-mode
```

The workload (`stress_caseA.yaml`) is **closed-loop / `--mode realistic`**: each
live session issues one request at a time and waits before its inter-turn delay,
so offered load is set by the **live-session count**, never a QPS field. The
operator's stress knobs: `initial_sessions: 8`, `max_inflight: 16`,
`max_sessions: 24` (mission-specified), `new_session_rate: 0.20`,
`ramp_duration: 400` (warmup exclusion), `sustain_duration: 3600` (honest window).
Case-A request shape (ISL p50/p90/p99 = 74K/155K/235K, OSL = 320/3300/17000,
turns/session 3/20/103, inter-turn delay, `cache_hit_rate: 0.89`) is unchanged.

> **`new_session_rate` must be re-solved to your server's E2E** — see §Expected
> output and `notes.md`. The value 0.20 here was solved off run 1; the default in
> the yaml comments (0.10, for N=32 @ E2E~15 s) under-loads this server.

## 5. Read the result

The driver writes into `--data-dir` (`.../results/stress`):
- `summary.json` — per-phase rollups (ramp / **sustain** / drain) + whole-run totals
- `metrics.jsonl` — per-window (30 s) samples across the whole run
- `metadata.json` — the resolved run parameters (the knobs that actually took)

Copy them out and read the **sustain** phase (the reported number):

```bash
cp /shared_nfs/yihou_final_pr/mix/results/stress/{summary.json,metrics.jsonl,metadata.json} ./
python3 -c "import json;p=[x for x in json.load(open('summary.json'))['phases'] if x['phase']=='sustain'][0];print('sustain:',p['completed'],'reqs',round(p['qps'],3),'qps  cache',round(p['cache_hit_rate'],3),'  ttft_p50',round(p['ttft_p50_ms']),'  tpot_p50',round(p['tpot_p50_ms'],1))"
```

The captured table is `results/stress_summary.md`.

## Expected output

Sustain phase (3600 s): **2092 requests**, success **98.2%**, offered
**0.58 req/s**, **cache-hit 89.0%**, **live sessions ~23–24** (at the session
cap), **in-flight ~15–16** (at the in-flight cap ⇒ backpressure binds — a genuine
saturation point), **TTFT p50/p90 1460/3610 ms**, **TPOT p50/p90 19.0/32.6 ms**,
input **2.99M TPM**, uncached **41.3K TPM/GPU**. These agree with the reference
kit (TPOT p50 ~14–16 ms, prefix cache ~88%).

## If it doesn't reproduce

See `notes.md` — the closed-loop semantics, the `new_session_rate` re-solve
(run 1 calibration → run 2 result), why the in-flight cap binding is the point,
DSA garbage, MTP degeneration, host-vs-container driver, and cluster gotchas
(`/tmp` not writable, etcd entrypoint) are all covered there.
