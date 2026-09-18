# Reproduce

Ordered and copy-pasteable. Steps 0-6 are shared by both runs; step 7 branches
into **C32** and **C40**. Total wall clock ≈ 2 h for C32, ≈ 3.5 h for C40 — most
of it the image build (~45 min), the model loads (~15 min each) and C40's
3600 s profiling.

## 0. Prerequisites

- SSH from the control machine to `crsuse2-m2m-135` and `crsuse2-m2m-138` with
  `BatchMode=yes` (no password prompt). No sudo is needed anywhere.
- Both nodes must see these at the **same absolute paths**:
  - the Infera checkout — this run used
    `/home/yihou/dev/git/infera.glm52.pd` (shared `/home`)
  - model weights `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`
  - host RDMA lib `/lib/x86_64-linux-gnu/libionic.so`
  - local NVMe scratch `/mnt/m2m_nobackup`
- Python `matplotlib` on whichever machine runs the plot (step 7).

**Secrets needed** (names and sources only — no values anywhere in this kit):

| what | where it comes from |
|---|---|
| SSH access to the two nodes | the operator's own agent/key; `SSH_OPTS` defaults to `BatchMode=yes` |
| Docker registry pull of the sglang base image | the nodes' existing docker config; the base image is public on Docker Hub |
| GitHub access to `AMD-AGI/Infera` | the operator's own credentials; only needed to obtain the checkout |

No API keys, tokens, S3 or etcd credentials are involved.

## 1. Check out the repo at the right point

```bash
cd /home/yihou/dev/git/infera.glm52.pd
git checkout dev/pd_opt/glm_5.2_agentx
git checkout 5a342acffe10e09729662ff40e81b22a4367fd75
```

## 2. Apply the two patches — the run does not work without them

```bash
git apply /path/to/packup/patches/01-router-pd-dp-rank-affinity.patch
git apply /path/to/packup/patches/02-bench-pd-dp-rank-affinity-wiring.patch
```

Patch 01 is the rust half of upstream commit `99fa0406`; patch 02 wires it into
the bench. **Patch 02 is not optional cosmetics** — without it the router
container receives no `-e` flag at all and the rust change is dead code. See
`patches/README.md`.

## 3. Install this run's config and topology

```bash
cd bench/glm5p2_pd
mkdir -p results/yihou-1p1d-c64
cp /path/to/packup/scripts/config.yihou.sh      results/yihou-1p1d-c64/
cp /path/to/packup/scripts/config.yihou.full.sh results/yihou-1p1d-c64/
cp /path/to/packup/scripts/topology.yihou.tsv   results/yihou-1p1d-c64/
```

`config.yihou.sh` is the **C32** run (sources the repo's `config.sh`);
`config.yihou.full.sh` is the **C40** run (sources `config.full.sh`, and
substitutes the two DSA backends the nightly rejects — see `notes.md` §7).

`config.yihou.sh` sources the repo's `config.sh` and overrides only what differs;
every override carries a comment explaining why. It never mutates `config.sh` or
`topology.tsv` in place.

## 4. Clear and check the nodes

```bash
./check_nodes.sh crsuse2-m2m-135 crsuse2-m2m-138
```

**Expect a non-zero exit**: crsuse2-m2m-135 GPU[1] is held by a root-owned
Kubernetes pod and will report busy. That is correct and must not be papered
over by raising `GPU_IDLE_VRAM_PCT`. What actually matters is that devices
**2,3,4,5** are idle on both nodes — check those specifically. See `notes.md` §4.

## 5. Build, distribute, verify the image

```bash
./build_image.sh build \
  CONFIG=results/yihou-1p1d-c64/config.yihou.sh \
  TOPOLOGY=results/yihou-1p1d-c64/topology.yihou.tsv
./build_image.sh distribute CONFIG=... TOPOLOGY=...   # same two args
./build_image.sh verify     CONFIG=... TOPOLOGY=...
```

`verify` must report the **same image id on both nodes**. `distribute` streams
`docker save | docker load` through the control machine and takes ~10 min.

After the build, confirm the DSA patch set landed correctly — see `notes.md` §2
for the specific slot check that the automated markers do **not** cover.

## 6. Preflight — and the pinned-NIC probe

```bash
IMAGE=infera-sglang:v0519-yihou-0917 \
HOST_RDMA_LIB=/lib/x86_64-linux-gnu/libionic.so \
OUT_DIR=results/yihou-1p1d-c64/preflight/run1 \
  ./preflight.sh crsuse2-m2m-135 crsuse2-m2m-138
```

With the stock shared device list this **fails on GPUs 4-7 in both
directions**. That is the bug this run exists to characterise, not a hardware
fault. To prove it, run the pinned-NIC probe (copied into `scripts/`):

```bash
cp /path/to/packup/scripts/pin_probe.sh results/yihou-1p1d-c64/preflight/
cd results/yihou-1p1d-c64/preflight && chmod +x pin_probe.sh
./pin_probe.sh ionic_4 pin-ionic4        # then ionic_2, ionic_3, ionic_5
```

Each must report 16/16 GPU variants verified. `pin_probe.sh` mirrors
`preflight.sh`'s `docker run` exactly and adds only
`INFERA_PREFLIGHT_RDMA_DEVICE`, which `preflight.sh` does not forward.


## 7. Launch, benchmark, analyse

Both runs follow the same shape; only `CONFIG`, `OUT_DIR` and `CONC` differ.
**Run them one at a time** — they share the same GPUs and the same
`CONTAINER_PREFIX` namespace.

### 7a. C32 — the validated baseline

```bash
cd /home/yihou/dev/git/infera.glm52.pd/bench/glm5p2_pd

./launch.sh \
  CONFIG=results/yihou-1p1d-c64/config.yihou.sh \
  TOPOLOGY=results/yihou-1p1d-c64/topology.yihou.tsv \
  OUT_DIR=results/yihou-1p1d-c64/launch/run3-perf
```

### 7b. C40 — the full target, minus the two unported items

```bash
./launch.sh \
  CONFIG=results/yihou-1p1d-c64/config.yihou.full.sh \
  TOPOLOGY=results/yihou-1p1d-c64/topology.yihou.tsv \
  OUT_DIR=results/yihou-1p1d-c64/launch/run4-full
```

C40 captures CUDA graphs at BS 128 instead of 64 and brings up the Prefill
HiCache host pool, so expect a longer startup.

### Verify before benchmarking — applies to both

Model load plus CUDA-graph capture takes ~15 min. **Watch
`OUT_DIR/server-logs/*.log`; do not kill it for being slow.** Then confirm
same-rail actually engaged:

```bash
grep "RDMA device:" <OUT_DIR>/server-logs/prefill-0.log
grep "RDMA device:" <OUT_DIR>/server-logs/decode-0.log
```

Each worker must show **four distinct NICs** (`ionic_2/_3/_4/_5`) with rail ids
(GID 2nd hextet) `0500/0600/0400/0300` — and the two nodes must agree. For C40
also confirm the substitution landed and was accepted:

```bash
ssh crsuse2-m2m-135 'docker inspect glm52-pd-yihou-1p1d-full-prefill-0 \
  --format "{{join .Config.Cmd \" \"}}"' | tr " " "\n" \
  | grep -A1 -E "dsa-prefill-backend|cuda-graph-max-bs-prefill|hicache-ratio"
```

Expect `tilelang`, `128`, `1.5`. If the engine exited at startup instead,
`flydsl` leaked through — re-read `notes.md` §7.

### Run the benchmark

From the control node, so bulk output lands on local NVMe. `DURATION` comes
from the config (`1200` for C32, `3600` for C40) — do not pass it explicitly.

```bash
# C32: CONC=32, CONFIG=…/config.yihou.sh
# C40: CONC=40, CONFIG=…/config.yihou.full.sh
REMOTE_OUT=/mnt/m2m_nobackup/yihou/agentx-results/<run>-$(date -u +%Y%m%dT%H%M%SZ)
ssh crsuse2-m2m-135 bash -s -- "$PWD" "$REMOTE_OUT" <<'REMOTE'
set -euo pipefail
cd "$1"
./agentx_bench.sh CONC=<32|40> \
  CONFIG=results/yihou-1p1d-c64/<config.yihou.sh|config.yihou.full.sh> \
  TOPOLOGY=results/yihou-1p1d-c64/topology.yihou.tsv \
  OUT_DIR="$2" \
  AGENTX_CACHE_DIR=/mnt/m2m_nobackup/yihou/agentx-cache
REMOTE
```

Warmup takes ~10 min for C32 and ~18 min for C40 (10 warmup requests per lane
instead of 1) against a cold prefix cache — `returned=0/N` early on is normal,
not a hang (`notes.md` §5). Then:

```bash
# place each run's artefacts under results/yihou-1p1d-c64/sweep/<point>/
./analyze_agentx.sh RESULT_DIR=results/yihou-1p1d-c64/sweep   # results.csv + pareto.png
./stop.sh CONFIG=<the same config> TOPOLOGY=... OUT_DIR=results/yihou-1p1d-c64/stop/<name>
```

`analyze_agentx.sh` needs `matplotlib`. If the analysis machine lacks it, a
throwaway venv works and keeps the host clean:

```bash
python3 -m venv .venv-tmp && .venv-tmp/bin/pip -q install matplotlib
.venv-tmp/bin/python tools/plot_agentx.py results/yihou-1p1d-c64/sweep
```

## 8. Check the results against the success criteria

| criterion | how to check | C32 | C40 |
|---|---|---|---|
| devices 2,3,4,5 idle pre-launch | `rocm-smi --showmemuse` per device | pass | pass |
| each rank on its own NIC, rails match across nodes | `grep "RDMA device:"` | pass | pass |
| preflight Mooncake WRITE byte-verified | `results/preflight/*.json` | pass, 16/16 × 4 rails | (shared) |
| `workers.json` has 2 workers, roles correct | `launch/*/workers.json` | pass | pass |
| AgentX JSON non-empty, error rate < 10% | `agentx_conc*.json` | **0/1150** | **0/4128** |
| no rank-affinity 503 in router | `docker logs …-router \| grep -c "PD DP-rank affinity"` | **0** | **0** |
| zero **cross-rank** Mooncake failures | worker logs | **0** | **0** |
| decode GPU fault (issue.md §3.3) | worker log | none | none (3600 s window) |
| prefill OOR (issue.md §3.4) | `rocm-smi` sampling | none, 90-91% | none, 86→91% |
| smoke / no garbled output | — | **WAIVED** | **WAIVED** |

Two entries that need reading rather than ticking:

- **C32** shows 5 `Decode transfer failed` lines, all `rank=3` (**same** rank,
  not the §3.1 cross-rank signature), in the final 10 s of a 1240 s profiling
  window, with AgentX counting 0 errors. Most plausibly teardown aborts; not
  traced per-rid, so treat that as strong inference, not established fact.
  **C40 has zero such lines** across a 3640 s window.
- **C40** has `records_error_dropped: 3` (`InvalidInferenceResultError`). Real
  errors, excluded from the headline rate. See `notes.md` §8.
