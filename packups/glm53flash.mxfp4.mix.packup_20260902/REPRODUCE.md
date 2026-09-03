# Reproduction kit — GLM-5.3-Flash-MXFP4 MIX on MI355X

Goal: bring GLM-5.3-Flash-MXFP4 up through the full infera MIX stack on one
8×MI355X node (using 4 GPUs), decode CUDA graphs on, and reproduce the fixlen
p50 numbers in `results/fixlen_p50.csv`.

Measured wall-clock on n01-33, 2026-09-02, from the log mtimes:

| step | duration |
|---|---|
| base image pull | not timed (finished 04:05:45) |
| `Dockerfile.sglang.glm53` build | **~3 min** (04:05:45 → 04:08:35; buildkit step times sum to 169 s) |
| MIX bring-up, graphs on | **~11 min** (→ 04:24:24), of which **650 s** is worker start to `/health` |
| fixlen p50 sweep (4 arms) | **~11 min** (→ 04:35:31) |
| fixlen p90 sweep (4 arms) | **~28 min** (→ 05:03:42) |

The 650 s is real. On n04-33 with a warm page cache the same bring-up took
240 s. **Do not kill it early.**

## 0. Prerequisites

**Machine.** One node with 8 × MI355X (`gfx950`), ROCm driver 6.14.14, docker.
This was done on `smci355-ccs-aus-n01-33` (numbers) and
`smci355-ccs-aus-n04-33` (ladder). Only **GPUs 0-3** are used; the other four
stay free for a second arm.

**Secrets** (values not included — arrange them yourself):
- docker.io pull access for `lmsysorg/sglang:v0.5.18-rocm720-mi35x` (public).
  If docker.io is unreachable, the Dockerfile header documents a Harbor
  pull-through mirror as a `--build-arg`.
- SSH to the host, via your normal `~/.ssh/config`.
- No HF token, no etcd credentials.

**External dependency.**
`/apps/data/models/GLM-5.3-Flash-MXFP4` — 212 GB, 120 shards, unmodified. It is
a **symlink to `/perf_apps/data/models/...` on a separate NFS mount**; bind it
explicitly (the scripts do). Upstream: `OneNexus/GLM-5.3-Flash-MXFP4`.

**Repo state.** Branch `yihou.dev.glm53.expr`. The run itself was on
`f48b79d04316907f29478f9f037d893bdf50cd4a` plus uncommitted changes; those are
now committed — **check out `ea989b3b39c30a25d659fb331a0d0dce2ab9c3e1` or
later**, which is where `deploy/docker/Dockerfile.sglang.glm53` lands. Copies of
every needed file, and the commit breakdown, are in `repo-changes/`.

## 1. Build the engine image

From the **repo root**:

    docker build -f deploy/docker/Dockerfile.sglang.glm53 \
      -t infera/engine-sglang:glm53-c821c425 .

The pin `ARG SGLANG_GLM53_REF=c821c425c31b0e6c8151324b60fbc2857c39eaef` is
deliberate — read the Dockerfile header before changing it. Reference build log:
`logs/n0133/build_glm53.log.gz`.

**Verify the overlay landed** before spending a bring-up on it:

    docker run --rm infera/engine-sglang:glm53-c821c425 bash -lc '
      git -C /sgl-workspace/sglang rev-parse HEAD
      wc -l /sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py
      wc -l /sgl-workspace/sglang/python/sglang/srt/layers/quantization/quark/quark.py'

Expect exactly:

    c821c425c31b0e6c8151324b60fbc2857c39eaef
    1942 .../glm5_next.py
    1172 .../quark.py

1834 / 1103 means you built the wrong pin (`9e692c92`) — that image cannot load
MXFP4 and silently takes the pre-mHC slow path.

## 2. Bring up the MIX stack

`scripts/mix_up.sh` does the whole sequence: teardown → GPU reset gate →
container → etcd → infera worker → kv-aware router.

    export MY_IP=<this node's data-plane IP, e.g. 10.235.192.136>
    CUDA_GRAPH=1 TP=4 GPUS=0,1,2,3 \
    SERVED=glm5.3-flash-mxfp4 \
    MODEL=/apps/data/models/GLM-5.3-Flash-MXFP4 \
    IMAGE=infera/engine-sglang:glm53-c821c425 \
    bash scripts/mix_up.sh 2>&1 | tee mix_up_graphs.log

Three defaults in `scripts/mix_worker.sh` are load-bearing and are **not**
sglang defaults — read that file's header before changing any of them:

- `--disable-shared-experts-fusion` (via `SHARED_EXPERT_FUSION=0`, the default)
  — **without this the model does not load at all.** See `results/root_cause.md`.
- `SGLANG_USE_AITER=1` — gates the AITER mHC path. Without it the server starts,
  answers correctly, and is 4.3-5.4× slower with nothing in any log saying so.
- `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0` — vendor-set for this checkpoint.

`ETCD_PORT` defaults to **12379**, not 2379: port 2379 was held by a foreign
host etcd on one of these nodes, and a foreign process is never killed.

Expected tail:

    worker serving after 650s
    router healthy
    ===== up. endpoint: http://<MY_IP>:8100 =====

## 3. Health checks — these matter more than "no errors"

    bash scripts/mix_smoke.sh          # MY_IP set, SERVED=glm5.3-flash-mxfp4

and, on the worker log inside the container:

    docker exec glm53_mix bash -c '
      L=/tmp/glm53_mix.log
      echo -n "mHC lines (expect 8): "; grep -ac mHC $L
      echo -n "shared-expert fusion (expect 0): "; grep -ac "Shared experts fusion optimization enabled" $L
      echo -n "decode lines with BOTH pools: "; grep -a "Decode batch" $L | grep -ac "mamba usage"
      echo -n "memory access fault: "; grep -ac "memory access fault" $L
      echo -n "HIP error: "; grep -ac "HIP error" $L
      echo -n "Traceback: "; grep -ac Traceback $L'

Reference readings from the run that produced the numbers (n01-33, 2026-09-02):

| check | expected | measured |
|---|---|---|
| AITER mHC lines | 8 = 2 per rank × 4 (`Using AITER gfx950 mHC pre/post kernels` ×4, `Using fused AITER mHC attention-to-FFN boundary` ×4) | **8** |
| `Shared experts fusion optimization enabled.` | 0 | **0** |
| decode lines carrying `full token usage` **and** `mamba usage` | many | **3123** at the 04:59 UTC health check; **4818** in the captured `logs/n0133/worker_glm53_mix.log.gz`, which kept growing through the p90 sweep |
| `cuda graph:` in decode lines | `True` | `True` |
| `memory access fault` / `HIP error` / `Traceback` | 0 / 0 / 0 | **0 / 0 / 0** |
| `max_total_num_tokens` | — | 7650368, `available_gpu_mem 54.35 GB` |
| Mamba (KDA) cache | — | `max_mamba_cache_size: 2371, conv_state 2.77GB, ssm_state 78.76GB` |

Note the **Traceback** row. On the earlier bare-sglang rung-0 run there were 8
`Traceback` hits, all from `torch/_dynamo/metrics_context.py` (2 per rank) —
`torch.compile` telemetry, not the model path. They were recorded rather than
waved through (`results/rung0_nosef.md`). On the MIX run measured here the count
is 0.

Router-side, `scripts/mix_smoke.sh` should show `/v1/workers` with exactly **1**
worker at `disagg_mode: mixed`, `/v1/models` reporting `glm5.3-flash-mxfp4`,
and `17 * 23` → `391` with `reasoning_content` separated from `content`.

## 4. fixlen sweep

    HOST=<MY_IP> PORT=30000 MODEL=glm5.3-flash-mxfp4 \
    ARM=p50 ISL=7400 OSL=320 CONCS="1 8 16 24" \
    OUT=$PWD/fixlen_p50.csv \
    bash scripts/fixlen_sweep.sh

Note it targets the **worker** (`:30000`), not the router, so the numbers are
engine throughput and not router overhead. Three flags in that script are
load-bearing and are documented in its header: `--random-range-ratio 1.0`,
`--temperature 1.0 --top-p 0.95` (the checkpoint's own `generation_config`;
greedy makes this reasoning model repeat on long prompts), and
`--num-prompts 10 × conc`.

The p90 arm is the same command with `ARM=p90 ISL=15500 OSL=3300`.

## Expected output

`fixlen_p50.csv` should match `results/fixlen_p50.csv`:

| conc | out tok/s | TTFT p50 ms | TPOT mean ms |
|---:|---:|---:|---:|
| 1 | 111.02 | 255.29 | 7.92 |
| 8 | 561.04 | 1064.73 | 10.81 |
| 16 | 962.55 | 743.67 | 12.36 |
| 24 | 1391.15 | 619.42 | 13.11 |

Compare against GLM-5.2 only with the three caveats in
`results/fixlen_vs_glm52.md`.

## If it doesn't reproduce

`notes.md` has the full list. The three that cost the most time here:

1. **`RuntimeError: The size of tensor a (256) must match the size of tensor b
   (512)`** in `_load_w2` → shared-expert fusion is on. Check the log for
   `Shared experts fusion optimization enabled.` and add
   `--disable-shared-experts-fusion`.
2. **`Unrecognized processing class`** → you bound `/apps` instead of
   `/apps/data/models`; the model dir is its own NFS mount and the tokenizer
   files are dangling symlinks inside the container.
3. **`The memory capacity is unbalanced. Some GPUs may be occupied by other
   processes.`** → a previous round's VRAM has not been reclaimed.
   `scripts/reset_gpus.sh` is the gate; it kills only our own containers'
   processes and **aborts** rather than touching a foreign one.
