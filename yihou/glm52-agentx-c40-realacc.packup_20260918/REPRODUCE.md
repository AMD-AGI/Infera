# Reproduce

Cold reproduction of the AgentX C40 real-acceptance point. Roughly 1 h 15 min
wall clock: ~1 min image layer, ~13 min bring-up, ~40 min benchmark.

## 0. Prerequisites

| | |
|---|---|
| nodes | `crsuse2-m2m-135` (prefill, `10.245.148.209`), `crsuse2-m2m-138` (decode, `10.245.157.237`), passwordless ssh from the control node |
| control / builder | `crsuse2-m2m-135` |
| GPUs | devices **2,3,4,5** on each node, idle |
| base image | `infera-sglang:v0519-yihou-0917` present on **both** nodes |
| model | `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` readable on both |
| host RDMA provider | `/lib/x86_64-linux-gnu/libionic.so` on both |
| repo | `AMD-AGI/Infera` @ `ad3b85d3` on `dev/pd_opt/glm_5.2_agentx` |

Secrets: none are needed for the run itself. AgentX pulls its trace corpus from
Hugging Face (`semianalysisai/cc-traces-weka-062126`); if that dataset is gated
in your account, `HF_TOKEN` must be present in the environment of the control
node. No token value appears anywhere in this pack-up.

**Do not touch crsuse2-m2m-135 GPU[1]** — root-owned k8s vLLM pod (Qwen3-32B,
~97 GB). `check_nodes.sh` flags it; that is expected, and `GPU_IDLE_VRAM_PCT`
must not be relaxed to hide it.

## 1. Apply the uncommitted `engine.sh` change

The run needs `DECODE_EXTRA_ARGS`, which is not in the tracked tree:

```bash
cd <repo>
git apply patches/0001-engine-sh-extra-args-env.patch
```

It adds role-scoped `*_EXTRA_ARGS` / `*_EXTRA_ENV` escape hatches and makes
`--enable-aiter-allreduce-fusion` conditional. All three default to the previous
behaviour, so the patch is a no-op unless a config opts in.

## 2. Build the image layer on **each** node

The base is 66 GB and already present on both — build locally on each, never
`docker save`.

```bash
for n in crsuse2-m2m-135 crsuse2-m2m-138; do
  ssh $n mkdir -p /tmp/yihou-hicache-build
  scp patches/Dockerfile.yihou.hicache \
      patches/pr37152.sources.yihou.diff $n:/tmp/yihou-hicache-build/
  ssh $n 'cd /tmp/yihou-hicache-build && docker build \
      -t infera-sglang:v0519-yihou-0917-nextnfix-hicache \
      -f Dockerfile.yihou.hicache .'
done
```

If your base is the plain `v0519-yihou-0917` without the NextN fusion fix, build
that layer first from `patches/Dockerfile.yihou.nextnfix` +
`patches/apply_nextn_fusion_fix.sh`, or pass
`--build-arg BASE=infera-sglang:v0519-yihou-0917` to skip it — R06 in the
root-cause pack-up showed the fusion fix is **not** required for correct output.

**Verify in a running container, not from the build log.** `docker build` has no
GPU, so anything that imports `sglang.srt.*` fails there with
`No HIP GPUs are available`; and `__pycache__` is keyed on mtime, so grepping
source proves nothing about what will run:

```bash
ssh $n 'docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=2 infera-sglang:v0519-yihou-0917-nextnfix-hicache \
  bash -c "
    grep -c kCopyGroupThreads /sgl-workspace/sglang/python/sglang/kernels/jit/csrc/kvcacheio/hicache.cuh
    python3 -c \"import sglang.kernels.ops.kvcache.hicache as m; print(m._tiles_across_lanes)\"
    grep -n can_use_jit /sgl-workspace/sglang/python/sglang/srt/mem_cache/pool_host/mha.py
    python3 -c \"from sglang.srt.models.glm4_moe import GlmMoeDsaForCausalLMNextN as C; print(C.fused_shared_experts_architecture)\"
  "'
```

Expect: marker count > 0; `_tiles_across_lanes` imports; the `mha.py` gate reads
`(_is_cuda or _is_hip) and can_use_hicache_jit_kernel(`; the last line prints
`GlmMoeDsaForCausalLMNextN`.

## 3. Place the configs

`config.yihou.base.sh` reaches the repo `config.sh` via `../..`, so the config
directory must sit **exactly two levels** under `bench/glm5p2_pd/`:

```bash
mkdir -p bench/glm5p2_pd/results/yihou-agentx-hicache
cp scripts/config.yihou.*.sh scripts/topology.yihou.tsv \
   scripts/collect_accept.yihou.sh scripts/sweep.yihou.sh \
   bench/glm5p2_pd/results/yihou-agentx-hicache/
```

Self-check before launching — this catches the one subtlety that silently
changes the experiment:

```bash
cd bench/glm5p2_pd/results/yihou-agentx-hicache
bash -c 'set -a; source ./config.yihou.fast.sh; set +a;
  echo "SIMACC=[${DECODE_SIMULATE_ACC_LEN-UNSET}] EXTRA=$DECODE_EXTRA_ARGS HICACHE=$PREFILL_HICACHE"'
```

Must print `SIMACC=[]` — **empty but set**. `config.sh` uses
`${DECODE_SIMULATE_ACC_LEN-3.61}` with a *single* dash, so an empty-but-set value
survives and disables simulation; `:-` would silently restore 3.61 and you would
be measuring a forced acceptance length instead of a real one.

## 4. Launch

```bash
cd bench/glm5p2_pd
W="$PWD/results/yihou-agentx-hicache"
./launch.sh CONFIG=$W/config.yihou.fast.sh \
            TOPOLOGY=$W/topology.yihou.tsv \
            OUT_DIR=$W/t1-launch
```

`TOPOLOGY=` is not optional: the tracked `topology.tsv` points at
136/137/140, not these nodes. `launch.sh` refuses a pre-existing `OUT_DIR`.

Prefill is healthy in ~3 min; decode takes ~13 min (MTP CUDA-graph capture).
Watch `OUT_DIR/server-logs/decode-0.log` rather than idling.

Confirm from the launcher's own emitted command line, not from the config:

- decode carries `--disable-custom-all-reduce`,
- decode carries `--speculative-algorithm EAGLE --speculative-num-steps 5
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 6`,
- **no `SGLANG_SIMULATE_ACC_LEN` appears anywhere** — real acceptance is proven
  by that variable's absence,
- `HiCache=0` on both legs.

And from `server-logs/decode-0.log`: `Shared experts fusion optimization enabled`
**2×**, `Config does not support fused shared expert(s)` **0×**.

## 5. Correctness probe

```bash
ROUTER=http://10.245.148.209:28000
N="YIHOU$(date -u +%H%M%S)"
for p in "What is 2+2? Answer with a single digit." "Reply with exactly: BASELINE_OK_$N"; do
  curl -sS "$ROUTER/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys;print(json.dumps({"model":"glm5.2-mxfp4","messages":[{"role":"user","content":sys.argv[1]}],"temperature":0,"max_tokens":64}))' "$p")"
done
```

Coherent text, and the nonce reproduced verbatim.

## 6. Benchmark

```bash
bash $W/collect_accept.yihou.sh $W/t1-accept before     # expect all-zero; see below
./agentx_bench.sh CONC=40 CONFIG=$W/config.yihou.fast.sh \
                  TOPOLOGY=$W/topology.yihou.tsv \
                  OUT_DIR=$W/t1-agentx-c40
```

`config.sh` already sets `AGENTX_DURATION=1200` and
`AGENTX_WARMUP_REQUESTS_PER_LANE=1`, which is exactly what InferenceX's
`AIPERF_EXPERIMENTAL_FAST=1` does (`benchmarks/benchmark_lib.sh:1979` sets those
two values and nothing else). So this **is** fast mode; the env var is not
plumbed through `agentx_env.py` and must not be set separately.

Dataset setup takes 4–14 min before profiling starts. That is normal.

**Read acceptance while the profiling phase is live:**

```bash
DECODE_LOG=$W/t1-launch/server-logs/decode-0.log \
  bash $W/collect_accept.yihou.sh $W/t1-accept after
```

The `before` snapshot reads `0.0` on every rank. That is **not** a measurement of
zero acceptance — an idle rank reports `spec_accept_length 0.0`, indistinguishable
from "accepts nothing". It is kept only to make the trap visible. Only ranks whose
gauge moved count.

## 7. Analyse the output shortfall — from the JSONL, per request

```bash
python3 - <<'PY'
import json
recs=[json.loads(l) for l in open('aiperf_artifacts/profile_export.jsonl') if l.strip()]
prof=[r for r in recs if r['metadata'].get('benchmark_phase')=='profiling']
g=lambda r,k:(r['metrics'].get(k) or {}).get('value')
tot_a=tot_r=0.0
for r in prof:
    a=g(r,'output_sequence_length') or 0
    p=g(r,'osl_mismatch_diff_pct')
    tot_a+=a; tot_r+= a if (p is None or abs(p)<1e-9) else a/(1+p/100.0)
print(f"deficit {100*(tot_r-tot_a)/tot_r:.2f}%")
PY
```

**Do not** derive this from the aggregate JSON's
`request_metrics.tokens.output_expected` mean. It is a differently-defined field,
and the C72 pack-up records a correction where exactly that substitution inflated
the figure by more than 10×.

## 8. Tear down

```bash
./stop.sh CONFIG=$W/config.yihou.fast.sh TOPOLOGY=$W/topology.yihou.tsv
```

`stop.sh` prints `cleanup completed with errors` even on a clean teardown. Verify
rather than trust it: no `glm52-pd-yihou-*` containers on either node and
`rocm-smi` reporting no KFD processes.
