# Environment

## Hardware

| | |
|---|---|
| node | `smci355-ccs-aus-n01-33` (all arms in this packup) |
| GPUs | 8 × AMD Instinct MI355X, **gfx950**, 288 GB HBM3E each |
| driver | amdgpu **6.14.14** |
| kernel | 6.8.0-107-generic |
| CPU / RAM | 256 threads / 3023 GB |
| fabric | 8 × `ionic` RDMA rails @ 400 Gb/s, all `PORT_ACTIVE`; `ib_peer_mem` loaded |
| data-plane NIC | `fenic` (10.235.192.x) — the interface the engine and router bind |

RDMA is **not exercised by any arm in this packup** — MIX is single-worker and
aggregated. It is recorded because the same node and image serve the PD packup.

## Software

| | |
|---|---|
| image | `infera/engine-sglang:v0518-glm53` |
| image ID | `sha256:7489a5a0eb6178a58b084a0656d6b6c28351ae022e6c3ba445c27638c87d1b00` |
| base | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` |
| sglang | **0.5.18** |
| torch | **2.9.1+rocm7.2.0.git7e1940d4** |
| built from | `deploy/docker/Dockerfile.sglang` (DSA patch set ON) |
| repo | `AMD-AGI/Infera`, branch `yihou.dev.glm53.expr`, commit `46e79746` |
| etcd | `quay.io/coreos/etcd:v3.5.14` |

**The image digest above is from a 2026-09-03 rebuild on `n01-21`.** The arms in
this packup ran on `n01-33` against an earlier build of the same Dockerfile. The
two were verified equivalent structurally (same sglang 0.5.18, same mooncake tree,
12 DSA patch anchors present) and behaviourally (PD conc 1/8 within 1.01×/1.05×).
**Say "rebuilt image, verified equivalent" rather than implying one binary.**

## Model

| | |
|---|---|
| checkpoint | `GLM-5.3-MXFP4` |
| absolute path | `/perf_apps/data/models/GLM-5.3-MXFP4` |
| size | 408 GB, 282 safetensors shards |
| `model_type` / arch | `glm_moe_dsa` / `GlmMoeDsaForCausalLM` |
| quantization | Quark MXFP4 E2M1, per-group 32, E8M0 scales |
| layers / nextn | 78 / 1 |

**Path trap:** `/apps/data/models` is a **symlink onto a separate NFS mount**
(`/perf_apps`). Bind-mounting the symlink's parent gives the container an empty
directory of dangling links, and the failure surfaces minutes later as
`Unrecognized processing class` — which reads as a model-version problem. Always
bind the realpath: `MODEL_MOUNT=/perf_apps/data/models`.

The fp8 sibling `GLM-5.3` (704 GB, 141 shards) is smoke-only here; see
`results/bigfp8_smoke.log`.

## Baseline this is compared against

GLM-5.2-MXFP4 MIX fixlen packup, ran 2026-08-06 on node `chi2835` — a
**different cluster**. Path:
`~/dev/git.16-19/infera.glm52.mix.experiment/fixlen.glm52.mix.packup_20260806/`

Its deployment, read from its own log rather than its recipe defaults:
`tp_size=8`, `dp_size=8`, `enable_dp_attention=True`, EAGLE MTP 3/1/4 (n=37,878
accept-len lines), kvd on (8 adapters, `hicache_size=32`,
`hicache_storage_backend='dynamic'`), `max_running_requests=256`,
`mem_fraction_static=0.8`, `context_length=262144`.

## Secrets required

None are stored here. A reproducer needs: SSH access to the node, and a docker
registry login **only** if the image must be pulled rather than built locally.

## Environment capture gaps

- No `collect_env.sh` snapshot was taken at run time; the table above was
  reconstructed on 2026-09-03 from the same node class and the image.
- ROCm userspace version inside the image was not captured beyond the torch
  build string.

## Archive sizing — a compression estimate that was wrong by 3×

Per-request JSONL compresses at **~2.75×**, not the 5-10× that log-file intuition
suggests: it is high-entropy generated text and float arrays, not repetitive log
lines. The 44 MB TP4 control JSONL became **16 MB**, where an estimate from log
behaviour predicted ~5 MB. By contrast the 13 MB engine crash log compressed
**30×**, to 436 KB.

**Size is also a terrible proxy for value.** The smallest artifact in either
packup, `deadknob_evidence.tar.gz` at **30 KB**, is the *only* support for a
retraction that corrects both a commit message and a shipped README — and it is
not reproducible, because the container it came from is gone. It was the item
most at risk of being reasoned away as "small, probably re-runnable".
