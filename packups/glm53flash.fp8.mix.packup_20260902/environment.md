# Environment

Everything below was read off the host on **2026-09-02**, the day of the run.

## Node — and it is not ours

| | |
|---|---|
| host | `smci355-ccs-aus-n05-29.prov.aus.ccs.cpe.ice.amd.com` |
| IP used | `192.168.3.26` |
| GPUs | 8 × **AMD Instinct MI355X**, model `0x75a3`, gfx950, 288 GiB each (309220868096 B) |
| **GPUs used** | **4, 5, 6, 7 only** — 0-3 were never touched |
| driver | amdgpu **6.14.14** |
| host ROCm | `/opt/rocm-7.0.1` (the container carries its own, 7.2.0) |
| CPU | AMD EPYC 9575F 64-Core, 256 logical |
| RAM | 3023 GiB |
| host uptime at teardown | 12 days — no reboot during this work |

**This node belongs to a colleague and was shared throughout.** Four foreign
containers were up for the whole session and were never touched:
`turbo.jax.zhuang12`, `xiaoming-dev`, `primus.zhuang12.20260826`,
`primus.zhuang12.20260824`. Their GPU footprint oscillated 0 → ~30 GiB per card
on a 7-15 minute cycle (`notes.md` §5). Every number in this packup is tagged
with that contention state.

## Image

| | |
|---|---|
| tag | `infera/engine-sglang:glm53-c821c425` |
| **image id (this node)** | `sha256:dbff9e03ce2788398de8a808e31dbe06597a096930231c0d32a2f6cee6a3b99b` |
| built from | `deploy/docker/Dockerfile.sglang.glm53`, **unmodified** |
| base tag | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` |
| **base digest** | `lmsysorg/sglang@sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72` |
| build log | `logs/build.log.gz` |

Image ids are **per-node** — a build of the same Dockerfile on another host will
have a different id. The base digest and the pinned source SHA below are what
make two such builds equivalent.

### Pinned sources, verified by running the image (not by trusting the build)

```
$ docker run --rm --entrypoint bash infera/engine-sglang:glm53-c821c425 \
    -c 'cd /sgl-workspace/sglang && git rev-parse HEAD && wc -l \
        python/sglang/srt/models/glm5_next.py \
        python/sglang/srt/layers/quantization/quark/quark.py'
c821c425c31b0e6c8151324b60fbc2857c39eaef
  1942 python/sglang/srt/models/glm5_next.py
  1172 python/sglang/srt/layers/quantization/quark/quark.py
```

| pin | value | why this one |
|---|---|---|
| sglang | **`c821c425c31b0e6c8151324b60fbc2857c39eaef`** (PR #36607 head, merged 2026-08-28, frozen) | the content signature 1942/1172 distinguishes it from the `9e692c92` trap at 1834/1103, which lacks AITER mHC and cannot load MXFP4 |
| mooncake | `faae8dd4a6309c3ecd47e0721a83b0250d686fa2` | Dockerfile default; irrelevant to MIX but built in |
| libionic | `54.0-149.g3304be71`, build gate printed `LIBIONIC_ABI ... = 4..4` | passed unaided; the `--build-arg LIBIONIC_REQUIRE_ABI=` escape hatch was **not** needed |
| infera repo | branch `yihou.dev.glm53.expr`, commit **`f6ee2da34f8bab5c6c7229c3c25ed61b7d506eaf`** | the build context |

Note the in-build overlay verifier ran but **skipped its import check**:
`[verify-glm53-overlay] import skipped, builder has no GPU`. Only the static
check (symbol defined and exported) ran at build time; runtime settled the rest.

## External paths (not in the repo)

| path | note |
|---|---|
| `/apps/data/models/GLM-5.3-Flash` | **a symlink.** Resolves to `/perf_apps/data/models/GLM-5.3-Flash` |
| `/perf_apps/data/models/GLM-5.3-Flash` | the real weights: **306 GB, 62 safetensors shards**, `config.json` 69416 B, mtime 2026-08-30 |
| build context | `/home/yihou/dev/git.16-19/infera.glm53.series.integration` (shared NFS, visible from the node) |
| scratch workspace | `/apps/yihou/glm53.series.workspace_20260901/flash-fp8-0529/` — **left intact**; this packup is a copy of it |

**`/apps` and `/perf_apps` are separate NFS mounts.** Bind-mounting
`/apps/data/models` gives an **empty directory inside the container**. The
scripts resolve `readlink -f` at runtime and bind the realpath — see
`scripts/mix_up.sh`.

## Model

```json
model_type       : glm5_next
architectures    : ["Glm5NextForConditionalGeneration"]
num_hidden_layers: 45     hidden_size: 4096     n_routed_experts: 288
quantization_config: {"fmt": "e4m3", "activation_scheme": "dynamic", ...}
```

Because `config.json` already declares e4m3, **no `--quantization` flag is
passed**, and the engine resolves it on its own — confirmed in the log:

```
FlashInfer TRTLLM MoE deferred finalize is disabled (moe_runner_backend=triton, quant_method=Fp8MoEMethod)
```

This is the same auto-detect claim the big-MXFP4 model card makes, checked
independently here on the Flash side.

## Ports (chosen because the defaults were taken on this node)

`2379` and `18100` were **BUSY**. Verified free and used:

| service | port |
|---|---|
| engine | 31400 |
| etcd client / peer | 23795 / 23796 |
| infera router | 18105 |
| kv events pub / snapshot | 15570 / 18801 |

Do not assume these; re-run `ss -lnt` on whatever node you reproduce on.

## Secrets — by name and source, never by value

| need | source |
|---|---|
| SSH to the node | the user's existing `~/.ssh/config`; no key material in this packup |
| container registry | the host's existing `docker login`; the base image was already pulled locally, so **no registry auth was exercised during this run** |
| `repo.radeon.com` (libionic .deb) | anonymous HTTPS, no credential |
| GitHub (sglang PR fetch) | anonymous; `git fetch origin pull/36607/head` needs no token |

No API keys, tokens or passwords are required to reproduce this.

**Secret scan, run before declaring done**, over `*.md *.sh *.py *.txt *.csv`
for `(api[_-]?key|passwd|password|secret|bearer|token)` followed by a 12+ char
value, plus a `BEGIN .* PRIVATE KEY` sweep. **Two hits, both false positives,
recorded rather than silently ignored:** `results/sweep_f8_by_lead.txt` lines
11 and 85 both carry HuggingFace's boilerplate
`...by passing \`token=<your_token>\`` — a literal placeholder emitted by the
library, not a credential. No private-key blocks. Nothing redacted.

## Gaps — captured honestly

- **The packup skill's bundled `collect_env.sh` was not run** (and so is not
  shipped in `scripts/`). The table above was assembled from
  individual commands (`rocm-smi`, `lscpu`, `free`, `docker inspect`). No
  single machine-readable env snapshot file exists for this run, unlike the
  MXFP4 packup's `env_n0133.txt`. Re-running that script on 05-29 today would
  not reproduce the run's state anyway, since the deployment is gone.
- **RDMA fabric was not characterised.** This is single-node MIX; mooncake and
  the NICs are not in the path. `ibv_devices` output was never collected. If
  someone extends this to PD, that gap must be closed first.
- **Round 0's resolved `max_running_requests` is unrecoverable** — see
  `notes.md` §1.
- **The container's own ROCm/torch versions** were not recorded beyond
  `/opt/rocm-7.2.0` appearing in build output. The image digest pins them.
