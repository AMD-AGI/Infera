# Image transfer + verification — `infera-sglang:v0519-yihou-0917-nextnfix-hicache`

Task: get the run image from `crsuse2-m2m-137` onto `crsuse2-m2m-276` and prove
it is correct on 276.

## Result: PASS (all 4 checks)

| check | evidence | verdict |
|---|---|---|
| 1. content equality | RootFS layer DiffIDs byte-identical on both nodes | **PASS** |
| 2. sglang version | `0.5.19.dev20260917+ga9fb1c3238` | **PASS** |
| 3. PR #37152 markers | `.cuh` grep + module import + mha.py hunk, all present | **PASS** |
| 4. NextN fusion fix | `fused_shared_experts_architecture` marker present | **PASS** |

Transfer elapsed: **START 06:18:35Z → DONE 06:24:12Z = ~5m37s**, EXIT_CODE=0.

## Route chosen: direct pull (`docker save` on 137 | `docker load` on 276), and why

Evidence gathered before choosing:

- **276 CAN ssh OUT to 137 as `yihou`** — tested `spur exec 165913 bash -c 'ssh
  -o BatchMode=yes crsuse2-m2m-137 echo ...'` → `SSH_OUT_OK`. This makes a direct
  pull possible.
- **NFS staging rejected**: `$HOME` is 93% full (795 GB free on a shared 10 TB
  volume) and `/shared_nfs` is 100% full (1.9 TB free on 410 TB). Writing a 70 GB
  tar there is tight and slow. The direct pull materialises **no** tar anywhere.
- **276 docker root** (`/mnt/m2m_nobackup/docker`) has 26 TB free — ample.
- **No gzip**: image layers are already compressed; measured stream throughput was
  ~115 MB/s (df-used delta over 90 s), CPU-bound gzip would only slow a fast local
  10.245.x link. The prior 135→137/136 transfer also streamed raw. Raw wins.

The pull was launched **decoupled** on 276 via `setsid` (survives `spur exec`
return), logging to NFS `$HOME` so the login node can poll it.

### Exact commands

Source id (137, read-only):
```
ssh -o BatchMode=yes crsuse2-m2m-137 \
  "docker image inspect infera-sglang:v0519-yihou-0917-nextnfix-hicache --format '{{.Id}} {{.Size}}'"
# sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35  66252559778
```

Transfer (launched on 276 through `spur exec 165913`):
```
setsid bash -c '
  set -o pipefail
  ssh -o BatchMode=yes crsuse2-m2m-137 "docker save infera-sglang:v0519-yihou-0917-nextnfix-hicache" \
    | docker load
' > /home/yihou/yihou-img-xfer-276.log 2>&1 < /dev/null &
# log: START 06:18:35Z / Loaded image: infera-... / EXIT_CODE=0 DONE 06:24:12Z
```

## Check 1 — content equality (top-level image ID intentionally NOT used)

Raw `docker image inspect --format {{.Id}} {{.Size}}`:

| node | image ID | .Size |
|---|---|---|
| 137 source | `sha256:fd7220a57b7d…` | 66,252,559,778 |
| 276 loaded | `sha256:291f148aab0a…` | 66,713,112,973 |

The top-level ID and `.Size` **differ**, but this is a store-backend artifact, not
corruption:

- **137** = `Storage Driver: overlay2` (legacy graphdriver image store; docker
  29.6.1). Top-level image ID = **config digest**.
- **276** = `Storage Driver: overlayfs`, `driver-type:
  io.containerd.snapshotter.v1` (**containerd image store**; docker 29.6.2).
  Top-level image ID = **manifest digest**, and `.Size` is accounted differently.

So the two IDs *cannot* be equal even for identical content — the legacy vs
containerd stores name images by different digests. The correct content check is
the **RootFS layer DiffIDs**, which both stores compute the same way:

```
docker image inspect ... --format '{{range .RootFS.Layers}}{{println .}}{{end}}' | sha256sum
# 137: 82048ee56ced25f577dcd1dcd78400d53a0507b2fd5c748d229a5317870887cd
# 276: 82048ee56ced25f577dcd1dcd78400d53a0507b2fd5c748d229a5317870887cd
```

**Byte-identical.** The image's filesystem content on 276 equals 137's. PASS.

## Checks 2–4 — in-container markers (GPUs present, per the fp8 import trap)

GPUs on 276 were at idle baseline (~298 MB / GPU) at check time. Container:
```
spur exec 165913 bash -c 'docker run --rm \
  --device /dev/kfd --device /dev/dri --group-add video --group-add render \
  --entrypoint bash infera-sglang:v0519-yihou-0917-nextnfix-hicache -lc "<checks>"'
```
Source root inside container: `/sgl-workspace/sglang/python/sglang`.

Raw output:
```
=== version ===            0.5.19.dev20260917+ga9fb1c3238
=== cuh kCopyGroupThreads ===   6      (grep -c, no import possible for .cuh)
=== cuh pick_group_bytes ===    2
=== mha.py hunk ===             2      (grep -c 'can_use_jit = (_is_cuda or _is_hip)')
=== nextn fusion ===            1      (grep -c 'fused_shared_experts_architecture = "GlmMoeDsaForCausalLMNextN"')
```

`_tiles_across_lanes` verified **by import**, not grep (module `__pycache__` mtime
trap avoided):
```
MODULE_FILE /sgl-workspace/sglang/python/sglang/kernels/ops/kvcache/hicache.py
HAS_ATTR True
GROUP_BYTES (128, 64, 32, 16)      # HIP-widened copy-round tuple from PR#37152
COPY_GROUP_THREADS 32
tiles_576_u4 True                  # MLA 576 B fp8 row now tiles (legacy path rejected non-128-multiples)
tiles_512_u4 True
```

- **Check 2 (version):** `0.5.19.dev20260917+ga9fb1c3238`. PASS.
- **Check 3 (PR #37152):** all three hunks present — `.cuh` `kCopyGroupThreads`
  (6) + `pick_group_bytes` (2); `hicache.py` `_tiles_across_lanes` importable with
  the widened `GROUP_BYTES` and a live functional result (`_tiles_across_lanes(576,
  4) == True`); `mha.py` `can_use_jit = (_is_cuda or _is_hip)` (2). PASS.
- **Check 4 (NextN fusion):** `fused_shared_experts_architecture =
  "GlmMoeDsaForCausalLMNextN"` in `srt/models/glm4_moe.py` (1). PASS.

## Notes

- No image/container on 137 was stopped, removed, or pruned (read-only there).
- 276's other containers untouched; verification used `--rm` throwaway containers.
- Temp log named `yihou-img-xfer-276.log` (removable).
