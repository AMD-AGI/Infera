# build-verification.md — `infera-sglang:v0519-yihou-0917-nextnfix-hicache`

Thin layer applying upstream sglang PR #37152 (HiCache JIT copy kernels on
ROCm/wave64) on top of `infera-sglang:v0519-yihou-0917-nextnfix`. Built locally
and independently on each node — the 66 GB base was already present on both, so
no `docker save`/transfer.

Verdict: **PASS on crsuse2-m2m-135, PASS on crsuse2-m2m-138.**

## Image ids

Both images are locally built and never pushed, so `--digests` reports
`<none>`; the sha256 image id is the identifier. The two nodes have different
ids — expected, since each was built independently on top of an
independently-built base (base ids also differ: `6d6393c4070a` vs
`e17780d2d0fc`). The applied layer is the identical one-line PR patch.

| node | base id | new hicache id (full) |
|---|---|---|
| crsuse2-m2m-135 | `6d6393c4070a` | `sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35` |
| crsuse2-m2m-138 | `e17780d2d0fc` | `sha256:972d8fd952e97b3915d742072874b57a1796c10de1dc0cdcc85e92123167f5d8` |

## Commands run

### Copy + build (per node)

```
# local: bench/.../yihou-agentx-hicache/image/
scp Dockerfile.yihou.hicache pr37152.sources.yihou.diff \
    <node>:/tmp/yihou-hicache-build/
ssh <node> 'cd /tmp/yihou-hicache-build && \
  docker build -t infera-sglang:v0519-yihou-0917-nextnfix-hicache \
               -f Dockerfile.yihou.hicache .'
```

Both builds patched all three targets and exited 0. Build-time output
(identical on both nodes modulo timings):

```
patching file python/sglang/kernels/jit/csrc/kvcacheio/hicache.cuh
patching file python/sglang/kernels/ops/kvcache/hicache.py
patching file python/sglang/srt/mem_cache/pool_host/mha.py
[pr37152] applied and bytecode caches dropped
```

The Dockerfile's own build-time assertions (cuh marker, hicache.py marker,
mha.py gate widened, `__pycache__` dropped) all passed — the build fails loudly
otherwise.

### Runtime verification (per node)

Run in a GPU container (build has no GPU, so the import checks must run here).
`HIP_VISIBLE_DEVICES=2` restricts the container to a workspace GPU (device 2),
keeping the run entirely off 135's root-owned k8s GPU[1].

```
ssh <node> 'docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add video -e HIP_VISIBLE_DEVICES=2 \
  infera-sglang:v0519-yihou-0917-nextnfix-hicache bash -lc "..."'
```

## Four runtime checks — verbatim output

### crsuse2-m2m-135

```
--- CHECK a (kCopyGroupThreads count in cuh) ---
6
--- CHECK b (import _tiles_across_lanes) ---
<function _tiles_across_lanes at 0x7c6939d0e710>
--- CHECK c (mha.py gate) ---
110:        self.can_use_jit = (_is_cuda or _is_hip) and can_use_hicache_jit_kernel(
272:                if self.can_use_jit:
293:                if self.can_use_jit:
433:                if self.can_use_jit:
780:        self.can_use_jit = (_is_cuda or _is_hip) and can_use_hicache_jit_kernel(
851:                if self.can_use_jit:
868:                if self.can_use_jit:
916:                if self.can_use_jit:
935:                if self.can_use_jit:
--- CHECK d (NextN fusion attr) ---
GlmMoeDsaForCausalLMNextN
```

### crsuse2-m2m-138

```
--- CHECK a (kCopyGroupThreads count in cuh) ---
6
--- CHECK b (import _tiles_across_lanes) ---
<function _tiles_across_lanes at 0x7b1a1feea710>
--- CHECK c (mha.py gate) ---
110:        self.can_use_jit = (_is_cuda or _is_hip) and can_use_hicache_jit_kernel(
272:                if self.can_use_jit:
293:                if self.can_use_jit:
433:                if self.can_use_jit:
780:        self.can_use_jit = (_is_cuda or _is_hip) and can_use_hicache_jit_kernel(
851:                if self.can_use_jit:
868:                if self.can_use_jit:
916:                if self.can_use_jit:
935:                if self.can_use_jit:
--- CHECK d (NextN fusion attr) ---
GlmMoeDsaForCausalLMNextN
```

Interpretation:
- **a** — `kCopyGroupThreads` present (6 occurrences, > 0). PASS.
- **b** — `_tiles_across_lanes` imports from `sglang.kernels.ops.kvcache.hicache`
  without raising. Module path matches the diff (`python/sglang/kernels/ops/kvcache/hicache.py`);
  no adjustment needed. PASS.
- **c** — Both class gates (lines 110 and 780) read
  `(_is_cuda or _is_hip) and can_use_hicache_jit_kernel(`. No bare
  `_is_cuda and can_use_hicache_jit_kernel` remains (the build asserted this).
  Line 110 was already widened in the base for the first host-pool class; the PR
  hunk widened the second (line 780). PASS.
- **d** — NextN fusion fix from the base layer survived: prints
  `GlmMoeDsaForCausalLMNextN`. PASS.

## `.orig` backup comparison (7 DSA patches undisturbed)

`find /sgl-workspace/sglang -name '*.orig' | sort` in the base and in the new
image, both nodes — all four listings identical:

```
/sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py.orig
/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py.orig
/sgl-workspace/sglang/python/sglang/srt/managers/schedule_batch.py.orig
/sgl-workspace/sglang/python/sglang/srt/managers/scheduler_components/dp_attn.py.orig
/sgl-workspace/sglang/python/sglang/srt/model_executor/forward_batch_info.py.orig
/sgl-workspace/sglang/python/sglang/srt/speculative/base_spec_worker.py.orig
/sgl-workspace/sglang/python/sglang/srt/speculative/eagle_worker_v2.py.orig
```

7 files, identical set base ↔ new on both nodes. The thin layer did not touch
the DSA patch set.
