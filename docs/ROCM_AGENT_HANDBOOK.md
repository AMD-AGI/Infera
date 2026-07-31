# Working on the AMD ROCm inference stack — field notes

Notes for whoever (or whatever) works on this stack next. Everything here cost
real debugging time; none of it is guessable from the source alone. Hardware is
MI300X (`gfx942`) and MI355X (`gfx950`), ROCm 7.2.

The organising principle: **on this stack, the expensive failures are silent.**
Things do not crash — they degrade, or return plausible numbers that are wrong.
Every section below is a case of that.

---

## 1. Silent failures, ranked by how much time they cost

### RDMA quietly becomes TCP (libionic ABI)

Engine images ship libionic built against an old kernel ABI (e.g. `54.0-149` =
ABI 1). The MI355X ionic driver wants ABI 4 (`54.0-187`). On mismatch
`ibv_get_device_list` returns **0 HCAs**, and Mooncake/MoRI fall back to TCP
without an error. Cross-node PD "works" and is 15-50x slower, or fails later
with `remote mooncake session ... is not alive`.

```bash
# in the pod/container — this is the check that matters
ibv_devinfo | grep -c PORT_ACTIVE     # 0 = broken, 8 = healthy on these nodes
```

Fixes, in order of preference: bake ABI-4 libionic into the image
(`INSTALL_LIBIONIC=1` in `Dockerfile.vllm` / `Dockerfile.sglang`), or mount the
host's `/usr/lib/x86_64-linux-gnu/libionic.so` at `/host-libionic/libionic.so`
and let the `infera-inject-host-ionic` entrypoint replace it at start.

**Note the entrypoint interaction:** overriding the container command (which
Kubernetes `extraPodSpec` does) bypasses the injector. Either prepend
`/usr/local/bin/infera-inject-host-ionic` to the command, or use an image with
libionic baked in.

### Dead GPU processes hold VRAM after their container is gone

`docker rm -f` and `kubectl delete` do not always reap the engine's TP worker
processes. They keep the memory. The next deployment then dies with something
that looks like a config error:

```
ValueError: Free memory on device cuda:4 (126.77/287.98 GiB) on startup is less
than desired GPU memory utilization (0.85, 244.79 GiB)
```

This cost two failed deployments before it was recognised. **Check before every
launch:**

```bash
rocm-smi --showmeminfo vram | grep "Used Memory"      # ~0.3 GB/card = clean
sudo rocm-smi --showpids                              # then kill -9 the survivors
```

### HIPIFY translates warp-32 code into warp-64 code that compiles and lies

CDNA wavefronts are **64** lanes. Any `& 31`, `>> 5`, `% 32`, `(x+31)/32`,
`__shared__ T smem[32]`, or `T acc[N/32]` survives HIPIFY untouched and produces
wrong results with no diagnostic. A `-DWARP_SIZE=64` does not help — most of
these bypass the constant. Budget a manual sweep; it is the dominant cost of any
CUDA-to-HIP kernel port, and the failure mode is bad numbers, not a crash.

### `ais-check` missing downgrades kvd's L3 to CPU bounce

Same shape of problem: `Dockerfile.vllm` asserts `test -x /opt/rocm/bin/ais-check`
precisely so that a regression fails the build instead of silently halving
throughput.

---

## 2. Per-model serving traps

These are model properties, not configuration preferences. Copying flags between
models is how you hit them.

### Kimi-K3 (vLLM, ROCm)

| Flag | Why |
|---|---|
| `--kv-cache-dtype auto` | **Required.** `infera.engine.vllm` injects `fp8_e4m3` by default, but Kimi-K3's MLA then selects the `mla_gluon` kernel, which is batch-1 only, and warmup dies: `mla_gluon[bh16bn128] requires batch_size=1, got 128`. Equivalent: `INFERA_DEFAULT_KV_FP8=0`. |
| **no** `--enable-prefix-caching` | Kimi-K3 is a hybrid **Mamba** model; enabling it forces the experimental Mamba "align" cache and engine init fails. |
| **no** `--load-format fastsafetensors` | Needs GPU Direct Storage (cufile). Without GDS the loader stalls — measured ~30 s queue wait per batch, stuck at 0% for 10 minutes. Use `auto`. |

Loads ~1.5 TB in ~9 min from local NVMe, ~1 h from a shared NFS mount. Stage
weights locally before benchmarking anything.

### GLM-5.2-MXFP4 (`GlmMoeDsaForCausalLM`, MLA + DeepSeek Sparse Attention)

**Use SGLang, not vLLM.** vLLM loads and serves, then decode degrades to `!!!!`
garbage after a few tokens — an MLA/DSA numerical bug. Verified side by side:
vLLM returned `'1!!!!!!!...'` where SGLang returned a correct answer.

SGLang needs ROCm-specific env or it crashes at init trying to JIT a CUDA-only
DSA top-k kernel on gfx950:

```
SGLANG_OPT_USE_TILELANG_INDEXER=1  SGLANG_OPT_USE_TOPK_V2=0  SGLANG_OPT_USE_JIT_NORM=0
SGLANG_USE_AITER=1                 SGLANG_ROCM_FUSED_DECODE_MLA=0
--nsa-prefill-backend tilelang --nsa-decode-backend tilelang --kv-cache-dtype fp8_e4m3
```

Needs SGLang **0.5.15+**; older ROCm images cannot load GLM-5.2 at all (its
`head_dim` changed).

---

## 3. aiter as a C++ operator library

aiter is the only source of paged attention, fused MoE, MLA and KV-cache-write on
ROCm — hipBLASLt and CK cover GEMM and FlashAttention, nothing else covers these.
It **does** have a usable, torch-free C++ API, but the shape of it is not
documented anywhere.

### Three entry layers

| Layer | What | Verdict |
|---|---|---|
| `csrc/include/*.h` + `aiter_tensor_t` | POD struct (`void* ptr`, shape/strides/dtype). The Python pybind layer is a thin shim **on top** of this. | **Primary** |
| C-ABI `extern "C"` | One entry per kernel + TLS error API; some ship as torch-free `.so`s. | For the asm kernels |
| `csrc/cpp_itfs/` | Raw-pointer headers, but shells out to Python + hipcc on first call unless AOT-warmed. Only covers PA and MLA. | Avoid unless it is the only path |

### Rules

- **Compile the kernel TUs from source.** The prebuilt `.so`s under `aiter/jit/`
  list `libtorch`/`libc10` as `DT_NEEDED`. Compiling
  `csrc/kernels/*.cu` yourself yields a binary needing only `libamdhip64`,
  `libstdc++`, `libm`, `libgcc_s`, `libc` — verified with `nm -uC | grep -c
  'c10::\|at::\|torch::'` → 0.
- A few TUs reach `<c10/util/BFloat16.h>` through `aiter_opus_plus.h`. It is
  header-only: you need torch's *include path* at compile time and link nothing.
- ROCm 7.2's rocPRIM `texture_cache_iterator.hpp` uses `memset` without
  `<cstring>` — pass `-include cstring`.
- **No CMake package, no pkg-config.** You own the include paths and the TU list.
- **Pin a commit.** The API is undocumented and actively churning.

### Kernel availability is per-arch, and gfx950 has less than you expect

```
hsa/gfx942/   58 asm blobs, including many pa_*.co
hsa/gfx950/   16 blobs, ZERO pa_*.co
hsa/gfx1250/   9 blobs
```

So on MI355X the assembly paged-attention path does not exist; the HIP-source
path (`cpp_itfs/pa/pa_ragged`, JIT ~18 s then cached, AOT-warmable to a pure
`dlopen`) is the one that works. Check `hsa/<arch>/` before assuming a kernel is
available, and read the arch guards — `pa_decode_bf16_asm` hard-asserts
`gfx1250` and is fp8-only.

### Bugs found and fixed upstream

`csrc/cpp_itfs/pa/pa_ragged.cpp` had its `run_lib()` argument order out of sync
with its own kernel template (a `float` where the kernel expects `int*`, and
`q_scale` omitted), so every C++ call fed the kernel garbage; and line 42 used
`warpSize` in **host** code, which stopped compiling when ROCm 7.2.3 made
`warpSize` a device-side class. Both fixed in ROCm/aiter#4486 — pin a commit that
includes it, or `dlsym` the AOT-built kernel using the template's signature.

The general lesson: **when a dispatcher and its code-generation template
disagree, trust whichever one has test coverage.** Here the Python path bypassed
the C++ dispatcher entirely, so the dispatcher had rotted unnoticed.

---

## 4. Kubernetes on these boxes

### k3s

- **Put `--data-dir` on a large disk.** k3s stores imported images there; the
  default lives on `/`, and a single 30-80 GB engine image trips the kubelet
  DiskPressure threshold, which evicts the operator and the workers. Symptom:
  pods `Evicted`, node tainted `disk-pressure:NoSchedule`.
  ```bash
  curl -sfL https://get.k3s.io | \
    INSTALL_K3S_EXEC="--write-kubeconfig-mode=644 --data-dir /mnt/<big>/k3s" sh -
  ```
- **The node token follows `--data-dir`** — `<data-dir>/server/node-token`, not
  the default path. Joining a second node fails confusingly otherwise.
- **containerd is not docker.** A locally built image must be imported:
  `docker save img -o /tmp/i.tar && sudo k3s ctr images import /tmp/i.tar`, then
  `imagePullPolicy: IfNotPresent`. Stream-piping `docker save | ctr import` was
  unreliable here; write the tar.

### Operator / InferaDeployment

- **Mounting a model requires `extraPodSpec` with an explicit `command`.** The
  operator's simpler `args` mode auto-builds the entrypoint but only mounts
  `dshm` and `/boot`.
- **Use a `startupProbe` for slow loads**, not `skipReadinessProbe`. A generous
  `startupProbe` on `/health` (e.g. `failureThreshold: 120`, `periodSeconds: 10`)
  holds the pod non-Ready through a multi-minute load and then hands over to the
  operator's readiness probe, so you get a real Ready signal.
  `skipReadinessProbe: true` works but never surfaces one.
- **sglang ↔ vllm is not just the image.** The router is engine-agnostic, but the
  worker's entrypoint module *and* flag names differ (`--model-path`/`--tp-size`
  vs `--model`/`--tensor-parallel-size`).

### Cross-node RDMA needs a routable fabric, not just NICs

Two nodes each having 8 active RoCE NICs is not sufficient. Check that they share
a routable subnet:

```bash
# on each node — group-4 field of the GID_Index-1 RoCE v2 GID
for d in 0 1; do cat /sys/class/infiniband/ionic_$d/ports/1/gids/1; done
```

Nodes whose rails sit on isolated `/64`s will reconcile, register, and dispatch —
and then fail every KV transfer. That is a fabric topology fact; no image or
config change fixes it.

---

## 5. How to not waste a day

**Run it. Do not reason about it.** Three wrong conclusions in this session were
caught only by executing:

1. An entry point recommended after reading headers turned out to be
   `gfx1250`-only — the arch guard was 40 lines below the signature.
2. A paged-attention integration that "obviously" matched on paper aborted in the
   kernel, because the shipping dispatcher was buggy.
3. A test reporting 4.3% error looked like a kernel problem; it was the *test's*
   error metric — with random inputs, attention outputs average near zero, so
   per-element relative error explodes. Normalise by output magnitude.

**Verify a test can fail.** After fixing the aiter dispatcher, the fix was
confirmed by reintroducing the bug and watching the new test abort. A test that
has never failed is not evidence.

**Cross-check comments against runtime logic.** `frame_manager.h` documents
`SCATTERED` as the default cache pool mode; `model.cpp:285` selects `CONTIGUOUS`
unless an env var says otherwise. The comment is stale, and believing it would
have led to the wrong port strategy.

**Grep for the guard, not just the symbol.** `#ifdef ENABLE_CUDA` appears 586
times in dash-infer, and the overwhelming majority guard *"a GPU exists"*, not
*"NVIDIA specifically"*. Counting occurrences without reading what they guard
gives a wildly wrong estimate of porting effort.

---

## 6. Where things live

| | |
|---|---|
| k8s recipes (single-node mixed, PD 1P1D, Kimi-K3, GLM-5.2) | `examples/k8s-deployments/`, `manual/examples/k8s_*.md` |
| Kimi-K3 standalone docker serve + multimodal smoke test | `examples/kimi_k3/` |
| AMD port of dash-infer | `kzjeef/dash-infer` branch `dev/amd`, plan in `docs/AMD_ROCM_PORT.md`, tracked in that repo's issue #684 |
| aiter C++ integration PoCs (torch-free op call; paged attention numerics) | `kzjeef/dash-infer` `tools/rocm_poc/` |
