# Overlay payload — one image, any stock vendor base

`infera-overlay` carries **everything we add** to a vendor image, and is mounted
over an **unmodified** base. Changing base costs nothing: no rebuild, no fork,
no per-model image.

```
/payload/
  py310/                  cp310 deps + infera + kvd        (SGLang bases)
  py312/                  cp312 deps + infera + kvd        (vLLM bases)
  native/rocm7-py312/     mooncake + hipFile + their libs
  bin/infera-router       the Rust data plane
  bin/infera-exec         picks the trees matching this container, then execs
```

## Why everything can be overlaid

The engine images exist because some of what we ship is compiled. But none of it
binds the *specific vendor image* — only coarse, stable interfaces:

| Component | Actually binds | Overlay key |
|---|---|---|
| infera, kvd | nothing — pure Python | — |
| infera-router | `libc`/`libstdc++` only | — |
| infera deps (msgspec, xxhash, pyzmq, msgpack) | CPython minor | `py3XX` |
| **Mooncake** | CPython minor + ROCm major + ~40 system libs | `rocmN-py3XX` |
| **hipFile** | ROCm major | `rocmN-py3XX` |
| libionic | the host kernel, not the image | injected at start |
| aiter kernels | GPU arch, JIT-compiled at run time | cache `~/.aiter/build` |

Mooncake's extension needs ~40 libraries (glog, jsoncpp, asio, gflags, …) a
stock base does not ship, so the payload bundles them next to the `.so` and
points `LD_LIBRARY_PATH` at the bundle — the same thing `auditwheel` does when
it vendors deps into a manylinux wheel.

The consequence: a payload is rebuilt only when the **ROCm major** or the
**CPython minor** changes, not when the vendor image moves.

## Build

```bash
docker build -f deploy/overlay/Dockerfile.payload -t infera-overlay:latest .
```

`NATIVE_IMAGE` (default `rocm/infera:vllm-v0.1.1`) supplies the already-compiled
Mooncake/hipFile/router; they are harvested rather than rebuilt, so the payload
build is fast and reuses binaries that have already been validated.

## Use

```bash
docker volume create infera_payload
docker run --rm -v infera_payload:/out infera-overlay:latest

docker run -v infera_payload:/payload:ro \
  --entrypoint /payload/bin/infera-exec \
  <ANY vendor image> \
  python3 -m infera.engine.vllm --model ... --served-model-name ...
```

In Kubernetes: an initContainer copies `/payload` into an `emptyDir` the engine
container mounts, with the same `infera-exec` entrypoint.

## Verified

On MI355X against an **unmodified** `vllm/vllm-openai-rocm:kimi-k3`:

```
1. infera      : /payload/py312/infera        (kvd + engine.vllm present)
2. router      : Infera router data plane (Rust)
3. mooncake    : TransferEngine OK            -> PD available
4. hipFile     : libhipfile.so loaded
   ais-check   : HIP runtime True, amdgpu True -> kvd GPU-direct L3 available
```

Serving was verified separately end to end on the same stock image: engine ready
in 70 s, `/v1/chat/completions` returning tokens. The cp310 tree was verified on
`rocm/infera:sglang-v0.1.1` (Python 3.10).

## Two traps this handles for you

**cwd beats PYTHONPATH.** Python puts the working directory at `sys.path[0]`,
ahead of `PYTHONPATH`. Vendor images that already ship infera set `WORKDIR` to
its parent, so the baked-in copy silently shadows the payload and the overlay
appears to do nothing — this actually happened on the SGLang image during
testing. `infera-exec` `cd`s to `/` first; override with `INFERA_EXEC_CHDIR`.

**A missing native tree is not fatal, and should not be silent.** Aggregated
serving needs no native tree at all. Only PD (Mooncake) and kvd's GPU-direct L3
(hipFile) do, so `infera-exec` warns rather than failing; set
`INFERA_REQUIRE_NATIVE=1` to make it an error on a PD deployment.
