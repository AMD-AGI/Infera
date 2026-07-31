# Overlay payload — one image, any stock vendor base

`infera-overlay` carries the base-agnostic half of an engine deployment and is
mounted over an **unmodified** vendor image. No rebuild when upstream bumps, no
forking a base for one model.

```
/payload/
  py310/            cp310 deps + infera + kvd    (SGLang bases)
  py312/            cp312 deps + infera + kvd    (vLLM bases)
  bin/infera-router the Rust data plane (links only libc/libstdc++)
  bin/infera-exec   picks the tree matching the container's Python, then execs
```

## Why this split

| Component | Bound to | Overlay? |
|---|---|---|
| infera, kvd | nothing — pure Python | yes |
| infera-router | libc/libstdc++ only | yes |
| libionic | the host kernel, not the image | yes (already injected at start) |
| aiter kernels | JIT-compiled per arch at runtime | yes (share `~/.aiter/build`) |
| **Mooncake `engine.so`** | CPython minor **and** `libamdhip64.so.7` + 6 system libs | **no — stays in the engine image** |
| **hipFile** | the ROCm version | **no** |

Four dependencies are C extensions (msgspec, xxhash, pyzmq, msgpack), so the
Python trees are per-CPython-minor. The vendor bases have split — vLLM ships
3.12, SGLang ships 3.10 — so the image carries both and `infera-exec` selects.

## Build

```bash
docker build -f deploy/overlay/Dockerfile.payload -t infera-overlay:latest .
```

## Use

```bash
docker volume create infera_payload
docker run --rm -v infera_payload:/out infera-overlay:latest

docker run -v infera_payload:/payload:ro \
  --entrypoint /payload/bin/infera-exec \
  <ANY vendor image> \
  python3 -m infera.engine.vllm --model ... --served-model-name ...
```

In Kubernetes this is an initContainer that copies `/payload` into an `emptyDir`
the engine container mounts, with the same `infera-exec` entrypoint.

## Verified

On MI355X, against **unmodified** vendor images:

- `vllm/vllm-openai-rocm:kimi-k3` (Python 3.12) — engine ready in 70 s and
  serving `/v1/chat/completions`; `infera` resolves to `/payload/py312/infera`.
- `rocm/infera:sglang-v0.1.1` (Python 3.10) — `infera` resolves to
  `/payload/py310/infera`, C-extension deps import, `infera.engine.sglang` and
  `infera.kvd` both present.
- `infera-router` runs from the same payload on both.

## One trap this handles for you

Python puts the working directory at `sys.path[0]`, **ahead of `PYTHONPATH`**.
Vendor images that already ship infera set `WORKDIR` to its parent
(`/opt/infera`), so the baked-in copy silently shadows the payload and the
overlay appears to do nothing. `infera-exec` therefore `cd`s to `/` before
exec'ing; override with `INFERA_EXEC_CHDIR` if a command needs its own cwd.
