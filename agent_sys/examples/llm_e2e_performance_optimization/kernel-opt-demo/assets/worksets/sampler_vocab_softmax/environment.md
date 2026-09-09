# Environment — where this workset's numbers were measured

Measured 2026-08-31. Everything below was read off the machine, not assumed.
Anything not captured is listed at the bottom rather than left out.

## Hardware — `smc300x-ccs-aus-a17-31`

| | |
|---|---|
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.5.0-45-generic |
| CPU | 2× AMD EPYC 9554 64-Core (256 logical, 2 threads/core) |
| RAM | 1.5 TiB |
| GPU | 8× AMD Instinct MI300X OAM, **`gfx942`**, SKU M3000100, 192 GB HBM3 each |
| GPU partitioning | NPS1 / SPX — one partition, whole card |
| GPU driver | amdgpu 7.1.3.31500000 |
| CUs per card | 304 |

The measurements use **one** card, selected with `HIP_VISIBLE_DEVICES`. Nothing
here needs more than one, and nothing here uses the RDMA fabric.

**This is a shared host.** Other people's containers are running on it. The
numbers below were taken while the other cards showed 0% utilisation, but no
exclusivity was enforced and no clock was locked — see the honesty section.

## Container

| | |
|---|---|
| Image | `lmsysorg/sglang:v0.5.14-rocm720-mi30x` |
| Digest | `sha256:dab8984486941438b3002734b38d1566e47f4ef4fbc20890d8122dcc83b49928` |
| Image created | 2026-06-26T02:39:14Z |
| Python | 3.10.12 |
| PyTorch | `2.9.1+rocm7.2.0.git7e1940d4` |
| Triton | 3.6.0 |
| ROCm | 7.2.0 (`/opt/rocm/.info/version`) |
| rocprofv3 | `/opt/rocm/bin/rocprofv3` (ships with ROCm) |
| rocprof-compute | 3.4.0 (release) |
| sglang | v0.5.14, commit `49e384ce9d304648e9959666ecb8ce8cd98d0deb`, at `/sgl-workspace/sglang/python/sglang` |

Driver 7.1.3 against container ROCm 7.2.0 is a normal forward-compatible
combination; no problem was observed from it.

## Changed inside the container, beyond the image

Two, and both matter.

```bash
pip install -e ".[dev]"                                            # KernelForge itself
pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt
```

The second one exists because KernelForge pins `astunparse==1.6.3` /
`kaleido==1.3.0` while ROCm 7.2's rocprofiler-compute 3.4.0 needs `1.6.2` /
`0.2.1`. Both cannot hold. **ROCm's side wins**, pip prints a conflict warning
that can be ignored, and `kernel-agents status` goes from
`rocprof-compute: dependencies are not ready` to `ready`.

Skipping it does **not** produce an error. Hardware profiling silently drops to
a lightweight PMC path and the analysis quality falls with it. That one status
line is the only place it shows.

## Container start line

Note the `-v` of the scratch directory: a container has its own `/tmp`, so a
local-disk scratch path is invisible unless it is mounted explicitly.

```bash
docker run -d --name <yours> \
  -v "$HOME:$HOME" -v "$SCRATCH:$SCRATCH" \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --network=host --shm-size 64g --security-opt seccomp=unconfined \
  -e HOME="$HOME" -w "$REPO" \
  lmsysorg/sglang:v0.5.14-rocm720-mi30x sleep infinity
```

## Environment variables that change the result

`$HOME` here is NFS with `root_squash`, and the container runs as root, so root
maps to nobody and every `~/.cache` write fails. **Two of the three observed
failures were silent.** Every cache therefore points at local disk:

```bash
export TRITON_CACHE_DIR=$SCRATCH/triton_cache     # else Triton JIT cache write fails
export KNOWLEDGE_LOCAL_ROOT=$SCRATCH/knowledge    # else the experience KB fails, quietly
export CLAUDE_CONFIG_DIR=$SCRATCH/claude_cfg      # must also be *clean*, see below
export HIP_VISIBLE_DEVICES=<one card>
export PYTHONUNBUFFERED=1
export IS_SANDBOX=1                               # the claude CLI, running as root
```

`CLAUDE_CONFIG_DIR` must be a directory with no plugins or MCP servers in it.
With the operator's real `~/.claude`, `claude --print --output-format json`
returns a **message array** instead of a single object, and KernelForge's
backend probe crashes on `payload.get(...)`
(`src/forge_llm/agent_backends/claude.py:333`).

## Not captured — stated rather than omitted

- **No GPU clock lock and no exclusive reservation.** Other containers were
  present and idle (`rocm-smi`: 0% util, 2–3% vram). Baseline spread of 0.3%
  across five rounds says the interference was negligible on the day; it does
  not say it is bounded.
- **Per-run GPU clock state was not recorded.**
- Optimized kernels measured on this machine showed ~8% round-to-round spread
  against the baseline's 0.3%. The cause was not identified. Anything compared
  against this baseline must be measured several times and reported as a median;
  a single measurement produced a wrong conclusion once already.
