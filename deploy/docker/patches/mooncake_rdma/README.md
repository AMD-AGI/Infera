# Optional Mooncake RDMA overlays

These patches make destination-local GPU WRITE routing, deterministic destination
HCA pinning, and configurable RC ACK timeout available in Infera without waiting
for Mooncake upstream PRs. They apply to Mooncake
`faae8dd4a6309c3ecd47e0721a83b0250d686fa2`; a different source revision fails the
bundled-patch check. Upstream submissions are maintained separately against main:
[destination-local/pinning #4540](https://github.com/kvcache-ai/Mooncake/pull/4540)
and [ACK timeout #4541](https://github.com/kvcache-ai/Mooncake/pull/4541).
The build uses the local patches below, not mutable PR downloads, and does not
depend on either PR being merged.

## Build and enable

Build from the Infera repository root:

```bash
docker build -f deploy/docker/Dockerfile.sglang \
  --build-arg APPLY_MOONCAKE_RDMA_PATCHES=1 \
  -t infera-sglang:mooncake-rdma .
```

The same argument is supported by `Dockerfile.sglang.gfx942` and
`Dockerfile.sglang.glm53`. These normal images default to no RDMA overlays.
`BUILD_MOONCAKE=0` cannot be combined with enabling the patches.

To rebuild only Mooncake on an existing compatible ROCm SGLang/Infera image:

```bash
docker build -f deploy/docker/Dockerfile.sglang.mooncake \
  --build-arg BASE_IMAGE=infera-sglang:your-existing-tag \
  --build-arg MC_GPU_ARCH=gfx950 \
  -t infera-sglang:mooncake-rdma .
```

This dedicated overlay enables the **build-time** patch bundle by default. Use
`gfx942` for MI300/MI325. The base must contain the ROCm compiler/development
libraries and Python 3.10 Mooncake layout used by `build_mooncake_sglang.sh`.
It preserves the base image's other engine code, entrypoint, and command.
It does not install Infera into a stock SGLang base or patch vLLM/ATOM images.

At engine launch, pass these environment variables to both P and D containers:

```bash
-e MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
-e MC_ENABLE_DEST_LOCAL_RAIL=1 \
-e MC_IB_TIMEOUT=18
```

For example, add these `docker run` options to the existing P/D launch commands;
they are environment variables, not SGLang CLI arguments. Rebuilding only the
Infera Python package or setting these variables in an unpatched image is not
sufficient. Both container images should use the same patched Mooncake build.

The patches themselves retain the original runtime defaults: destination-local
routing off and ACK timeout exponent 14 (about 67 ms). Set
`MC_ENABLE_DEST_LOCAL_RAIL=0` to disable the new selection and unset
`MC_IB_TIMEOUT` to restore the default timeout. The existing upstream
`MC_ENABLE_DEST_DEVICE_AFFINITY` flag is presence-based; unset it to disable it.

## Routing contract and limits

- Only initial classic-RDMA WRITEs to GPU locations with preferred HCA metadata
  receive a destination-local hint. READ, host memory and unknown/ambiguous
  locations retain normal selection. Retry/failover remains available.
- Each GPU gets one preferred HCA. GPUs with the same preferred HCA set are
  assigned across it in GPU ordinal order; an existing single-HCA preference is
  preserved. Disabled HCAs are removed from the selection. GPU and NIC ordinal
  equality is not hardcoded, and this does not discover hidden physical PCIe links.
- Destination affinity uses matching NIC names on the two hosts. Enable this
  only where those names describe compatible rails. Missing local names fall
  back to normal selection; this is not a strict rail-isolation guarantee.
- A source GPU may read across NUMA to keep the destination write local. Pinning
  bounds receiver fan-in to one NIC per destination GPU and can reduce aggregate
  bandwidth; multiple GPUs may share a NIC when there are fewer NICs than GPUs.
  P and D rank numbers do not need to match. Router session affinity is unrelated.
- `MC_IB_TIMEOUT` accepts finite integer exponents 1..31, with timeout
  `4.096 microseconds * 2^n`. It changes QP ACK timeout, not the transfer deadline,
  RNR retry interval or retry count. Larger values delay loss/stall recovery.
  The experimental deployment used 18 (about 1.07 s); it is not a universal default.
- These are classic Transfer Engine patches, not TENT patches.

## Caller-supplied overlays and provenance

`MOONCAKE_PATCHES` accepts a single whitespace-separated line of **absolute,
whitespace-free filenames**, applied in the supplied order after the bundled
patches. No shell evaluation or wildcard expansion occurs. A derived Dockerfile
can copy extra patches into its image and set this variable for the build script.
Keep the script/helper/patch directory layout when copying the build machinery:

```dockerfile
COPY deploy/docker/scripts/build_mooncake_sglang.sh deploy/docker/scripts/apply_mooncake_overlays.sh /tmp/mooncake-build/scripts/
COPY deploy/docker/patches/mooncake_rdma /tmp/mooncake-build/patches/mooncake_rdma
COPY my-fix.patch /tmp/my-fix.patch
RUN APPLY_MOONCAKE_RDMA_PATCHES=1 MOONCAKE_PATCHES=/tmp/my-fix.patch \
    bash /tmp/mooncake-build/scripts/build_mooncake_sglang.sh
```

Every patch is checked and applied with `git apply`. Missing files, source drift,
conflicts or an invalid switch stop the build. The script starts from a clean
checkout; reapplying the bundle to an already patched checkout is not supported.
With no bundle and an empty custom list, no source patches are applied.

The original HIP dma-buf, cross-host routing and large-MR capability checks run
against the patched source and installed `engine.so`. Bundle builds also require
both new environment keys in that installed library. The source SHA and ordered
patch SHA256 list are saved as `infera-build.txt` alongside `mooncake.engine`:

```bash
python3 -c 'import pathlib, mooncake.engine; print(pathlib.Path(mooncake.engine.__file__).with_name("infera-build.txt").read_text())'
```

## Validation and upstream lifecycle

The development patch combination was tested on MI355X/ionic: 64 P-to-D GPU pairs
passed data checks, and a C80 run had zero transfer failures versus 51 in its
baseline. Residual ACK timeouts remained. Those measurements validate the
historical topology-specific combination, not every topology or this extraction.
The extraction adds deterministic ordering, strict timeout parsing, default/fallback
coverage, and CPU tests; a fresh ROCm image and multi-node GPU run remain required
before claiming new end-to-end performance validation.

When the upstream changes land, advance the pinned revision and retire the
corresponding bundled patch in the same change. Keep the runtime defaults and
regressions. Do not silently skip an arbitrary failed patch or fetch mutable PR
heads as a production build dependency.
