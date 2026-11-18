(installation)=
# Installation and setup

Install **Infera {{ infera_version }}** (`amd-infera`, imported as `infera`).
Infera orchestrates vLLM, SGLang, or ATOM; a serving deployment needs Infera and
at least one engine. ROCm 10.2 validation is complete for the configuration
recorded in the [compatibility matrix](../features/compatibility_matrix.md#rocm-102-validation).

(system-requirements)=
## Prerequisites

| Component | Requirement |
| --- | --- |
| GPU target | AMD Instinct MI355X (`gfx950`), including the recorded ROCm 10.2 configuration. MI325X / `gfx942` has a separate ROCm 7.2 SGLang recipe; it is not part of the recorded 10.2 validation. |
| ROCm | Select a version below and check the [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) for GPU, operating system, kernel, and driver requirements. |
| Operating system | Linux x86-64; Infera validation uses Ubuntu 24.04. See the Infera matrix for each image's OS version. |
| Python | 3.10+ for the Infera package. Use the engine image's Python and PyTorch versions; the recorded ROCm 10.2 image uses Python 3.14. |
| Docker | Required for the container method and for the quick start's etcd service. Configure access to the allocated AMD GPU devices. |
| Discovery | etcd for development, or the Kubernetes API for production. See the [quick start](quickstart.md). |
| RDMA NIC | AMD AINIC for cross-node prefill-decode. Cross-node operation is outside the recorded ROCm 10.2 validation scope. |
| Engine | At least one of vLLM, SGLang, or ATOM; choose an engine validated for the selected ROCm version. |

The ROCm matrix describes platform support; the
[Infera compatibility matrix](../features/compatibility_matrix.md) describes
the narrower tested combinations and known issues. A GPU supported by ROCm is
not automatically validated for every Infera engine and scenario.

## Select a ROCm version

The selected tab identifies the runtime and engine image to use in the
installation method below. Preview tabs retain their original validation scope.

::::{tab-set}

:::{tab-item} ROCm 10.2
:sync: rocm-102

**Infera {{ infera_version }} — validated configuration.** Use SGLang on
MI355X (`gfx950`) with `deploy/docker/Dockerfile.sglang.rocm102`.
The recorded base is ROCm `10.2.0rc0`, pinned by digest in that Dockerfile.
It is access-gated; obtain authorized access to the base image before building.

### Install ROCm

Use the host driver qualified with the recorded ROCm 10.2 image and the
ROCm 10.2 distribution instructions supplied for your environment. The container
supplies its own ROCm user-space SDK; installing a different host SDK does not
replace it.

```{note}
The public ROCm 10.2 installation guide is not yet available at the expected
version-specific address. Until it is published, this guide documents the
validated container path; it does not provide a bare-metal ROCm 10.2 installation
recipe. Do not substitute the installation instructions of another release.
```

Review the [10.2 validation scope and known defect](../features/compatibility_matrix.md#rocm-102-validation)
before deployment. The recorded SGLang 0.5.21 image must not serve disaggregated
traffic: responses degrade after a long request. Use the aggregated configuration
for this image. vLLM, ATOM, and `gfx942` are outside the recorded 10.2 scope.
:::

:::{tab-item} ROCm 7.2
:sync: rocm-72

The retained baseline covers vLLM, SGLang, and ATOM. Use the exact engine images
in the [compatibility matrix](../features/compatibility_matrix.md#engines):

- MI355X (`gfx950`): `Dockerfile.vllm`, `Dockerfile.sglang`, or `Dockerfile.atom`.
- MI325X (`gfx942`): the separate `Dockerfile.sglang.gfx942` recipe.

### Install ROCm

Follow the [ROCm 7.2 installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.0/install/quick-start.html).
Match the image's ROCm patch version and host driver requirements to the matrix.
:::

:::{tab-item} ROCm 10.1 preview
:sync: rocm-101

This is a retained preview using access-gated nightly bases. Only the
[recorded configurations](../features/compatibility_matrix.md#rocm-101-preview)
are validated. Do not substitute the default ROCm 7.2 engine images.

### Install ROCm

Follow the [ROCm 10.1 installation guide](https://rocm.docs.amd.com/en/docs-10.1.0/install/rocm.html).
Use the digest-pinned preview bases and image adjustments in the compatibility
matrix; the generic build commands below do not select a 10.1 base.
:::

:::{tab-item} ROCm 10.3 preview
:sync: rocm-103

This is a retained alpha-nightly preview on MI355X (`gfx950`). Follow the
[two-stage SGLang image build](../features/compatibility_matrix.md#rocm-103-preview),
which replaces the vendor image's SGLang source while preserving its ROCm SDK.

### Install ROCm

The recorded runtime is ROCm `10.3.0a20261103`, supplied by the pinned preview
container. Use that base and its host requirements. A public release installation
guide is not provided for this alpha configuration; do not replace the pinned
SDK with another release's packages.
:::

::::

(docker)=
(install-infera)=
(install-an-engine)=
## Select an installation method

::::{tab-set}

:::{tab-item} Docker
:sync: install-docker

Engine containers bundle Infera, the engine connector, the `sitecustomize` hook,
and Mooncake / ionic RDMA shims on the vendor base. Build from the release source:

```bash
git clone --branch v0.3.3 --depth 1 https://github.com/AMD-AGI/Infera.git
cd Infera
git describe --tags --exact-match  # v0.3.3
```

Choose **one** build matching the ROCm tab above.

**ROCm 10.2 — SGLang on MI355X:**

```bash
docker build -f deploy/docker/Dockerfile.sglang.rocm102 \
  -t infera-sglang:0.3.3-rocm10.2 .
```

**ROCm 7.2 — choose an engine:**

```bash
# vLLM on MI355X
docker build -f deploy/docker/Dockerfile.vllm -t infera-vllm:0.3.3-rocm7.2 .

# SGLang on MI355X
docker build -f deploy/docker/Dockerfile.sglang -t infera-sglang:0.3.3-rocm7.2 .

# ATOM on MI355X
docker build -f deploy/docker/Dockerfile.atom -t infera-atom:0.3.3-rocm7.2 .
```

For MI325X use `deploy/docker/Dockerfile.sglang.gfx942`. For the 10.1 and 10.3
previews, use their version-specific instructions above.

Continue with [Deployment](../serving/deployment.md) to launch etcd, the server,
and engine workers using your built image. The image build alone does not start
a service. Observe the selected version's scenario restrictions.
:::

:::{tab-item} Python package
:sync: install-pip

Use this method in an environment that already contains a compatible engine and
ROCm runtime. It installs Infera; it does not install ROCm or an engine. The
recorded ROCm 10.2 validation uses the Docker configuration, so a new bare-metal
engine combination is outside that validation scope.

```bash
python -m pip install "amd-infera==0.3.3"
```

Keep the engine's existing Python and PyTorch environment. Choose the engine
version and image from the [compatibility matrix](../features/compatibility_matrix.md).
Do not install a generic engine wheel over a working ROCm engine environment.
:::

::::

(verify)=
## Verify the installation

Run in the environment where you installed Infera, or inside the built engine
container:

```bash
python -c "from importlib.metadata import version; print(version('amd-infera'))"
python -m infera.server --help
```

For a Python package installation, the version should be `0.3.3`. Source-built
containers can report the packaging fallback `0.0.0` because the build excludes
Git metadata; verify the `v0.3.3` source tag before building, as shown above,
rather than relying on the local image tag as proof of the package version.
These commands check package availability and CLI imports; the
[quick start](quickstart.md) launches a stack and sends a model request.
For Docker, run checks in the image you built.

The module entry points used throughout this manual are:

| Command | Purpose |
| --- | --- |
| `python -m infera.server` | OpenAI-compatible server and router |
| `python -m infera.engine.{vllm,sglang,atom}` | Model worker; select one engine |
| `python -m infera.kvd` | Optional tiered KV-cache daemon |

See [Release notes](../about/release-notes.md) for documentation changes in
Infera {{ infera_version }}.
