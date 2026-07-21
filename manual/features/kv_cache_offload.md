# KV-Cache Offload

```{admonition} One-pager
:class: tip
**What:** offload KV blocks below the engine's GPU cache — to host RAM, local
disk, or a distributed store — through the `infera.kvd` daemon, and read them
back on a hit. **Why:** warm prefixes survive engine restarts and are reused
across engines, so you don't recompute them. **How it works under the hood:** see
[KV-Cache Management](../components/kvd.md).
```

## Turn it on — single host

```{admonition} vLLM only (for now)
:class: important
KV-Cache Offload currently supports the **vLLM** engine
(`infera.engine.vllm` + `InferaKvdConnector`). SGLang is not supported yet.
```

One `infera.kvd` daemon per host; every engine on that host shares it.

**1. Start the daemon.** RAM tier (`--max-bytes`) + a local disk long tier
(`--long-path`/`--long-bytes`). Size the RAM tier small enough that the hot set
spills to disk if you want to exercise L3:

```bash
python -m infera.kvd \
  --socket /var/run/infera-kvd.sock \
  --max-bytes 16G \
  --long-path /nvme/kvd-long --long-bytes 256G
```

**2. Launch the engine.** vLLM does **not** add the connector on its own — set
`INFERA_KVD_SOCKET` (the daemon path) **and** pass `--kv-transfer-config`. Set
`PYTHONHASHSEED=0` so vLLM's block hashes — and therefore the kvd chunk keys —
are stable across restarts (otherwise a restart orphans the whole L3 cache):

```bash
PYTHONHASHSEED=0 \
INFERA_KVD_SOCKET=/var/run/infera-kvd.sock \
python -m infera.engine.vllm --model <model> --port 8000 --host 0.0.0.0 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"InferaKvdConnector","kv_role":"kv_both","kv_connector_module_path":"infera.engine.vllm.kvd_connector"}'
```

That runs the POSIX read path (daemon `mmap`/socket). For **GPU-direct** L3, the
connector writes chunk files directly under the roots and reads them back via
hipFile/AIS, bypassing the daemon. Add:

```bash
  INFERA_KVD_AIS=1 \
  INFERA_KVD_HIPFILE_ROOTS=long=/nvme/kvd-long
```

:::{admonition} Run `ais-check` on the host, never inside the container
:class: warning
In a container there is no `dkms` binary, so `ais-check` silently falls back from
reading the authoritative KFD capability to grepping `/boot/config-*`, and reports
`Kernel P2PDMA support: True` on a host whose driver reports **False**. Same machine,
opposite answers. The line that actually matters is **`amdgpu:`** — it agrees in both
places. Do not mount `/boot` into the container to "fix" the check: that only feeds
the weak fallback and manufactures a false positive.

Note also that `capability & 0x40` in the KFD topology is **not** a reliable AIS signal
on these driver builds — a node logging `AIS: registered` per GPU still reports the bit
clear. Trust `ais-check`'s `amdgpu:` line and the `dmesg` `AIS:` messages.
:::

:::{admonition} GPU-direct needs a new-enough amdgpu driver — and it fails *silently*
:class: warning
`INFERA_KVD_AIS=1` is a **request**, not a guarantee. If the driver can't do AIS the
connector logs one line and quietly serves L3 over POSIX (mmap+H2D) instead. You get
correct results at a fraction of the speed, with no error.

Requirements (per [hipFile's INSTALL.md](https://github.com/ROCm/rocm-systems/blob/develop/projects/hipfile/INSTALL.md)):

| | Minimum | Notes |
|---|---|---|
| **amdgpu driver** | **≥ 6.16.6** (`amdgpu-dkms ≥ 30.20.1`) | This is what initializes AIS. Measured: `6.16.6` brings AIS up; `6.14.14` does **not**. |
| **ROCm** | **≥ 7.2** | Both the host and the engine image. A 7.1 host can still initialize AIS in the driver, but reports `HIP runtime : False` — don't rely on it; standardize on 7.2. |
| kernel | `CONFIG_PCI_P2PDMA=y` | |
| storage | local NVMe, `xfs` or `ext4` (`data=ordered`) | Software RAID **does** work despite `ais-check` reporting `HIPFILE=no` for `md` backing — measured 20.8 GB/s cold reload on a 7-drive RAID0, zero fallbacks. That column keys off the backing type, not actual capability; don't let it talk you out of a RAID array. |
| GPU | large-BAR dGPU, not SR-IOV | |

**Check the driver that is *loaded*, not the one installed.** A DKMS upgrade without a
reboot leaves the old module running, and AIS stays off even though the package looks
current — this is easy to miss and can cost a full benchmarking session:

```bash
cat /sys/module/amdgpu/version        # what is actually running
modinfo amdgpu | grep ^version:       # what is installed on disk
# if they differ -> reboot; AIS will not work until you do
```

Then confirm on the **host** (see above):

```bash
/opt/rocm/bin/ais-check               # want: amdgpu : True, and exit code 0
dmesg -T | grep 'AIS:'                # want: "AIS: registered NNNNNMB device memory" per GPU
```
:::

:::{admonition} Leave INFERA_KVD_CHUNK_TOKENS at `auto`
:class: important
GPU-direct's cuFile fan-out only pays off at chunks **≥ 128 MiB**, which `auto`
(the default) sizes to. A small fixed value makes each chunk overhead-bound and
the reload runs *slower than recompute*.
:::

**3. Verify.** Send the same prompt (or a shared long prefix) twice, then check
the engine's stats line for a non-zero **`External prefix cache hit rate`**, and
the daemon counters:

```bash
python -m infera.kvd.statctl --socket /var/run/infera-kvd.sock
# want: sets_total > 0, gets_total > 0, hits_total > 0, misses_total == 0
```

`misses_total > 0` means the scheduler and worker derived different chunk keys —
usually a `PYTHONHASHSEED` mismatch or a `INFERA_KVD_CHUNK_TOKENS` that differs
between runs. Under GPU-direct the daemon counters stay 0 by design (the connector
owns the files); confirm via the engine's `External prefix cache hit rate` and by
`ls`-ing the `.kvcache` files under `INFERA_KVD_HIPFILE_ROOTS` instead.

## Turn it on — prefill/decode disaggregation

Set up the P/D pair, router, etcd, and Mooncake transport as in
[PD disaggregation](pd_disaggregation.md) first. kvd adds an **L3 tier on the
prefill side, alongside** the P→D transport — the decode engine is unchanged (it
only consumes over the transport, no kvd).

**1. Start a kvd daemon on the prefill host** (as in step 1 above).

**2. Prefill engine** — take your working prefill command and (a) add the kvd env,
(b) replace the bare `MooncakeConnector` with a `MultiConnector` wrapping the kvd
tier + the transport. The **top-level `kv_role` must be `kv_producer`** — the
router reads it to emit the Mooncake bootstrap and pair P↔D; a `kv_both`/absent
top role registers the worker as MIXED and P↔D never pairs. The kvd child is
`kv_both` (a cache tier), the transport child is `kv_producer`:

```bash
PYTHONHASHSEED=0 \
INFERA_KVD_SOCKET=/var/run/infera-kvd.sock \
INFERA_KVD_AIS=1 INFERA_KVD_HIPFILE_ROOTS=long=/nvme/kvd-long \
VLLM_HOST_IP=<routable> \
python -m infera.engine.vllm --model <model> --port 8001 --host 0.0.0.0 \
  --advertise-host <routable> --etcd-endpoint <etcd>:2379 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"InferaKvdConnector","kv_role":"kv_both","kv_connector_module_path":"infera.engine.vllm.kvd_connector"},{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}]}}'
```

**3. Decode engine** — exactly the PD-disaggregation decode leg, no kvd:

```bash
PYTHONHASHSEED=0 VLLM_HOST_IP=<routable> \
python -m infera.engine.vllm --model <model> --port 8002 --host 0.0.0.0 \
  --advertise-host <routable> --etcd-endpoint <etcd>:2379 \
  --discovery-backend etcd --request-transport http --kv-event-transport zmq \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

```{admonition} PYTHONHASHSEED=0 on BOTH legs
:class: warning
The prefill and decode engines are separate processes. vLLM seeds its block-hash
chain from `PYTHONHASHSEED`; if it is unset it is **random per process**, so the
two legs (and the L3 keys) compute different hashes for the same prompt and reuse
silently misses. Set `PYTHONHASHSEED=0` on every engine and keep it stable across
restarts.
```

The daemon flags and connector env vars are in the
[CLI reference](../reference/cli.md) and the
[environment variables](../reference/environment.md) reference.

## Usage modes

L2 (host RAM) is on by default; the **long tier is one backend** — local disk
(L3) **or** a distributed store (L4), never both. Pick the tiers you need:

| Mode | Daemon flags / env | Use when |
|---|---|---|
| **RAM only (L2)** | `--max-bytes 16G` | share warm prefixes across engines on one host; no durability |
| **RAM + local disk (L3)** | `+ --long-path … --long-bytes 64G` | survive restarts; hold a large hot set on NVMe/NFS |
| **Distributed (L4)** | `--long-backend mooncake\|lmcache` | share one KV pool across nodes |
| **GPU-direct L3** | `INFERA_KVD_AIS=auto` + `INFERA_KVD_HIPFILE_ROOTS=…` | DMA L3 chunks straight into VRAM (hipFile/AIS), bypassing the daemon |

## Supported KV formats

The connector offloads paged attention and MLA KV caches. Quantized/packed
(fp8, uint8) caches are handled automatically — **no configuration**:

- **Plain fp8 MLA latent** (hidden = `kv_lora_rank + qk_rope_head_dim`, a plain
  cast with no interleaved scale — e.g. Kimi-K2.6's 576 = 512 + 64) → **offloaded
  to L3**. Auto-detected at `register_kv_caches` from the model config.
- **Scale-packed layouts** (fp8_ds_mla's 656/584-byte latent that interleaves tile
  scales, nvfp4, per-token-head-scale fp8) → **skipped** (raw-byte chunk gather has
  no scale awareness, so reload could mis-stride). No L3 for these layers; the rest
  of the request is unaffected. A `register_kv_caches: … SKIPS it` warning names the
  layer. Supporting a new packed layout is a code change to the detector, not a knob.
- **DSA sparse-attention** (`glm_moe_dsa`, `deepseek_v32`) → the main latent **and**
  the co-packed indexer are offloaded together via a dedicated split path (both must
  be restored or the sparse top-k goes wrong).
- **bf16 / fp16 / fp32** → offloaded unconditionally.

```{admonition} No L3 hits on a quantized model?
:class: note
If a model's `Prefix cache hit rate` shows L1 hits but `External prefix cache hit
rate` stays 0, check the engine log for a `register_kv_caches … SKIPS it` warning —
its KV is a scale-packed layout the raw-byte path can't safely round-trip yet.
```

## Per-request control

Hint per request via OpenAI-compat `extra_body.kv_transfer_params` (mirrors
Anthropic's `cache_control`):

- **`infera_retention`** — `none | short | long` (**default `long`**, cache by
  default). `none` disables caching for that request; the class biases RAM
  eviction order and whether the block persists.
- **Prewarm** — an agentic harness can `POST /v1/cache/prewarm` to async-pull L3
  blocks into the warm arena before the next call (needs the server's
  `--kvd-socket-path`). Combined with the zero-copy read path, NFS-backed L3 reads
  drop out of the TTFT budget.

```{admonition} ephemeral / TTL are not per-request (yet)
:class: note
The daemon/wire protocol has an `ephemeral` class and a TTL field, but the vLLM
connector doesn't expose either per request — `ephemeral` maps to `long` and
`infera_ttl_seconds` isn't propagated. Tune those at the daemon for now.
```

## Related

- [KV-Cache Management](../components/kvd.md) — the tier design, the
  offload/onboard mechanism, the tablespace on-disk format, and optimization.
- [CLI reference](../reference/cli.md) · [Environment variables](../reference/environment.md)
  — the daemon flags and connector env vars.
