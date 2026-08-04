# Adapt to your cluster — change things HERE

Everything site-specific in this kit lives in one of the two wrapper files in this
directory. `common.sh` and `engine/*.sh` carry the tuned recipe and contain no
addresses, paths, NIC names or GID indices at all. **If you need to adapt the
deployment to your cluster, edit a wrapper — not the engine scripts.**

## Which wrapper?

Run the mooncake-mode probe on one of your nodes first. It reads the node, not a
config file, and prints which registration modes are viable:

```bash
bash preflight_rdma.sh mode
```

Then pick by what it says about **peer-mem**:

| preflight says | wrapper | why |
|---|---|---|
| `peermem: present`, **mode A viable** | [`cluster.peermem.sh`](cluster.peermem.sh) | bare `ibv_reg_mr` hands the NIC the GPU pages directly — nothing pinned, KV pool not duplicated, **every** rail can carry KV |
| `peermem: absent`, **mode B viable** (an ODP NIC exists) | [`cluster.dmabuf.sh`](cluster.dmabuf.sh) | dma-buf is the only GPUDirect path without peer-mem, and it is only safe on an ODP NIC; KV is locked to that one NIC |
| **only mode C viable**, or nothing viable | *neither, yet* | mode C pins and **doubles** the KV pool and needs an explicit, model/TP/VRAM-specific cap. Preflight exits `2` in this case on purpose: it is a human decision, not a default. |

The probe does not just classify — for the mode it picks it prints the exact env
and launch flags. The wrapper fields below are a place to paste those values, not
a puzzle to solve.

## The fields you must fill in

Both wrappers have the same four blocks. Placeholders read `<like-this>`.

### 1. Nodes

| var | what |
|---|---|
| `PREFILL_NODE` / `DECODE_NODE` | names the bring-up host can run a command on |
| `PREFILL_IP` / `DECODE_IP` | each node's **data-plane** IP — the address on the KV network |
| `KIT_DIR` | where this kit lives; must be the **same path on both nodes** |

**The data-plane IP is not the management IP.** The legs advertise these addresses
to each other and to the router for the KV handoff. A worker that advertises a
non-routable address (`0.0.0.0`, `127.0.0.1`) is rejected at launch by infera's own
config preflight, but a *wrong-but-routable* one is not — it just hangs.

### 2. Image and weights

| var | what |
|---|---|
| `INFERA_IMAGE` | the engine image — an infera-sglang build newer than 0.2.0. Ships as the placeholder `<infera-sglang-image>`. |
| `MODEL_MOUNT` | host directory bind-mounted into the container |
| `MODEL` | the checkpoint, which must live **under** `MODEL_MOUNT` |
| `TOKENIZER` | usually the same path; the router loads it for kv-aware routing |
| `HOST_RDMA_LIB` / `HOST_RDMA_MOUNT` / `ENTRYPOINT_KEEP` | only if your image injects a host RDMA provider library at entrypoint — see below |

The checkpoint is ~400 GB and both legs read it during bring-up. Prefer local storage
on both nodes: a slow mount can take an order of magnitude longer and blow the ready
timeout — which presents as a crash loop, not as slow storage.

**If your image injects a host RDMA provider library**, all three variables are
required together. `HOST_RDMA_LIB` is the host path (point it at the *symlink*, so nodes
carrying different provider builds both resolve); `HOST_RDMA_MOUNT` is the in-container
path **that image's entrypoint reads**; `ENTRYPOINT_KEEP=1` keeps the entrypoint that
does the injecting. Set only some of them and nothing happens — the library is mounted
where nothing reads it, which is not an error. The failure is silent: the leg boots and
serves, `ibv_devinfo` inside the container reports `does not support the kernel ABI` and
finds **zero** devices, and mooncake moves KV over a 5-20× slower transport.
`start_container` warns when the in-container device count is 0; that line is the check.

### 3. Transport

The block that differs between the two wrappers. Each field is a value to copy from
the preflight report:

| var | mode A | mode B |
|---|---|---|
| `RDMA_IB_DEVICES` | all `PORT_ACTIVE` rails, comma-separated | the single ODP NIC |
| `MC_GID_INDEX` | the routable RoCEv2 GID index | same |
| `MOONCAKE_DISABLE_HIP_DMABUF` | `1` | `0` |
| `MC_MS_AUTO_DISC` / `MC_MS_FILTERS` | unset | `0` / the ODP NIC |
| `RDMAV_FORK_SAFE` | `1` | only if non-ODP rails are also present |

Two things about this block are worth knowing before you fill it in.

**`MC_GID_INDEX` is per-node, not per-cluster.** Two identical nodes routinely expose
the routable GID at different indices, because an empty slot on one shifts everything
after it. A wrong index fails loudly (`GID is NULL, please check your GID index`, on
every DP rank, at init) — cheap to catch, expensive to assume. If your nodes disagree,
set it per node rather than once in the wrapper. The link-local `fe80::` GID (usually
index 0) is never the answer; it is not routable across the fabric.

**A down rail must not be listed.** `RDMA_IB_DEVICES` should enumerate what is
actually `PORT_ACTIVE`, which is not always every card:

```bash
for d in /sys/class/infiniband/*; do
  grep -q ACTIVE "$d/ports/1/state" && basename "$d"
done | paste -sd,
```

Run it on **each** node — the two can legitimately differ.

### 4. Deployment shape

`TP`, `CTX`, `CHUNK`, the DPA/MTP/kvd switches, the router policy, and the two
`mem-fraction-static` values. The defaults are the recommended production shape;
the [main README](../README.md#recommended-configuration) explains what each one buys
and which pairs are coupled.

## Schedulers where `ssh <node>` does not work

`engine/up.sh` reaches each node through `$SSH_CMD`, defaulting to
`ssh -o StrictHostKeyChecking=no`. On a cluster where compute nodes are only
reachable through the scheduler, point it at whatever does work:

```bash
export SSH_CMD="<your-scheduler> exec"
```

It is invoked as `$SSH_CMD <node> <command>`. If your scheduler's syntax does not
fit that shape, run `engine/leg.sh` on each node yourself — it is a single command
per leg and takes the same env vars.

## What NOT to change

`engine/leg.sh` encodes a recipe where several settings are coupled in ways that are
not obvious, and where getting one wrong tends to produce a *plausible* result rather
than an error:

- `--ep-size` is emitted **unconditionally**, outside the DP-attention branch. They
  are different parallelism axes (expert vs attention); coupling them means turning
  DP-attention off silently collapses the MoE too.
- `--chunked-prefill-size` is a **global** budget that SGLang divides by `dp_size`
  only when DP-attention is on. One value serves both modes; a per-rank number
  hardcoded in a DPA-off branch cuts the global budget 8×.
- `--disable-custom-all-reduce` is on by default and **independent of MTP**, because
  the custom all-reduce kernel deadlocks on this architecture during speculative
  verify. Letting it follow MTP turns any MTP comparison into a two-variable one.
- The DSA-on-ROCm env block is mandatory on gfx950. Without it the model still serves
  — it just returns garbage.
