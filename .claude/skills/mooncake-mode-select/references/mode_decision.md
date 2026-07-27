# Mode decision — the full tree and how each precondition is probed

The detector (`infera/tools/preflight/mooncake_mode.py`) computes the
recommendation in `decide()`. This doc explains each branch and the probe behind
each precondition, so you can trust — or fix — the output.

## Decision tree

```
peer-mem module loaded?
├─ YES → Mode A  (bare ibv_reg_mr; all rails; stock image; MOONCAKE_DISABLE_HIP_DMABUF=1)
└─ NO
   └─ any ACTIVE NIC with ODP?
      ├─ YES → Mode B  (ibv_reg_dmabuf_mr on that NIC; dma-buf image; MC_MS_FILTERS=<nic>)
      │        └─ ODP NIC slower/fewer than the fast rails? → PERF-REGRESSION warning
      └─ NO  → Mode C  (cap-KV / driver-bug workaround; STUB)
```

The order is a **safety ranking**, not a performance ranking: A is the path with no
pinning and every rail available; B trades all-rails for no-pin on one ODP NIC; C
is where no full-KV path is safe.

## Precondition 1 — peer-mem module (decides A vs B/C)

**What it means:** a GPU peer-memory kernel module lets `ibv_reg_mr` register a raw
device pointer. Without it, `ibv_reg_mr` on VRAM returns **EFAULT**.

**Probe (`probe_peermem`), positive-evidence, multi-signal:**
1. `/proc/modules` contains `nvidia_peermem` / `nv_peer_mem` / `ib_peer_mem` /
   `amdp2p`.
2. MOFED peer-mem client registry: a dir under `/sys/kernel/mm/memory_peers/*`
   (strongest signal on MOFED — each registered client, e.g. `amdkfd`, `nv_mem`).
3. A nonzero `/sys/class/infiniband_verbs/uverbs*/peer_mem_clients`.

**Why positive-evidence only:** if we can't prove peer-mem is present we declare it
absent, steering toward the dma-buf/cap paths that are *safe without* peer-mem —
better than recommending Mode A and hitting EFAULT at KV registration.

## Precondition 2 — ODP NIC (decides B vs C)

**What it means:** ODP (on-demand paging) lets `ibv_reg_dmabuf_mr` **dynamic-attach**
GPU pages instead of pinning them. With ODP the KV pool is registered without a
copy; **without** ODP the driver pins the whole region → the KV pool is duplicated
in VRAM (and can exhaust a KFD resource → SIGSEGV / HIP-209 on a large pool).

**Probe (`_odp_support`):** `ibv_devinfo -d <dev> -v`, look for `ODP_SUPPORT`
(+`ODP_SUPPORT_IMPLICIT`) in the caps. `ibv_devinfo` lives in the **engine
container**, so on a bare login host every NIC reads ODP=absent — always probe
inside the image on the GPU node.

On spur, only **mlx5** reports ODP; the eight **ionic** rails do not. That single
fact is the entire reason spur dma-buf PD is forced onto the one mlx5.

## The NIC pick (Mode B)

Among ACTIVE ODP NICs, pick the **fastest, then lowest device name** (deterministic
— prefill and decode must independently pick the same NIC). Emit
`MC_MS_AUTO_DISC=0` + `MC_MS_FILTERS=<dev>` and `--disaggregation-ib-device <dev>`
so mooncake registers **only** on that NIC and never touches a non-ODP rail (which
would pin).

If any non-ODP rail is also active on the box (the spur ionic case), the env adds
`RDMAV_FORK_SAFE=1` — ionic needs it (`rdma_context setup failed: fork
compatibility [22]`), harmless for mlx5.

## Perf-regression flag (Mode B)

Fires when the ODP NIC(s) are **not** the node's fast-rail tier — i.e. slower link
speed and/or fewer of them than the fastest active rails. All KV is then funneled
through a downgraded fabric. The spur example: 1×200G mlx5 carrying what eight
400G ionic rails otherwise would. This is a real bandwidth loss the user must
accept; the detector prints it in bold red and the skill must repeat it verbatim in
the confirmation prompt.

## GID index

Each NIC's routable RoCE-v2 GID is auto-detected (`_routable_gid`): prefer an
IPv4-mapped RoCE-v2 GID (e.g. mlx5 idx 3 on spur), else a global-IPv6 RoCE-v2 GID
(ionic idx 1). **Link-local `fe80::` (idx 0) is skipped — it times out cross-node.**
The chosen index goes into `MC_GID_INDEX`.

## engine.so USE_HIP_DMABUF check (Modes B/C)

`probe_dmabuf_engine` locates the mooncake `engine.so` and `nm -D`-greps for
`ibv_reg_dmabuf_mr` / `hsa_amd_portable_export_dmabuf`. **Do not** `strings | grep` —
it's an external call symbol, not a string literal, so strings is always empty. If
the symbol is missing, the stock image compiled dma-buf out (a known CMake
propagation bug) and Mode B/C needs the rebuild
(`deploy/docker/scripts/build_mooncake_dmabuf.sh` /
`deploy/docker/Dockerfile.sglang.dmabuf`).

## Filling Mode C (for the follow-up agent)

C is where no full-KV path is safe (no peer-mem, no ODP). The intended fix is to
run dma-buf anyway but **cap the KV pool** so weights + 2×(pinned KV) fit in VRAM,
or apply the specific driver-bug workaround. The stub leaves `env` with
`MOONCAKE_DISABLE_HIP_DMABUF=0` and a `# TODO(cap-kv)` launch flag. Replace the
placeholder with the computed cap (`--max-total-tokens <N>` and/or a reduced
`--mem-fraction-static`), and flip `viable` to `True` once validated.
