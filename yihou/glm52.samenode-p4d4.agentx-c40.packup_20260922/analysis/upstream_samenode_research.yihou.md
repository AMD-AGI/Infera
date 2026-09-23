# Upstream research: single-node (co-located P+D) SGLang PD on MI355X + ionic + Mooncake

Author: upstream-research teammate · Date: 2026-09-22
Scope: establish, from upstream/official sources, what is *documented and in-source* about
running prefill and decode on ONE node with Mooncake KV transfer over AMD Pensando ionic RoCE.

Version anchors:
- Our image `infera-sglang:v0519-yihou-0917-nextnfix-hicache`, in-image sglang `0.5.19.dev20260917+ga9fb1c3238`.
- Mooncake pinned in our tree to `kvcache-ai/Mooncake` **`faae8dd4`** (2026-08-04). Upstream `main` head
  at time of research: `7364f79d` (2026-09-22). Findings say explicitly when they apply to `main` vs our pin.

Evidence ranking: **[H]** first-hand (source I fetched + quoted, or a command run) · **[M]** second-hand
(mirror of an authoritative list, official blog) · **[L]** inference / my own reasoning, flagged as such.

> **CORRECTION (2026-09-22, from `strings` on our own `engine*.so` — first-hand binary evidence).**
> My first pass concluded our image compiles the locality routing OUT and carries the infera
> `MC_*_HIP_TRANSPORT` gate. **That was wrong.** The build script + `transfer_engine_impl.diff` gate I
> read target the **vLLM** image line, not this one. The actual sglang-hicache binary shows:
> `MC_ENABLE_HIP_TRANSPORT`/`MC_DISABLE_HIP_TRANSPORT` → **0** occurrences (infera gate absent);
> `mp_selectTransport`, `multi_transport_locality.h`, `is_multi_protocol`, the
> `MultiTransport::selectTransport route: target_id=` log, and `HipTransport: hipIpc*` strings all
> **present**. So **our image is built with `ENABLE_MULTI_PROTOCOL=ON`, HIP transport compiled in, and the
> locality routing live.** The single-node hip/XGMI fast path is therefore *available in our image*, gated
> at runtime only by (a) the hip transport being installed at engine init and (b) the KV pool being
> published as a `"rdma,hip"` segment — both verifiable at runtime with `MC_LOG_LEVEL=TRACE` (see §Q1a).
> Sections below that still say "compiled out of our image" are superseded by this note.

---

## Q1 — Does Mooncake have a documented same-host / single-node transfer mode?

**Answer: Yes, and it was built specifically for AMD PD.** Upstream Mooncake has a first-class
single-node fast path: a **`"rdma,hip"` multi-protocol segment** where the device KV pool is registered
under *both* rdma and hip, and `MultiTransport::selectTransport` picks the **hip transport (GPU IPC over
XGMI)** for same-host targets and **rdma** for cross-host targets — automatically, by comparing the host
portion of the segment name. This is gated by the **`ENABLE_MULTI_PROTOCOL` CMake option, which defaults
OFF**, and our build does **not** turn it on. So the mode exists in our pinned source but is **compiled
out of our image**. There is no single env var that "turns it on" at runtime; it is a build-time flag plus
a registration shape.

### Details & evidence

| # | Finding | Ev |
|---|---|---|
| 1.1 | Two upstream PRs built this for AMD PD. **#2682** "[TE] Support rdma+hip multi-protocol segments for **single-node disaggregation**" (merged 2026-07-01): *"lets a single node advertise and use both `rdma` and `hip` on one local segment … the device KV pool moves over the intra-node HIP transport (XGMI) while the host-resident metadata/aux buffers move over RDMA."* **#2753** "[TE] Route cross-host targets over rdma automatically in rdma+hip multi-protocol segments" (merged 2026-07-06) added the locality check. | [H] |
| 1.2 | The routing decision is in `mooncake-transfer-engine/src/multi_transport.cpp` `selectTransport`, under `#ifdef ENABLE_MULTI_PROTOCOL`. For a comma segment (`"rdma,hip"`) it computes `isLocalIpcReachableTarget(target, local)` and *skips* hip/musa/shm buffers when the target is **not** local, else prefers hip by fixed priority. Priority lambda: `if (p=="hip") return getenv("MC_DISABLE_HIP") ? 0 : 4; … rdma → 2; tcp → 1;`. | [H] |
| 1.3 | `isLocalIpcReachableTarget` (`include/multi_transport_locality.h`) is a **pure host-string compare**: it extracts the host part of the segment name (IPv4/hostname/bracketed-or-bare IPv6, dropping `:port`) and does a case-insensitive equality. **It does not probe XGMI/PCI reachability** — same host string ⇒ "local". For co-located P+D this is TRUE (both legs share the node IP, differ only by port) ⇒ it would route KV over hip GPU-IPC. | [H] |
| 1.4 | `ENABLE_MULTI_PROTOCOL` is `option(... OFF)` in `mooncake-common/common.cmake` (line 194 on main; line 169 at pin `faae8dd4`). The whole locality path is `add_compile_definitions(ENABLE_MULTI_PROTOCOL)`-guarded. **Both the locality header and `MC_DISABLE_HIP` already exist at our pin `faae8dd4`** (introduced by #2753, 2026-07-06) — so it is a *compile flag* question, not a version-drift question. | [H] |
| 1.5 | Our build script `deploy/docker/scripts/build_mooncake_rocm.sh` configures `cmake .. -DUSE_HIP=ON -DUSE_ETCD=OFF -DWITH_STORE=OFF -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF …` and **never passes `-DENABLE_MULTI_PROTOCOL=ON`**. Its own comment says: *"This build does not enable upstream's ENABLE_MULTI_PROTOCOL locality routing."* ⇒ in our image the locality path and `MC_DISABLE_HIP` are absent. | [H] |
| 1.6 | `installTransport("hip")` in `transfer_engine_impl.cpp` is **unconditional** on `main` (inside `#ifdef USE_HIP`, no env gate; lines ~498-508). Our tree wraps exactly this block with an **infera-invented gate** (`patches/mooncake_cpp/transfer_engine_impl.diff`): HIP transport **OFF by default on ROCm**, opt in with `MC_ENABLE_HIP_TRANSPORT=1`, hard veto with `MC_DISABLE_HIP_TRANSPORT=1`. | [H] |
| 1.7 | **`MC_ENABLE_HIP_TRANSPORT` and `MC_DISABLE_HIP_TRANSPORT` do NOT exist upstream** (GitHub code search over `kvcache-ai/Mooncake`: `MC_ENABLE_HIP_TRANSPORT` → 0 hits). They are ours. Upstream's only related knob is **`MC_DISABLE_HIP`** (1 hit, `multi_transport.cpp`) — and it only *de-prioritizes* hip inside the MP path; it cannot prevent the transport being installed, and it is a veto, not an enable. Our patch's comment states this correctly. | [H] |
| 1.8 | `MC_USE_HIP_IPC` (`hip_transport.cpp` `supportFabricMem()`): default is **IPC** mode (`hipIpcGetMemHandle`/`hipIpcOpenMemHandle`). Setting `MC_USE_HIP_IPC=0` switches to **fabric memory** (VMM/fabric-handle) mode. It selects *within* the HIP transport; it does not enable/disable the transport. | [H] |
| 1.9 | Same-host **DRAM** (not GPU): PR **#3918** (merged 2026-09-10) adds a POSIX **`ShmTransport`** for same-host different-process host-memory copies, opt-in `MC_FORCE_SHM=1`. Post-dates our pin; relevant only if a HiCache **host** pool needed same-host transfer. Not in our pin, not in our build. | [H] |

**Env vars the transfer engine actually reads (main, authoritative — grepped `getenv(...)`):**
`MC_FORCE_TCP`, `MC_FORCE_HCA`, `MC_FORCE_MNNVL`, `MC_FORCE_MUSA`, `MC_INTRANODE_NVLINK`,
`MC_TCP_BIND_ADDRESS`, `MC_RDMA_BIND_ADDRESS`, `MC_LEGACY_RPC_PORT_BINDING`, `MC_CXL_DEV_PATH`,
`MC_CUSTOM_TOPO_JSON`, `MC_MACA_HOST_TRANSPORT`, `MC_TE_METRIC[_INTERVAL_SECONDS]`,
`MC_DISABLE_HIP` / `MC_DISABLE_MACA` / `MC_DISABLE_MUSA` (multi_transport.cpp, MP path only),
`MC_HIP_NUM_STREAMS`, `MC_HIP_NUM_EVENTS`, `MC_USE_HIP_IPC`, `MC_USE_NVLINK_IPC`, `USE_BAREX`.
`MC_FORCE_SHM` exists on `main` (#3918) but not at our pin. **[H]**

**So, to actually get the single-node GPU-IPC path** you need all of: (a) rebuild the `.so` with
`-DUSE_HIP=ON -DENABLE_MULTI_PROTOCOL=ON`; (b) the hip transport installed (upstream: automatic; our
image: requires `MC_ENABLE_HIP_TRANSPORT=1`); (c) the KV pool published as a `"rdma,hip"` segment — which
is a *C++/registration* property, **not** something the sglang python `protocol="rdma"` arg sets (see Q2).
None of (a)/(c) is satisfied by our current image. **Confidence: HIGH.**

Sources:
- Mooncake `multi_transport.cpp` @ main: https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-transfer-engine/src/multi_transport.cpp
- `multi_transport_locality.h` @ main: https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-transfer-engine/include/multi_transport_locality.h
- `common.cmake` @ main: https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-common/common.cmake
- `hip_transport.cpp` @ main: https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-transfer-engine/src/transport/hip_transport/hip_transport.cpp
- PR #2682: https://github.com/kvcache-ai/Mooncake/pull/2682 · PR #2753: https://github.com/kvcache-ai/Mooncake/pull/2753 · PR #3918: https://github.com/kvcache-ai/Mooncake/pull/3918
- Local: `infera/deploy/docker/scripts/build_mooncake_rocm.sh`, `infera/deploy/docker/patches/mooncake_cpp/transfer_engine_impl.diff`

---

## Q1a — Re-aimed for OUR build line (`ENABLE_MULTI_PROTOCOL=ON`), read at pin `faae8dd4`

All quotes are at our pinned ref `faae8dd4` (`multi_transport.cpp`, `multi_transport_locality.h`,
`config.cpp`, `config.h`, `transfer_engine_impl.cpp`) and cross-checked against `main`.

**A. What our KV transfer actually calls.** `MultiTransport::submitTransfer` (the default path used by the
python `transferSync`/`batch_transfer`) calls **`selectTransport`** — the comma-protocol locality block.
`mp_selectTransport` is only reached from `mp_submitTransfer` (needs an explicit `proto` arg). So the
authoritative routing for us is `selectTransport`'s comma block. **[H]**

**B. What the locality decision keys on — the host string of the segment name only.** [H]
```cpp
// include/multi_transport_locality.h
inline bool isLocalIpcReachableTarget(target_segment_name, local_server_name) {
    return hostEquals(segmentHost(target_segment_name), segmentHost(local_server_name));
}
```
`segmentHost` strips a `:port` suffix (IPv4/hostname; bracketed `[v6]:port`; bare v6), `hostEquals` is
case-insensitive. **No GID, no subnet, no PCI/XGMI probe.** In `selectTransport`:
```cpp
if (p == "hip") return std::getenv("MC_DISABLE_HIP") ? 0 : 4;   // else cxl=3, rdma=2, tcp=1
const bool hip_reachable = isHipReachableTarget(target_segment_desc->name, local_server_name_);
... if (buffer.protocol == "hip" && !hip_reachable) continue;   // skip hip buffers cross-host
```
`mp_selectTransport` mirrors it: `if (preferred_proto=="hip" && !isHipReachableTarget(...)) → fall back to rdma/tcp`.

**C. Same host IP, different port ⇒ locality fires (our topology).** [H→ conclusion L]
`segmentHost("10.245.152.249:29001") == segmentHost("10.245.152.249:29002") == "10.245.152.249"` ⇒
`hip_reachable = TRUE` ⇒ the KV-pool hip buffer is not skipped and hip (prio 4) beats rdma (prio 2) ⇒ KV
rides **hip / XGMI GPU-IPC**. **Caveat (runtime, not source):** only if the segment is actually published
`"rdma,hip"` — i.e. the hip transport was installed at init and the pool registered under hip. Confirm at
runtime, do not assume.

**D. `MC_USE_HIP_IPC` and `MC_DISABLE_HIP` — defaults, gates, origin.** [H]
| var | default | what it gates | introduced |
|---|---|---|---|
| `MC_USE_HIP_IPC` | unset ⇒ **IPC mode** (`hipIpcGet/OpenMemHandle`) | IPC-handle vs fabric-memory (VMM) *within* the hip transport; `=0` opts into fabric mode. Does **not** enable/disable hip. | #1344 (2026-01-09 "Switch to IPC mode by default"); hip transport itself #1208 (2025-12-16) |
| `MC_DISABLE_HIP` | unset ⇒ **hip enabled** (prio 4) | any value ⇒ hip prio 0, disqualified in `selectTransport`, forcing rdma/tcp **even same-host**. Global veto of the hip fast path. | current priority use #2753 (2026-07-06); existed earlier as a blunt global kill (per #2753 body) |

**E. The per-transfer route log — the verification lever.** [H]
```cpp
if (globalConfig().trace)
  LOG(INFO) << "MultiTransport::selectTransport route: target_id=" << entry.target_id
            << " segment_protocol=\"" << proto << "\" hip_reachable=" << hip_reachable
            << " chosen=" << chosen;
```
`globalConfig().trace` is set **only** by `MC_LOG_LEVEL=TRACE` (exact string, `config.cpp`). Set it on the
engine to see, per transfer: segment protocol, `hip_reachable`, and the chosen transport. This directly
answers "is same-node KV on hip or rdma?".

**F. Two engines on one host — no default collision.** [H]
In P2P handshake mode (sglang passes `metadata=P2PHANDSHAKE`), `transfer_engine_impl.cpp` calls
`desc.rpc_port = findAvailableTcpPort(desc.sockfd)` — each engine **auto-binds a free RPC/handshake port**
from `[MC_MIN_RPC_PORT, MC_MAX_RPC_PORT]` (default **15000–17000**, `config.cpp`) ⇒ two co-located engines
get distinct ports automatically. `MC_LEGACY_RPC_PORT_BINDING` switches to the fixed port carried in the
hostname (`ip:port`), which then needs distinct ports per leg (sglang's `get_free_port()` supplies them).
`MC_HANDSHAKE_PORT` (default **12001**, `config.h`) is the metadata-server-mode listener, not the P2P
collision point. Conclusion: leave defaults ⇒ no collision; only `MC_LEGACY_RPC_PORT_BINDING` + identical
hostname ports would collide.

**Strategic consequence.** Because we are on the MP line, the clean same-node path (hip/XGMI, which
**bypasses ionic RoCE entirely** for intra-node KV) may already be one runtime-state away. First action on
the node: `MC_LOG_LEVEL=TRACE` and read the route log. If `chosen=hip`, the ionic same-host RoCE failure
(Q3) is **moot for intra-node KV**. If `chosen=rdma` despite `hip_reachable=1`, investigate why the pool
isn't published `"rdma,hip"` (hip transport install at init / buffer registration under hip).

Sources: as Q1, plus `config.cpp`, `config.h`, `transfer_engine_impl.cpp` @ `faae8dd4`; PRs #1208, #1344,
#2682, #2725, #2753.

---

## Q2 — Does SGLang special-case same-host PD?

**Answer: No.** SGLang initializes the Mooncake engine with a **single** protocol string
(`MOONCAKE_PROTOCOL`, default `"rdma"`; other choices efa/tcp/ascend — **no "hip", no "rdma,hip"**) and
defers all transport and locality selection to Mooncake. There is no same-host / colocation / loopback
detection in the KV manager or bootstrap; bootstrap is plain `host:port`, and two legs on one node simply
share the host and differ by port. Consequently, even if the `.so` were built with
`ENABLE_MULTI_PROTOCOL`, SGLang does not itself ask for the `"rdma,hip"` shape — the multi-protocol
segment is produced by Mooncake's C++ side from the *installed transports*, not by this python arg.

### Details & evidence

| # | Finding | Ev |
|---|---|---|
| 2.1 | `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py`: `initialize()` sets `protocol = envs.MOONCAKE_PROTOCOL.get()` (comment: *"selects the transport (rdma \| efa \| tcp \| ...). Default is 'rdma'"*), then `self.engine.initialize(hostname, "", protocol, device_name or "")`. **One** protocol string; hostname from `get_local_ip_auto()`; `device_name = ib_device`. `MC_FORCE_TCP=1` swaps to TCP. | [H] |
| 2.2 | `disaggregation/mooncake/conn.py` only does `engine.batch_register(ptrs, lens)` / `batch_deregister`. **No** protocol argument, no `installTransport`, no `rdma,hip`, no `MC_*` writes, no colocation branch. Bootstrap uses `rank_ip:rank_port` + `local_ip`; nothing special-cases equal host IPs. | [H] |
| 2.3 | GitHub code search across `sgl-project/sglang` for `colocate` / `same node` / `installTransport` in `disaggregation/` → **0** substantive same-host handling. `--disaggregation-ib-device` is passed straight through as `device_name`; it does not change when both legs share a host. | [H] |
| 2.4 | Open issues confirm same-node PD is still an area of active work rather than a settled feature, e.g. #39166 "[AMD][DSV4] enable PD-disagg with fp8 unified_kv on gfx950", #14775 "[Proposal] Memory-semantic PD-disaggregation paradigm", roadmap #21703. None documents a supported "1P1D one node" switch. | [M] |

**Caveat (version):** the above is `main`. Our in-image sglang is `0.5.19.dev20260917`; the wrapper
structure (`MOONCAKE_PROTOCOL` default `"rdma"`, single-protocol `engine.initialize`) is long-standing, but
I did not byte-verify the in-image copy — treat the exact line numbers as `main`. **Confidence: HIGH** that
SGLang has no same-host special path and passes a single protocol; **MEDIUM** on our exact image's lines.

Sources:
- `mooncake_transfer_engine.py` @ main: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py
- `disaggregation/mooncake/conn.py` @ main: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/disaggregation/mooncake/conn.py

---

## Q3 — AMD Pensando ionic RDMA: same-host cross-device limitations

**Answer (suspended, best current explanation):** The failure you measured — same-host, **different**
ionic devices, RoCEv2, **global IPv6 GID**: QPs connect and exchange QPN/PSN/RKey/GID but **no completion
ever arrives**, while **same-device loopback works** — matches a **known RDMA-core (NIC-agnostic)
same-host cross-NIC L2-resolution bug**, not anything ionic-specific I could find documented. A June-2026
`rdma-next` series ("RDMA: fix cross-NIC same-host IPv6 RDMA-CM connect", Alex Timofeyev) describes exactly
this: for a destination on a *different netdev of the same host*, `addr_resolve_neigh()` copies the
**source** NIC's MAC into the path record, so the destination NIC drops the frame; and on receive,
`validate_ipv6_net_dev()` rejects it because `rt6_lookup()` of a same-host v6 destination **collapses onto
`lo`** (local table 255). Both `rdma_resolve_addr()`/`ib_send_cm_req()` return success; the REQ never
lands. Crucially the author **withdrew the series**, concluding *"the right fix here is configuration, not
a kernel patch"* and recommending **VRF-per-NIC** (which moves each NIC's addresses out of the global v6
local table so the destination no longer collapses to `lo`).

Why this fits your data, and where it doesn't:
- **Same-device loopback works** because for *true* loopback (same netdev) copying the source MAC is
  correct — exactly the case the bug does **not** break. **Cross-device fails** because the source MAC is
  wrong for the peer netdev. This is a clean match to your same-device-OK / cross-device-FAIL split. **[L→M]**
- The **global IPv6 GID** (your ionic index 1) is *part of the trigger*: the collapse is specific to the
  v6 local route table. Your `mlx5_0` uses an **IPv4-mapped** GID (index 3), which resolves differently —
  so GID/address-family selection is genuinely implicated, and "switch to a non-global / IPv4 GID, or
  isolate via VRF" is the direction to test. **[M]**
- **Gap:** the withdrawn series is about **rdma_cm** specifically. Your plain `ib_write_bw` *without* `-R`
  does not use rdma_cm — it exchanges QP params over a TCP socket and builds the AH from the GID. That it
  *also* failed is **consistent** with the same L2-resolution collapse (RoCE AH creation resolves dgid→dmac
  through the same neighbour/route path), but the series does not directly evidence the non-CM path — this
  is my inference. **[L]**
- The official ionic_rdma kernel doc (v7.3.0-rc4) is high-level and says **nothing** about loopback,
  same-host, VRF, or GID selection. I found **no ionic-specific documentation** confirming same-host
  cross-device RoCE as a named unsupported case. **Suspend: not settled by an ionic source.** **[H]** (that
  the doc is silent)
- **Switch hairpin** is a *secondary* candidate (a frame leaving NIC-A destined to NIC-B on the same host
  must be hairpinned by the ToR; many switches disable reflective relay). But the MAC-resolution bug drops
  the frame at host L2 with `dst MAC == source NIC's MAC` — it likely never reaches the switch — so hairpin
  is the less-likely of the two. Not evidenced either way here. **[L]**

Primary source is a **mirror of lore** (lore.kernel.org itself is Anubis-bot-blocked to WebFetch), so rank
it **[M]**; the code paths it names (`addr_resolve_neigh`, `validate_ipv6_net_dev`, `rt6_lookup`→`lo`,
table 255, `RTF_LOCAL`) are real and checkable. **Overall confidence: MEDIUM** that this is the mechanism;
**HIGH** that it is not documented as an ionic-specific limitation anywhere I could find. Actionable next
tests: (a) **VRF-per-NIC** on the two ionic devices and re-run cross-device `ib_write_bw`; (b) try a
non-global / IPv4-mapped GID index; (c) sidestep RDMA entirely for same-host KV via the Q1 hip/XGMI path.

Sources:
- Patch series (lore mirror): https://ratatoskr.run/stable/2026/06/17135060/t — "[PATCH rdma-next v1 0/2]
  RDMA: fix cross-NIC same-host IPv6 RDMA-CM connect", Alex Timofeyev, 2026-06-15; Fixes-tags
  `c31e4038c97f`, `f887f2ac87c2`; **withdrawn**, VRF recommended.
- ionic_rdma kernel doc: https://docs.kernel.org/networking/device_drivers/ethernet/pensando/ionic_rdma.html
- LWN "Introduce AMD Pensando RDMA driver": https://lwn.net/Articles/1033763/
- AMD Instinct GPU-cluster networking troubleshooting: https://instinct.docs.amd.com/projects/gpu-cluster-networking/en/latest/how-to/troubleshooting.html
- Your own first-hand `ib_write_bw` matrix (276/137) — the strongest evidence, and it corroborates the mechanism.

---

## Q4 — ionic + GPUDirect (dma-buf / ODP / pinning)

**Answer: Confirmed from kernel source.** The ionic RDMA driver **does** implement
`ibv_reg_dmabuf_mr` (so GPUDirect via dma-buf works — this is the path that registers VRAM where
`ib_peer_mem` is absent), but it registers via the **pinned** dma-buf helper, and the device advertises
**no ODP**. So your detector is right: on ionic, dma-buf MR registration **pins** the backing pages and
there is no on-demand-paging / dynamic-attach path, none planned in-source.

### Details & evidence

| # | Finding | Ev |
|---|---|---|
| 4.1 | `drivers/infiniband/hw/ionic/ionic_ibdev.c` wires `.reg_user_mr_dmabuf = ionic_reg_user_mr_dmabuf` into `ib_device_ops` — so `ibv_reg_dmabuf_mr` is supported. | [H] |
| 4.2 | `ionic_controlpath.c:1017` — the body calls **`ib_umem_dmabuf_get_pinned(&dev->ibdev, offset, length, fd, access)`**. The `_get_pinned` variant **pins** the dma-buf immediately (no dynamic attach, no mmu-notifier move-notify). There is **no** `ib_umem_dmabuf_get` (dynamic) path in the driver. | [H] |
| 4.3 | `ionic_ibdev.c` `device_cap_flags = IB_DEVICE_MEM_WINDOW \| IB_DEVICE_MEM_MGT_EXTENSIONS \| IB_DEVICE_MEM_WINDOW_TYPE_2B` — **no `IB_DEVICE_ON_DEMAND_PAGING`**, and there is **no `odp_caps`** anywhere in the driver. Grep across `ionic_controlpath.c / ionic_ibdev.c / ionic_datapath.c / ionic_admin.c` for `odp\|on_demand\|IB_ACCESS_ON_DEMAND` → **0** hits. **ionic has no ODP.** | [H] |
| 4.4 | Upstream RDMA dma-buf design (Xiong series / LWN) states peer-to-peer device memory over dma-buf needs **dynamic attach ⇒ ODP**; the *pinned* fallback pins pages instead. So on an ODP-less NIC like ionic, registering VRAM dma-buf pins it for the MR lifetime. This is the general framework your detector's "pins and doubles the KV pool → SIGSEGV/HIP-209" warning sits on. The kernel source confirms **pinned + no ODP** first-hand; the exact "doubles" figure is your detector's operational observation, which I did not independently reproduce. | [H] source / [M] the doubling magnitude |

**Confidence: HIGH** on: ionic supports `ibv_reg_dmabuf_mr`; it pins; it has no ODP and none is in-source.
**MEDIUM** only on the precise "doubles the pool" magnitude (that is your runtime detector's claim, not a
number the source states).

Sources:
- ionic driver @ torvalds/linux: `drivers/infiniband/hw/ionic/ionic_controlpath.c`, `ionic_ibdev.c` —
  https://github.com/torvalds/linux/tree/master/drivers/infiniband/hw/ionic
- RDMA dma-buf series / ODP dependency: https://lwn.net/Articles/839314/ ,
  https://www.openfabrics.org/wp-content/uploads/2021-workshop-presentations/303_Xiong_DMA-BUF.pdf
- `ibv_reg_dmabuf_mr` man page: https://man7.org/linux/man-pages/man3/ibv_reg_mr.3.html

---

## Q4a — Resolving "pins and doubles the pool" vs. proven 0.85 runs with no OOM

Our own detector (`infera/examples/sglang_1p1d_glm5.2/cluster/cluster.dmabuf.sh`) says: use the **ODP** NIC
(mlx5) for dma-buf KV (no-pin); registering KV on a **non-ODP** rail (ionic) *"pins and doubles the pool"*
→ risk of OOM. Yet our proven cross-node runs registered KV over ionic dma-buf at `mem_fraction 0.85` with
no OOM. **First-hand resolution: pinning ≠ doubling on this stack. The detector's "pins" is right; its
"doubles the pool" is not borne out here.**

Kernel `drivers/infiniband/core/umem_dmabuf.c`, the exact path ionic takes (`ib_umem_dmabuf_get_pinned`):

| Fact | Ev |
|---|---|
| The pinned-attach ops set **`.allow_peer2peer = true`** (lines 185 & 209), then `ib_umem_dmabuf_get_pinned_and_lock` calls **`dma_buf_pin(attach)`** (line 230) and `dma_buf_map_attachment` (line 32). | [H] |
| `allow_peer2peer = true` tells the exporter (amdgpu) it may keep the buffer in a **peer-to-peer VRAM** aperture. On MI355X amdgpu with PCI P2PDMA, `dma_buf_pin` therefore pins the KV-pool BO **in place, in VRAM** — it does **not** allocate a second copy and does **not** migrate to host RAM. The MR maps the *existing* pages. | [H] source / [L] the "in VRAM on our HW" step is inference from `allow_peer2peer` + P2PDMA |
| So the only real cost of the missing ODP is that the registered region is **pinned resident (non-evictable, no on-demand faulting)** for the MR lifetime — a *flexibility* cost, not a footprint doubling. An active KV pool is already always-resident, so pinning adds ~0 VRAM ⇒ no OOM at 0.85. This matches the empirical result exactly. | [H]+[L] |
| The genuine concern the detector is gesturing at is the **classic non-ODP dma-buf caveat** (LWN/Xiong: without dynamic attach, if P2P is *unavailable* the exporter must migrate VRAM→system-RAM to satisfy a pin). That is a *migration* (host-memory pressure / perf cliff), **not** a VRAM doubling — and on our stack `allow_peer2peer` + P2PDMA + `hsa_amd_portable_export_dmabuf` mean P2P **is** available, so migration does not happen. A run where P2P were *not* honored would show as KV effectively in host RAM (catastrophic bandwidth), not a silent 2× VRAM OOM. | [M] (LWN) + [H] (build asserts the HSA export symbol) |

**Conclusion.** No contradiction. `ibv_reg_dmabuf_mr` on ionic pins the existing KV-pool VRAM in place
(P2P), it does not copy or migrate it, so it does not double the pool — the 0.85 no-OOM runs are the
expected behaviour, not luck. The detector's routing advice (prefer the ODP/mlx5 rail) remains reasonable
for its *stated* reasons (no-pin flexibility, avoiding a perf-regression warning), but its "doubles the
pool → OOM" wording overstates the risk for a P2PDMA-capable amdgpu + ionic. If you ever *do* see a 2×
symptom, the checkable cause is P2P **not** being honored (verify `dmesg` amdgpu P2P / that the BO stayed
in VRAM), not the pin itself. **Confidence: HIGH that pin≠copy from the kernel path; MEDIUM on the
in-VRAM-vs-migrate outcome being HW/topology-dependent — worth a one-line `rocm-smi` VRAM check before/after
`reg_dmabuf` to close it empirically.**

Sources: `drivers/infiniband/core/umem_dmabuf.c` @ torvalds/linux —
https://github.com/torvalds/linux/blob/master/drivers/infiniband/core/umem_dmabuf.c ;
detector `infera/examples/sglang_1p1d_glm5.2/cluster/cluster.dmabuf.sh` (lines 15-31, 74) and
`preflight_rdma.sh` (line 40); LWN dma-buf/ODP: https://lwn.net/Articles/839314/ .

---

## Bottom line for the bring-up

1. **The cleanest same-node KV path is Mooncake's hip/XGMI multi-protocol mode, not RDMA** — it sidesteps
   the ionic same-host RoCE problem entirely. But it is **not in our image**: needs a rebuild with
   `-DENABLE_MULTI_PROTOCOL=ON`, the hip transport installed (`MC_ENABLE_HIP_TRANSPORT=1` given our gate),
   and the KV pool published as `"rdma,hip"` — and SGLang does not request that shape today (Q2), so this
   is unproven on our stack and would need validation.
2. **If we stay on RDMA for same-node**, the leading (medium-confidence) explanation for the cross-device
   failure is the RDMA-core same-host v6-GID L2-collapse; the upstream-recommended fix is **VRF-per-NIC**,
   with "use a non-global GID" and "route both legs through one ionic device (same-device loopback works)"
   as things to test. None of this is ionic-documented.
3. **GPUDirect on ionic pins and has no ODP** — a hard constraint independent of 1/2.
