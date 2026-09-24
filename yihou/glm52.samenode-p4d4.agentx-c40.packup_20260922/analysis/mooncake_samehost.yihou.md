# Mooncake same-host transport — source investigation

Role: `mooncake-source`. Question that motivated this: the RDMA-loopback round
(`rounds/002-rdma-loopback/`) proved same-host RDMA between two *different* ionic
devices silently fails, but all of that was *host* memory. Does the mooncake
transfer engine even use RDMA for a same-host peer, or does it take a local
(GPU-IPC) path that would make the loopback problem moot?

Method: read the actual shipped artifacts in the run image
`infera-sglang:v0519-yihou-0917-nextnfix-hicache` on `crsuse2-m2m-137` (ssh + a
throwaway `yihou-mc-probe` container). The transfer engine is a **compiled
`.so`** — `mooncake/engine.cpython-310-x86_64-linux-gnu.so`; no C++ source is
shipped. So findings come from three tiers, marked per claim:
- **[binary]** first-hand: symbol/string extracted from the exact `.so` we run, or
  a log line produced by running that `.so`.
- **[run]** first-hand: output of actually initializing the engine in the image.
- **[upstream]** second-hand: `kvcache-ai/Mooncake` `main` on GitHub. The binary
  was built from `/tmp/mooncake-upstream/…` (build path embedded in the `.so`),
  but `main` has **drifted** (some binary strings are absent from `main`), so
  upstream is corroborating design, not the exact code. Every upstream claim
  below was cross-checked against a matching **[binary]** symbol.

---

## TL;DR for the two decision questions

**Q1 — Does mooncake take a local (non-RDMA) path for a same-host peer? YES
(by design, and the machinery is present and active in our exact binary).**

The engine ships an **intra-node HIP GPU-P2P transport** (GPU-IPC, not RDMA). It
**installs at runtime** in our image ([run], see §1). A registered GPU buffer is
advertised as multi-protocol **`"rdma,hip"`** even under the deployment's default
`MOONCAKE_PROTOCOL="rdma"` — **now PROVEN by a live 2-process transfer**, not
inferred: a peer with **zero RDMA transport** successfully read a GPU segment that
the other side registered under `protocol="rdma"`, which is only possible if that
segment offered hip (§15). When the peer is on the **same host**, the
multi-protocol selector prefers **hip (priority 4) over rdma (priority 2)** and
routes the KV copy over GPU-IPC, **bypassing RDMA entirely**; for a cross-host
peer it skips hip and falls back to rdma (§2). HIP moving GPU memory end-to-end
same-host (GPU0↔GPU4, cross-NUMA) is itself **proven by transfer** (§15). Same-host
GPU-P2P is physically available for our exact layout — every GPU pair, including
prefill 0-3 ↔ decode 4-7, is `True`/`XGMI` (§4). **So the same-host KV transfer
does not go through the broken cross-device RDMA loopback path** — provided
nothing disables hip. Nothing does: see the critical config bug in §3.

> **Correction to an earlier draft.** §1–§2 originally stated the `"rdma,hip"`
> advertisement as `[upstream]` inference cross-checked only to the presence of
> multi-protocol symbols. That was too strong for an inference. §15 now settles it
> `[run]` by experiment. The conclusion is unchanged, but it is now first-hand.

Citation: `MultiTransport::selectTransport route: target_id=` and
`protocol_priority` (`hip` → `std::getenv("MC_DISABLE_HIP") ? 0 : 4`) in
`multi_transport.cpp` [upstream], both present as [binary] symbols
(`_ZN8mooncake14MultiTransport18mp_selectTransport…`, `protocol_priority`,
`chosen_priority`, `local_server_name_`); HIP transport install log
`transfer_engine_impl.cpp:412 HIP transport installed for intra-node GPU P2P`
[run].

**Q2 — How is the destination RDMA device chosen; what does
`MC_ENABLE_DEST_DEVICE_AFFINITY=1` do?**

This only matters on the **RDMA** path (cross-host, or if hip is ever disabled).
The selector is `mooncake::selectPeerDevice(SegmentDesc*, …, localDeviceName, …)`
[binary symbol, file-local static in `rdma_transport.cpp`]. It takes the **local
device name** and picks a device on the peer segment; `MC_ENABLE_DEST_DEVICE_AFFINITY=1`
(read via `Environ::GetEnableDestDeviceAffinity`) switches it from default
selection to **destination-device affinity** — choosing the peer device that
corresponds to the local one, logged as `peer_device_name=…` [binary strings].
With **identical device lists on both legs** (both `ionic_0..3`), that yields the
`ionic_i ↔ ionic_i` pairing the loopback round wants. **Caveat (suspended):** I
could not extract the exact tie-break (match by *name* vs by *index*) — upstream
`main` no longer contains a function named `selectPeerDevice`, and I will not
guess from a drifted source. It is bounded by `[RDMA] Peer device index out of
range` [binary], i.e. an index-based lookup guarded against overflow. **This
question is moot for the same-host legs** because those transfers take hip, not
rdma — it only governs behavior if the hip path is disabled or unavailable.

---

## 1. The local path exists AND installs in our image — [binary]+[run]

The `.so` contains a full HIP transport module
(`src/transport/hip_transport/hip_transport.cpp`) built on GPU-IPC:
`hipIpcGetMemHandle` / `hipIpcOpenMemHandle`, VMM shareable handles
(`hipMemExportToShareableHandle`) with cross-process fd passing (`pidfd_open` /
`pidfd_getfd`), and `hipDeviceEnablePeerAccess` (strings
`HipTransport: P2P access not available between device`, `failed to enable P2P
access from device`). It is explicitly **intra-node**:
`Failed to install HIP transport (intra-node GPU P2P unavailable)` /
`HIP transport installed for intra-node GPU P2P`.

Running the engine in the image (GPU 0, `protocol="rdma"`, `device="ionic_0"`,
**with our exact env including `MC_DISABLE_HIP_TRANSPORT=1`**) prints:
```
transfer_engine_impl.cpp:259 Auto-discovering topology...
topology.cpp:168 IB device whitelist: ionic_0,ionic_1,ionic_2,ionic_3
transfer_engine_impl.cpp:412 HIP transport installed for intra-node GPU P2P
```
and `get_local_topology` returns a memory-location map with a **`hip:0`** class
alongside `cpu:0`/`cpu:1`. So the HIP transport comes up even though our config
"disables" it — because the flag name is wrong (§3).

## 2. How the transport is chosen for a transfer — [upstream] ⨯ [binary]

sglang hands the C++ engine `(session_id, src_ptrs, dst_ptrs, lengths)` and lets
it decide (§5). The decision is `MultiTransport::selectTransport` / the
multi-protocol `mp_selectTransport` [binary symbols]. Upstream
`multi_transport.cpp` [upstream], matching those symbols:

- `protocol_priority`: **hip=4** (`if (p=="hip") return std::getenv("MC_DISABLE_HIP") ? 0 : 4;`),
  shm=4, cxl=3, **rdma=2**, tcp=1.
- `const bool local_ipc_reachable = isLocalIpcReachableTarget(target_segment_desc->name, local_server_name_);`
  — hip/musa/shm are **skipped when `!local_ipc_reachable`** ("hip transport uses
  GPU IPC, which cannot reach a GPU on another host") and **preferred when local**.
- GPU KV pool "registered under both rdma and hip, so a cross-host target must
  skip its hip buffers and fall back to rdma" → segment advertised `"rdma,hip"`.
- Log: `MultiTransport::selectTransport route: target_id=` [present in [binary]].

Net: **same host → hip (local GPU-IPC); different host → rdma.** The binary
carries every one of these symbols (`is_multi_protocol`, `protocol_priority`,
`chosen_priority`, `local_server_name_`, `encodeMultiProtocolSegmentDesc`,
`decodeMultiProtocolSegmentDesc`, `mp_selectTransport`), so the design is the
one our `.so` implements.

## 3. CRITICAL config bug — our HIP-disable flag is a no-op — [binary]+[repo]

`config.sh:89` / `config.full.sh:97` set `MC_DISABLE_HIP_TRANSPORT=1` and
`engine.sh:116` forwards `-e MC_DISABLE_HIP_TRANSPORT=…` into the container.
**The binary does not read `MC_DISABLE_HIP_TRANSPORT`. It reads `MC_DISABLE_HIP`.**
First-hand: `strings … | grep -x MC_DISABLE_HIP_TRANSPORT` → **absent**;
`grep -x MC_DISABLE_HIP` → **present**; and the priority gate is literally
`std::getenv("MC_DISABLE_HIP")` [upstream]. So:

- Our intended "turn the HIP transport off" **does nothing**. `MC_DISABLE_HIP` is
  unset, hip priority stays 4, and same-host transfers take the local path.
- For the same-node design this is **the behavior we want** (it gives us the local
  path for free), but it is happening **by accident, via a misnamed flag**, not by
  intent. If anyone "fixes" the flag name to `MC_DISABLE_HIP=1`, the same-host legs
  would be forced back onto RDMA and would hit the cross-device loopback wall from
  round 002. Recommend: leave `MC_DISABLE_HIP` unset; do not rename the flag.

Not settled (suspended): I confirmed `MC_DISABLE_HIP` is the token the binary
reads and that upstream gates hip priority on it; I did not disassemble to prove
the binary's gate binds to the transport install as opposed to only the priority.
Either way the observable outcome (hip installs, hip has non-zero priority) holds
[run].

## 4. Same-host GPU P2P is available for our exact layout — [run]

`rocm-smi --showtopoaccess` / `--showtopotype` inside the image: **every** GPU
pair is `True` and linked by **`XGMI`**, including cross-NUMA pairs 0↔4, 1↔5,
etc. So the HIP GPU-IPC transport can serve prefill (GPUs 0-3, NUMA0) ↔ decode
(GPUs 4-7, NUMA1); the cross-socket boundary is **not** a P2P barrier here. This
removes the residual risk that the local path would fail on the 0-3/4-7 split.
(rocm-smi noted the idle GPUs are in a low-power state; accessibility/link-type
are static fabric properties, unaffected.)

## 5. sglang does NOT special-case a same-host peer — [binary/source in image]

`disaggregation/mooncake/conn.py`: no hostname/`is_local`/`127.0.0.1`/colocation
branch. The KV path is uniform — `_transfer_data()` (conn.py:641) calls
`self.engine.batch_transfer_sync(session_id, src_addrs, dst_addrs, lengths)`
→ `engine.batch_transfer_sync_write(...)`. The wrapper
`device_communicators/mooncake_transfer_engine.py` initializes the engine as
`self.engine.initialize(hostname, "P2PHANDSHAKE", protocol, device_name)` with
`protocol = envs.MOONCAKE_PROTOCOL` (default `"rdma"`) and `device_name` = the
per-GPU entry from the `RDMA_DEVICE` JSON map. **All same-host/local routing is
delegated to the C++ engine's `selectTransport`** — there is no Python-level
local shortcut and none is needed.

## 6. `MC_TE_FILTERS` = the IB-device discovery whitelist — [run]

First-hand: setting `MC_TE_FILTERS=ionic_0,ionic_1,ionic_2,ionic_3` makes the
engine log `topology.cpp:168 IB device whitelist: ionic_0,ionic_1,ionic_2,ionic_3`
during `Auto-discovering topology...`. So it filters **which HCAs are discovered
and enter the topology matrix** (device discovery), *before* any transfer — not a
per-transfer selection knob. It scopes the RDMA transport's device set; it has no
bearing on hip selection.

## 7. `MOONCAKE_DISABLE_HIP_DMABUF=0` and the HIP-transport flag — [binary]

- `MOONCAKE_DISABLE_HIP_DMABUF` gates GPU-memory registration for **RDMA**
  (GPUDirect via dmabuf). The message `HIP dmabuf disabled via
  MOONCAKE_DISABLE_HIP_DMABUF` fires only when it is truthy; our value **`0`**
  leaves dmabuf-based GPU registration **enabled**, i.e. GPU memory is registered
  for GPUDirect RDMA. This is what lets the KV pool advertise the `rdma` half of
  `"rdma,hip"`. (`MOONCAKE_DISABLE_HIP_DMABUF` *is* a real, read env var in the
  binary — unlike `MC_DISABLE_HIP_TRANSPORT`.)
- `MC_DISABLE_HIP_TRANSPORT` — inert (§3). The hip half of the advertisement stays
  on regardless of it.

## 8. Error path matching "QPs connect, transfer never completes" — [binary]

The RDMA path has `WorkerPool::isLocalWcFailure(const ibv_wc&)` and
`submitPostSend … failed_target_ids`, plus `[RDMA] Peer device index out of
range`. A silent WC failure (posted send, no completion) is the shape that would
match round 002's "QPs connect, no data, rc=1" **if** a same-host transfer ever
went over RDMA between mismatched devices. In the engine that path is **not
reached for same-host GPU transfers** because hip is chosen first (§2). It would
only surface if hip were disabled/unavailable and RDMA fell back onto a
cross-device pair. Noted, not concluded.

## 9. Operational flag found in passing — NOT part of the question — [run]

In the throwaway probe container, `libibverbs` rejected the ionic devices:
`Driver ionic does not support the kernel ABI of 4 (supports 1 to 1) …` for
`ionic_0..7`, `Skipping device: mlx5_0`, → `Found 0 HCAs`. This is **almost
certainly a probe-container artifact** (I launched a bare `sleep` container with
`--device /dev/infiniband` but none of `engine.sh`'s library setup): the proven
cross-node 20260920 runs used this same image and RDMA worked end-to-end, which
is impossible with 0 HCAs. Flag for whoever does the bring-up: confirm the real
`engine.sh` container discovers its HCAs (`Found N HCAs`, N>0) in its startup
log; if it also shows 0, the container's bundled rdma-core is ABI-incompatible
with the host ionic driver and RDMA won't install — in which case *everything*
same-host would fall to hip/tcp anyway, but cross-host KV would break. Undetermined
from this probe; needs a check against the real launch.

---

## 10. Locality routing key — sharpened, [binary] — same IP, different port is FINE

The selector's same-host test is `mooncake::isHipReachableTarget(segmentName,
localServerName)` [binary symbol `_ZN8mooncake20isHipReachableTargetE…`], with
helper `mooncake::segmentHost(segmentName)` [binary symbol]. `segmentHost` strips
the port and returns the host; the reachability check compares hosts (log field
` hip_reachable=`). Segment names are `host:port` (see §11). **Both our legs
advertise host `10.245.152.249`; the port differs, but `segmentHost` strips it,
so both resolve to the same host → `hip_reachable=true` → HIP path.** So the
same-IP-different-port co-location does NOT defeat the locality check — it is
exactly the case the check is built to accept. A cross-host peer has a different
`segmentHost` → not hip-reachable → falls back to rdma. This is the concrete
mechanism behind the Q1/Q3 "YES".

## 11. Q7 — two engines on one host do NOT collide — [run], decisive

First-hand: two `TransferEngine`s initialized in one process on `10.245.152.249`
(the co-location scenario) got:
```
A: RPC using P2P handshake, listening on 10.245.152.249:15307   (TCP 15330)
B: RPC using P2P handshake, listening on 10.245.152.249:15338   (TCP 15993)
segment keys: "10.245.152.249:15307" vs "10.245.152.249:15338"  (distinct)
```
No `EADDRINUSE`, no bind failure, both `init rc=0`, ports distinct.

- **RPC / P2P-handshake port is auto-assigned**, not fixed. `parseHostNameWithPort`
  parses a *default* `12001` from a bare `server_name`, but the engine then binds a
  different ephemeral port (15307/15338 across runs; 16608/15667/16580/16162 in
  earlier runs — always different). The segment name embeds that actual port
  (`removeSegmentDesc 10.245.152.249:15307 finish`).
- **`MC_HANDSHAKE_PORT` is ignored under P2P handshake** — [binary] string
  `Ignore value from environment variable MC_HANDSHAKE_PORT`. So you cannot (and
  need not) pin it; auto-assignment is what keeps the two legs distinct.
- `MC_MIN_RPC_PORT`/`MC_MAX_RPC_PORT` (and `MC_MIN_PRC_PORT`/`MC_MAX_PRC_PORT`)
  bound the auto-assignment range; `MC_LEGACY_RPC_PORT_BINDING` toggles the old
  fixed-binding behavior [binary strings]. **Leave all unset** — the defaults
  auto-assign and disambiguate. Do not set `MC_LEGACY_RPC_PORT_BINDING`.
- **Metadata backend is P2P handshake**, not a shared store. sglang passes
  `"P2PHANDSHAKE"` as the metadata arg (wrapper `initialize(hostname,
  "P2PHANDSHAKE", …)`), confirmed by [run] log `RPC using P2P handshake`. Segments
  are keyed by segment name = `host:port` and fetched peer-to-peer over the RPC
  port — there is no etcd/redis/http key for the two legs to clash on. (The binary
  supports etcd/redis/http too — `MC_REDIS_DB_INDEX`, `getFullMetadataKey` — but
  our config does not use them.)
- **`EADDRINUSE` behavior**: not reproduced (auto-assign avoids it). The engine
  binds within the port range; `SocketHandShakePlugin: bind (port …` is the bind
  site. Unsettled: exact retry/exhaustion behavior — not exercised, and not on our
  path.

**Verdict on Q7: no port or segment-key collision between the two legs.** The one
thing NOT to do is pin `MC_HANDSHAKE_PORT` / enable `MC_LEGACY_RPC_PORT_BINDING`
identically on both legs — the defaults already keep them apart.

## 12. `MC_USE_HIP_IPC` — mechanism selector, not an on/off — [binary]+[run]

The HIP transport has **two** GPU-memory-sharing mechanisms in the binary:
- **fabric/VMM**: `hipMemExportToShareableHandle` (+ `hipMemCreate`/`hipMemMap`),
  guarded by `use_fabric_mem_`;
- **IPC**: `hipIpcGetMemHandle`/`hipIpcOpenMemHandle`;
- with a fallback path — [binary] string `falling back to IPC mode`.

`MC_USE_HIP_IPC` [binary token; `MC_ENABLE_HIP_TRANSPORT`/`MC_DISABLE_HIP_TRANSPORT`
are **not** in the table] selects/forces the IPC mechanism. **It is not required
for the HIP path to exist:** [run] the transport installs and `register_memory`
on a real GPU buffer returns `rc=0` with `MC_USE_HIP_IPC` unset. So the default
brings HIP up; the env var only picks IPC vs fabric, and there is an automatic
fallback to IPC. Default value not pinned from the drifted upstream (suspended),
but the observable default behavior (installs, registers) is confirmed [run].
Recommendation: don't set it; if a real bring-up shows fabric-mode trouble, set
`MC_USE_HIP_IPC=1` as belt-and-suspenders — no evidence it is needed.

## 13. Q5 — dmabuf: kernel supports it; the "doubling" is not evident in mooncake

- **`MOONCAKE_DISABLE_HIP_DMABUF` is a real env var** the binary reads (unlike
  `MC_DISABLE_HIP_TRANSPORT`). Our `=0` leaves dmabuf **enabled**. The RDMA GPU
  registration path is `RdmaContext::exportDmabuf` → `registerMemoryRegion(…,
  DmabufExport)` [binary symbols], with a runtime guard
  `isKernelDmabufSupported()` and graceful fallbacks:
  `dmabuf export skipped (pages may migrate); falling back to ibv_reg_mr` and
  `Kernel lacks CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY` [binary strings].
- **Kernel DOES support dmabuf on our nodes** — [run] on both 137 and 276
  (`6.8.0-107-generic`): `CONFIG_PCI_P2PDMA=y`, `CONFIG_DMABUF_MOVE_NOTIFY=y`. So
  the fallback-to-`ibv_reg_mr` guard does NOT trigger; with `=0`, dmabuf export IS
  taken on the RDMA registration path.
- **Reconciling the contradiction (config `=0` vs leg.sh default `1`, "no-ODP
  dmabuf doubles the pool", yet 0.85 worked):** two independent reasons it holds:
  1. Mooncake's dmabuf path *exports the existing GPU allocation* as a dma-buf fd
     and registers that (`exportDmabuf(addr) → ibv_reg_dmabuf_mr`). It **pins**
     (prevents migration) but there is **no second allocation** in the mooncake
     code — nothing in the binary path duplicates the KV pool. The "doubles the
     pool" claim comes from **infera's own detector** (`infera/kvd/…`), a separate
     component and a conservative worst-case warning; I cannot confirm a doubling
     from the mooncake binary, and the working 0.85 run is consistent with *no*
     doubling. **Suspended** on the infera-detector claim itself — infera code,
     out of scope here.
  2. For the **same-host** legs the KV transfer uses HIP, not RDMA, so the RDMA
     dmabuf registration is not on the hot transfer path. (It still happens at
     `register_memory` time because the pool registers with every installed
     transport — so any RDMA-side dmabuf cost is incurred once at startup, not per
     transfer.)
  Net: dmabuf is genuinely taken (kernel supports it), it pins rather than
  duplicates, and it is irrelevant to the same-host transfer path. `=0` is safe.

## 14. Live cross-process transfer test — attempted, INCONCLUSIVE (harness)

I tried to prove the GPU copy end-to-end: two processes on 137, a `0xAB`-filled
buffer on physical GPU4, a `transfer_sync_read` from a peer on physical GPU0
(cross-NUMA — the real prefill↔decode pairing). It failed **before** any HIP
copy, at the P2P metadata handshake: `SocketHandShakePlugin: connect()
10.245.152.249:16162: Connection refused [111]`. That is my **ad-hoc harness**
missing sglang's bootstrap wiring (sglang exchanges bootstrap/segment info through
its own KV bootstrap server and keeps the engine serving; my bare two-script
handshake did not), **not** a HIP-transport result. A negative here is
inconclusive by design; it does not weaken the routing evidence in §1–2, §10. The
end-to-end GPU copy is best confirmed at the real bring-up (the KV transfer either
completes or errors visibly in the engine log). **Unsettled, first-hand:** the
actual byte-movement over HIP was not reproduced in isolation.

---

## 15. Decisive A/B transfer experiment — [run], settles the advertisement

Motivation: a reviewer challenged whether `protocol="rdma"` (the value sglang
passes — `mooncake_transfer_engine.py:205-219` with `MOONCAKE_PROTOCOL` default
`"rdma"` at `environ.py:803`, and engine.sh sets no override) limits GPU segments
to **rdma-only**, which would force same-host KV onto the broken cross-device RDMA
path. My earlier `[run]` did **not** settle this — it had 0 HCAs (ionic ABI, §9)
so RDMA never installed, making the advertisement unobservable, and my
`"rdma,hip"` statement was `[upstream]` inference. This experiment settles it.

Setup: exact image on 137, 2 processes, same host, GPU0↔GPU4 (cross-NUMA), a
4 MiB GPU buffer filled `0xCD`, `transfer_sync_read`. RDMA via **`mlx5_0`**
(`MC_TE_FILTERS=mlx5_0`, `MC_GID_INDEX=3`) because ionic is ABI-broken in a bare
container; the multi-protocol advertisement is device-independent. Handshake over
`127.0.0.1` (see the hairpin note below).

| run | init `protocol` | CLI transports | `transfer_rc` | data | verdict |
|---|---|---|---|---|---|
| HIP-only | `"hip"` | hip (+rdma) | **0** | correct | HIP moves GPU data end-to-end same-host cross-NUMA — **byte-movement proven** (closes the §14 gap) |
| A (default) | `"rdma"` | rdma(mlx5)+hip | 0 | correct | completes, but ambiguous (mlx5 same-device rdma AND hip both viable) |
| B | `"rdma,hip"` | rdma+hip | 0 | correct | completes |
| **DISCRIMINATOR** | SRV `"rdma"` / CLI **0 HCAs, hip-only** | hip only | **0** | **correct** | **decisive — see below** |

**The discriminator is the proof.** The SRV registered a GPU buffer under
`protocol="rdma"` with both mlx5 RDMA and HIP installed. The CLI was given a bogus
`MC_TE_FILTERS` → `Found 0 HCAs`, so it had **no RDMA transport at all** (only HIP
+ TCP). It still read the SRV's `rdma`-initialized GPU segment successfully. A
segment advertised **rdma-only** would have failed a hip-only peer with "transport
rdma not installed" (`NotSupportedTransport`). It did not fail. **Therefore a
registered GPU buffer is advertised under every installed transport that can serve
GPU memory — `"rdma,hip"` — regardless of the init protocol string.** The protocol
string selects the default/CPU protocol and which network transport installs; it
does **not** gate the GPU multi-protocol advertisement. The reviewer's
reconciliation hypothesis is **falsified**; the original "same-host → HIP"
conclusion holds, now first-hand.

Honest caveats:
- **mlx5_0, not ionic.** I used mlx5 to get a working RDMA transport in a bare
  container (ionic's provider is ABI-incompatible with the host kernel, §9). The
  advertisement logic is device-independent, but the identical test was not run on
  ionic. On the real 276 deployment ionic RDMA works (engine.sh mounts the
  host-matching `libionic.so`, `HOST_RDMA_LIB`).
- **No route log.** `MultiTransport::selectTransport route:` did not print even at
  `GLOG_v=5`, so routing was proven by the hip-only-peer discriminator, not by a
  log line. At real bring-up there is no "chose hip" line to grep.
- **hip vs tcp.** The hip-only CLI also had TCP installed. HIP outranks TCP (4 > 1)
  for GPU memory and I separately proved HIP moves GPU data, so the same-host path
  is hip — I did not additionally disable TCP to nail that last alternative.
- **Recommendation:** do **not** set `MOONCAKE_PROTOCOL="rdma,hip"` — it is
  unnecessary (default already yields the hip path) and would deviate from the
  proven cross-node config.

### Node-IP hairpin — a co-location networking caveat found here — [run]

While debugging a persistent handshake `Connection refused`, I found (with a plain
socket, nothing to do with mooncake) that on **137** a listener on `0.0.0.0`
cannot be reached by connecting to the node's own primary IP `10.245.152.249`
(refused); `127.0.0.1` and the RoCE IP `10.245.153.247` work. This is host
iptables (k8s), and it is why my earlier same-host 2-process transfers failed at
the handshake — a 137 artifact, **not** a mooncake bug. **On 276 (the target) the
same connect to `10.245.152.249` SUCCEEDS** (verified via `spur exec`). sglang
advertises the mooncake handshake endpoint as `HOST_IP=10.245.152.249`
(engine.sh:113) and mooncake connects to that literal IP for same-host peers (it
does **not** rewrite to loopback), so the same-host handshake depends on the node
allowing that connect. **276 allows it; 137 does not.** Record for bring-up: if a
same-node deployment is ever attempted on a node with 137's stricter rule, the KV
handshake would fail regardless of transport — but 276 is clear.

---

## Bottom line for the design

The same-node 1P1D plan does **not** depend on same-host RDMA working between
matched `ionic` devices. The engine routes same-host GPU KV transfers over the
**HIP GPU-IPC local path** (priority 4 > rdma 2), which is installed and backed by
an all-to-all XGMI P2P fabric across both NUMA nodes. The `ionic_0-3`-on-both-legs
decision from round 002 remains a safe belt-and-suspenders choice (it keeps the
*rdma fallback* same-device too), but the primary same-host path is hip, not rdma.
Keep `MC_DISABLE_HIP` **unset** (do not rename `MC_DISABLE_HIP_TRANSPORT`) and keep
`MOONCAKE_DISABLE_HIP_DMABUF=0`.

Co-location is otherwise clean: the two transfer engines auto-assign distinct RPC
ports and distinct `host:port` segment keys despite the shared IP, so they do not
collide (§11) — leave `MC_HANDSHAKE_PORT` / `MC_LEGACY_RPC_PORT_BINDING` unset.
`MC_USE_HIP_IPC` is not required (HIP installs and registers without it); set it
only if a real bring-up shows fabric-mode trouble (§12). The one thing left
unproven first-hand is the actual HIP byte-movement in isolation (§14) — confirm
it at the real bring-up via the engine log.
