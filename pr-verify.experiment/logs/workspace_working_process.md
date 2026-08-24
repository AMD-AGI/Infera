# Working process — verifying the three upstream sglang GLM-5.2 PRs

Session 2026-08-19, on n06-33 (8x MI355X / gfx950). Task spec: `pr.verify.md`.
Prior session's record: `pr.done.md` (treat as second-hand until re-verified).

Goal: review, validate on real gfx950, and flip #33968 / #33970 / #33973 from
draft to ready.

## Round index

| Round | Purpose | Outcome |
|-------|---------|---------|
| r00 (inline) | Upstream status + rebase onto current main | 3/3 rebased clean onto `c863760ae1`; all 3 defects re-confirmed present on main |
| r00 (inline) | Build + save the infera sglang image | `infera-local:sglang-prverify-20260819`, saved to `/data/yihou/images.backup/` (80 GB) |
| r00 (inline) | validate_A/B/C on gfx950 vs current main | **3/3 PASS** |
| r00 (inline) | `probe_host_devptr.py` on gfx950 (#33968 positive control) | **NEGATIVE RESULT — did not reproduce.** See below |
| r00 (inline) | RDMA topology + MVP | rails healthy to n06-25 (330-360 Gb/s); **n01-33 unroutable** |
| [r01](rounds/r01-stock-positive-control/) | Single-node TP4+TP4 PD premise checks | premises confirmed; see below |

## Decisions taken with the user

- **#33968**: stays draft. User is sourcing a machine that reproduces the fault.
- **#33970**: validate via **single-node TP4+TP4 1P1D over loopback RDMA**,
  since the 2-node path is blocked.
- **#33973**: single node, unblocked.
- **#30350 closed unmerged**: TODO only; user will look at it themselves.

## Established environment facts (first-hand, this session)

- n06-33: 8x MI355X gfx950, idle. ROCm 7.2.0, torch 2.9.1, amdgpu **6.14.14**.
- **GID index on this fabric is 1, not 3.** `ionic_0` port 1 exposes only
  gid[0] (link-local) and gid[1] (`::ffff:c0a8:010e` = 192.168.1.14, RoCE v2).
  `MC_GID_INDEX=3` makes mooncake fail with "GID is NULL ... No available RNIC".
  Note `ib_write_bw -x 3` works — perftest and mooncake index GIDs differently,
  so a passing perftest run does NOT validate mooncake's GID setting.
  **The kit's `leg.sh` requires MC_GID_INDEX and the value here must be 1.**
- `RDMAV_FORK_SAFE=1` is needed; without it mooncake logs
  "RDMA context setup failed: fork compatibility: Invalid argument".
- Rails: n06-33 `192.168.N.14/31` routes only to `192.168.N.12` = **n06-25**.
  n01-33 (`192.168.N.70/31`) has no rail route to us and we have none to it
  (ARP INCOMPLETE). Both nodes reach their own switch-side peer, so links are up.
- **Earlier "8/8 rails OK" was a measurement error**: `ping -I <srcIP>` only sets
  the source address; traffic still went over `fenic`. Use `ping -I <ifname>`.

## The #33968 negative result (why it stays draft)

`pr.done.md` predicted gfx950 would measure `same=False` (host VA != device
pointer) — the positive control. Measured here:

```
device: AMD Instinct MI355X gcn=gfx950:sramecc+:xnack-
torch: 2.9.1+rocm7.2.0  hip=7.2.26015
  [pin_memory] / [mmap+hipHostRegister] / [+Mapped] / [+Portable|Mapped]
  -> same=True  for all four
```

Suspected buffer size (the fault report used a 7.33 GB indexer buffer vs the
probe's 8 MiB), so swept 8 MiB / 256 MiB / 1 GiB / 4 GiB / 7.33 GB x 4 strategies:
**all `same=True`**. Size is ruled out.

Only known difference: this box runs amdgpu 6.14.14, which the patch record
attributes to the MI300X *negative* control; the original gfx950 fault report
does not record its driver version. **No mechanism claimed** — the statement that
survives is that this machine cannot reproduce the fault, so it is a negative
control like gfx942, and #33968's write-back evidence stays historical.

`validate_A.py` still PASSes here: it tests equivalence and scope against the
proven local fix, which does not require the fault to reproduce. It also
independently confirms the PR's core argument — stock dispatch for a
`torch.device('cuda:0')` key resolves to `alloc_with_host_register`, i.e. the
two pools that key with a device object do fall through the defaultdict.

---

## r01 — single-node TP4+TP4 PD: premise checks

**Hypothesis.** The #33970 race is between the mooncake transfer *thread* and the
CUDA stream, not between two hosts. If so, a single-node 1P1D with two TP4 legs
over loopback RDMA reproduces it, and the blocked 2-node path is not required.

**Evidence gathered (all first-hand):**

1. **No local/loopback shortcut in mooncake.** `mooncake/conn.py` on current main
   has no `is_local` / `same_host` / `loopback` / `local_transfer` branch, and no
   assertion that prefill and decode are on different hosts. The transfer path is
   the same regardless of peer locality.
2. **The race is thread-vs-stream.** `conn.py:252` starts
   `threading.Thread(target=self.transfer_worker)`; that worker calls
   `engine.batch_transfer_sync` (`:657`, `:1112`). So a CPU thread reads GPU
   memory outside the CUDA stream — exactly what the fix gates with an event.
   Nothing about that depends on the peer being remote.
3. **RDMA loopback works at the verb layer.** `ib_write_bw -R` on one node:
   cross-HCA (ionic_0 -> ionic_4) **348.59 Gb/s**, same-HCA **335.92 Gb/s**.
4. **mooncake itself works loopback** — the decisive check, done with the real
   engine rather than perftest, deliberately without sglang:
   two containers on this host, `transfer_sync_read` of 8 MiB,
   `rc=0`, every byte 0xAB, **RESULT: PASS**.
   (Also logged: "HIP transport installed for intra-node GPU P2P".)
5. **Capacity.** GLM-5.2-MXFP4 is 408 GB; TP4 -> ~102 GB/GPU on 288 GB MI355X.
6. **The kit is parameterizable.** `leg.sh` takes `TP`, `GPUS`, `MY_IP`, `PORT`
   from the environment; `common.sh:start_container` takes `CTR`; containers are
   `--network=host`. Two legs on one node need distinct container names, disjoint
   `HIP_VISIBLE_DEVICES`, and distinct ports. `up.sh` itself is 2-node-shaped and
   will not be used as-is.

**Outcome: premises hold — proceed to build the single-node 1P1D.**

**What this configuration can and cannot establish** (must go in the PR):
- CAN: the correctness claim (needle retrieval degraded -> clean) and the
  `synchronize()` cost, which is the reviewer's obvious first question.
- CANNOT: behaviour under real cross-node RDMA latency. Loopback is *faster*, so
  the race window is *narrower* and reproduction is *harder*. A positive
  reproduction here therefore implies the cross-node case is at least as bad;
  a failure to reproduce here would NOT clear the cross-node case.
- TP4 changes chunked-prefill timing vs the TP8 recipe, so prompt length and
  `--chunked-prefill-size` may need tuning to open the window.

**Discipline for the next rounds:** the positive control comes first. Stock
(unpatched) sglang must reproduce a degraded needle score. Without that, a clean
score on the patched tree proves nothing.
