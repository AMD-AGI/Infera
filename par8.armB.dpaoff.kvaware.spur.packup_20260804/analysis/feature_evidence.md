# Feature evidence — what was actually proven, and what was not

A green run that proves nothing is the default outcome on this stack. Each row
below names the *signal*, not the intention. Where a feature is only proven
*configured* rather than *effective*, that distinction is made explicitly.

Every command here reads from artifacts in this kit (`logs/`, `results/`), so it
can be re-checked without cluster access:

```bash
zcat logs/armB_prefill.log.gz | strings | grep -c 'MC_FORCE_TCP'
```

---

## 1. PD disaggregation — **proven effective**

Router `/health` returned `{"status":"ok","active_workers":2}` — 2, not 1. Both
workers registered in etcd with `kv_events` endpoints:

```
registered (http://10.245.158.155:30000, model=glm5.2-mxfp4, disagg=DisaggMode.PREFILL, kv_events=tcp://…)
registered (http://10.245.151.18:30001, model=glm5.2-mxfp4, disagg=DisaggMode.DECODE,  kv_events=tcp://…)
```

Cross-node smoke test through the router before the run: `17 × 23` → **391**,
`finish_reason: stop`, 10.5 s. The prompt was prefilled on `…158.155` and decoded
on `…151.18`, so the KV crossed the fabric.

## 2. DP-attention — **proven, in the intended asymmetric configuration**

This arm's whole point is prefill DPA **off** while decode keeps it **on**. Both
sides verified from the resolved `server_args`, not from the flags requested:

| leg | `dp_size` | `enable_dp_attention` | scheduler tag in log |
|---|---|---|---|
| prefill | **1** | **False** | `TP0 EP0]` — no `DP` prefix |
| decode | **8** | True | `DP0…DP7 TP0…TP7 EP0…EP7]` |

Per-rank work actually performed:

| | prefill | decode |
|---|---|---|
| batches by rank | `TP0`: **2,513** (single rank, as designed) | DP0 2,338 · DP1 2,244 · DP2 2,294 · DP3 2,296 · DP4 2,186 · DP5 2,233 · DP6 2,053 · DP7 2,030 |

The decode spread is **2,030–2,338**, i.e. ±7 % around the mean across 8 ranks.
No rank is starved and none is hot.

**`--ep-size 8` is present on BOTH legs**, including the DPA-off prefill. That is
the EP_DECOUPLE fix (`patches/README.md` §0001 hunk A) doing its job: the MoE
expert-parallel width is held fixed while only the attention layout changes.

## 3. kv-aware routing — **proven CONFIGURED, not proven EFFECTIVE**

This is the honest limit of this kit.

**What is proven.** The router came up under the kv-aware policy with the product
default weights, read back from its own startup log:

```
router-policy=kv-aware overlap_weight=1 prefill=20.0 decode=2.0
```

**What is NOT proven.** The per-rank pick distribution and the
`infera_router_pick_cache_hits` histogram — the two signals that show the policy
*steering* rather than merely being loaded — were never captured. `/tmp/router.log`
and the `/metrics` endpoint both live inside the prefill container, and the
allocation was reclaimed at the wall clock before they were pulled.

**Why it partly does not apply on this arm anyway.** Even with the metrics in
hand, kv-aware has little to steer here:

- **Prefill**: DPA off ⇒ `dp_size=1` ⇒ `expand_targets()` yields exactly **one**
  target. There is nothing to choose between. This is structural, not a capture
  gap — the vultr par8 kit reached the same conclusion for the same reason
  (2,884/2,884 picks to one target).
- **Decode**: a PD decode leg sets `disable_radix_cache=True` itself
  (confirmed in the resolved args), so its router-side KV view stays empty and
  the cost function's overlap term cancels. Routing degenerates to least-loaded.

So on **this** arm, "kv-aware" names the policy in force; it does not describe an
observable routing behaviour. The arm where it *would* have been observable is
arm A (DPA on ⇒ 8 prefill targets) — and arm A is precisely the arm that did not
complete. See `../notes.md`.

## 4. MTP (EAGLE speculative decoding) — **proven effective**

Engine-measured acceptance from the decode log:

```
mean accept len = 2.789   (n = 17,674 samples)
```

Configured with `--speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4`, so the ceiling is 4.00.

**2.789 is the healthy band. 4.00 would be the alarm**, not the ideal: a
saturated acceptance length means the draft model is agreeing with everything,
which in practice is a repetition loop, not perfect speculation. The vultr par8
sibling recorded 2.02 per-request against an engine mean in the same band.

TPOT p50 of **16.0 ms** is consistent with speculation being live — the MTP-off
reference on this stack is ~31 ms.

## 5. kvd (L3 host KV offload) — **proven wired, delta NOT measured**

`infera-kvd adapter connected` appears **8 times** in the prefill log — once per
DP rank of the hicache pool — and **0 times** in the decode log.

The decode zero is **by design**, not a failure: `_skip_kvd_on_decode_leg` (one of
the ten fixes this branch carries) deliberately skips kvd wiring on a PD decode
leg, because such a leg sets `disable_radix_cache=True` and sglang rejects
hierarchical cache alongside it.

Resolved prefill args confirm the chain end to end:

```
--enable-hierarchical-cache  --hicache-size 32  --hicache-storage-backend infera-kvd
```

`--hicache-size 32` is **absolute GB**, deliberately never `--hicache-ratio`: the
ratio default once sized the host pool at 355 GB *per DP rank*, and a TB-scale
pinned host allocation can wedge a spur node at kernel level.

**Not measured**: the `statctl` counter delta. The all-zero *before* baseline is
in `results/armB_prefill.kvd_before.json`; the *after* snapshot could not be taken
(same wall-clock reclaim). So write-back is proven *connected*, not proven
*exercised*, on this arm. The acceptance run on this same branch and image did
measure it: `entries 0 → 12,942`, `long_bytes 0 → 22.9 GB`.

---

## Transport health — the check that is easy to skip

| | prefill | decode |
|---|---|---|
| `MC_FORCE_TCP` occurrences | **0** | **0** |
| `GID is NULL` occurrences | **0** | **0** |

This matters more than it looks. Mooncake falling back to TCP **works** — the run
completes, the numbers look plausible, nothing errors. It is merely slow. A run
that silently degraded to TCP would be indistinguishable from a healthy one
without this grep, and every latency number in `sli_percentiles.md` would be
measuring the wrong thing.

## Engine faults

`Memory access fault` / `HSA_STATUS_ERROR` / `Fatal Python error` / `Traceback` /
`Scheduler hit an exception`, counted across each full leg log:

| | count |
|---|---|
| prefill | **0** |
| decode | **0** |

Zero across the entire 67-minute window plus boot.
