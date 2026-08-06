# Feature evidence — what was proven, and how

A green run that proves nothing is the default outcome on this stack. Each row
names the *signal*, not the intention. Every command reads from artifacts in this
kit, so it can be re-checked without cluster access.

---

## 1. PD disaggregation — **proven effective**

Router `/health`: `{"status":"ok","active_workers":2}` — 2, not 1. Both workers
registered in etcd with `kv_events` endpoints:

```
registered (http://10.245.156.167:30000, … disagg=DisaggMode.PREFILL, kv_events=tcp://…)
registered (http://10.245.152.164:30001, … disagg=DisaggMode.DECODE,  kv_events=tcp://…)
```

Cross-node smoke test through the router before the run: `17 × 23` → **391**,
`finish_reason: stop`. Prefilled on `…156.167`, decoded on `…152.164`, so the KV
crossed the fabric.

## 2. DP-attention — **proven effective on BOTH legs**

Unlike arm B (where prefill DPA is deliberately off), both legs run dp8 here.
Verified from the resolved `server_args`, and from work actually performed:

| leg | `dp_size` | batches by rank | spread |
|---|---|---|---|
| prefill | **8** | 515 · 495 · 508 · 493 · 508 · 502 · 475 · 499 | ±4 % |
| decode | **8** | 1,818 · 1,802 · 1,887 · 1,873 · 1,707 · 1,776 · 1,760 · 1,776 | ±5 % |

No rank starved, none hot, on either leg. `--ep-size 8` is present on both, held
fixed across arms by the EP_DECOUPLE fix (`../patches/README.md`).

## 3. Round-robin routing — **proven effective, two independent ways**

This is the arm's purpose and it has the strongest evidence in the kit. Full
treatment in [`routing_distribution.md`](routing_distribution.md); the summary:

- **Router pick log**: 16 targets (8 prefill ranks + 8 decode ranks), **290–291
  picks each**. A spread of exactly 1 — the remainder of 2,323 over 8. The two
  pools rotated independently, which is the per-candidate-set counter working as
  designed.
- **Engine log**: prefill batches 475–515 per rank, corroborating from the other
  side that the picks turned into real work.

Contrast with kv-aware on the same branch/image/workload: dp0=1,405, dp1=1,497,
**dp2..dp7 = 1 batch each**.

> `infera_router_picks_total` is **empty** in `results/armA2_router_metrics.txt` —
> HELP/TYPE headers, no samples. That counter is instrumented in the kv-aware path
> only. The pick log is the substitute and is strictly more informative.

## 4. MTP (EAGLE speculative decoding) — **proven effective**

Engine-measured acceptance from the decode log:

```
mean accept len = 2.720   (n = 14,399 samples)
```

Configured `--speculative-num-steps 3 --speculative-eagle-topk 1
--speculative-num-draft-tokens 4`, so the ceiling is 4.00.

**2.720 is the healthy band; 4.00 would be the alarm**, not the ideal — a
saturated acceptance length means the draft model agrees with everything, which in
practice is a repetition loop. TPOT p50 of **16.5 ms** is consistent with
speculation being live (the MTP-off reference on this stack is ~31 ms).

```bash
zcat ../logs/armA2_decode.log.gz | strings | grep -oE 'accept len: [0-9.]+' \
  | awk '{s+=$3;n++} END{printf "mean=%.3f n=%d\n",s/n,n}'
```

## 5. kvd (L3 host KV offload) — **proven EXERCISED, not merely wired**

**This is the strongest kvd evidence in the whole series.** `statctl` before →
after, on the prefill node:

| counter | before | after |
|---|---:|---:|
| `entries` | 0 | **47,975** |
| `host_bytes` | 0 | **84,583,729,152** (84.6 GB) |
| `long_bytes` | 0 | **297,092,570,112** (297 GB) |
| `sets_total` | 0 | **199,085** |
| `gets_total` | 0 | **14,864** |
| `hits_total` | 0 | **14,864** |
| `evictions_total` | 0 | **121,835** |

Three things follow that earlier runs could not show:

1. **Read-back actually happened.** Every prior run on this branch recorded
   `gets_total = 0` — the GPU radix served everything and nothing ever had to come
   back from host. Here there are **14,864 gets**.
2. **Every get hit.** `hits_total == gets_total`, a 100 % hit rate. The tier is not
   thrashing; when the engine asks host for a block, host has it.
3. **Both tiers were live.** `host_bytes` 84.6 GB exceeded the 64 GB `--max-bytes`
   cap, so spillover into the long tier (297 GB) was real, with 121,835 evictions.

Why this arm and not the others: round-robin spreads the shared prefix across 8
ranks instead of concentrating it on 2, so no single rank's GPU radix holds
everything and the L3 tier is genuinely consulted. The routing change is what
exercised the storage tier.

Decode is all-zero **by design** — `_skip_kvd_on_decode_leg` (one of the ten fixes
this branch carries) skips kvd wiring on a PD decode leg, because such a leg sets
`disable_radix_cache=True` and sglang rejects hierarchical cache alongside it.
`infera-kvd adapter connected` appears **8** times in the prefill log, **0** in
decode.

---

## Transport health — the check that is easy to skip

| | prefill | decode |
|---|---|---|
| `MC_FORCE_TCP` occurrences | **0** | **0** |
| `GID is NULL` occurrences | **0** | **0** |

Mooncake falling back to TCP **works** — the run completes, the numbers look
plausible, nothing errors. It is merely slow. Without this grep a silently
degraded run is indistinguishable from a healthy one, and every latency number in
`sli_percentiles.md` would be measuring the wrong thing.

Note these nodes carry **8 ionic rails alongside mlx5_0** (see
`../environment.md`), all ACTIVE. Only mlx5_0 was used, pinned three ways:
`MC_MS_AUTO_DISC=0`, `MC_MS_FILTERS=mlx5_0`, `--disaggregation-ib-device mlx5_0`.

## Errors — client-side timeouts, not server failures

17 errors against 2,323 requests, classified from the driver log by raw-byte
phrase grep:

| phrase | count |
|---|---|
| `timed out` | **16** |
| `failed: HTTP <code>` | **0** |

(The 17th is counted by the driver at a point the log phrase-grep does not reach;
no HTTP failure appears anywhere.) All are `asyncio.TimeoutError` against the
driver's own `aiohttp.ClientTimeout(total=240)` ceiling. At a p99 output length of
~11K tokens and TPOT p99 of 38 ms, a single long generation legitimately exceeds
240 s.

> **Grep the driver log by phrase with `-a`, not by line.** The progress bar uses
> `\r` overwrite, so error prints land on the *same physical line* as the bar and
> line-oriented `grep`/`tail` appear to show no errors at all.

## Engine faults

`Memory access fault` / `HSA_STATUS_ERROR` / `Fatal Python error` / `Traceback` /
`Scheduler hit an exception`, across each full leg log:

| | count |
|---|---|
| prefill | **0** |
| decode | **0** |

Zero across the entire 67-minute window plus boot — on the arm whose first attempt
died in 60 seconds. (That attempt's log did not survive; see `../logs/README.md`.)
