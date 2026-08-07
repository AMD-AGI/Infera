# Patches

One patch, **carried unchanged from Phase 2**
(`solo.glm52.mix.packup_20260806/patches/`) because this run used the same patched
driver on the same unrestarted deployment. It is copied here rather than
cross-referenced so this packup stands alone; the byte content is identical.

The full rationale — including the upstream-source analysis of *why* the two new
arrays are needed — lives in
[`../../solo.glm52.mix.packup_20260806/patches/README.md`](../../solo.glm52.mix.packup_20260806/patches/README.md).
Everything a cold reader needs to **apply and verify** it is restated below.

---

## `solo_m1_per_request_e2e_tpot.patch` — SOLO_M1

**Target:** `Optimus-AgenticBench`, branch `fix/realistic-profile-session-driver`
@ `1cf01cbf169d9370a0bc8fe574055c5e975d1be9`, file `agent/agent_throughput.py`.
33 changed lines across 5 hunks.

**Status:** staged-only. Applied to the copy at
`/mnt/vast/c_huggingface/bench_20260801/agbench`. **Never committed upstream.**

### What it changes

| # | change |
|---|---|
| 1 | Adds `RequestMetrics.actual_e2es` — per-request end-to-end latency, **seconds**. |
| 2 | Adds `RequestMetrics.actual_tpots_aligned` — TPOT with **one entry per request**, `0.0` where the sample was filtered. |
| 3 | `add_prefill()` takes a new `e2e: float = 0.0` kwarg and appends to both new lists on every call. |
| 4 | The per-tick emitter slices `actual_e2es` and `actual_tpots_aligned` with the **same bounds** as `new_ttfts`, and emits them as `new_e2es` / `new_tpots`. |
| 5 | The one real call site passes `e2e=total_time`. |

### Why it is load-bearing for THIS phase

Upstream records neither per-request end-to-end latency nor an index-aligned TPOT
array. Two consequences specific to this loaded run:

1. **Every E2E number in this packup exists only because of it** — the sustain p50
   13,931.9 ms, p90 70,313.9 ms, p99 188,236.2 ms.
2. **The censoring observation could not be made without it.** `notes.md` §4 shows
   the maximum *completed* request at **239.0 s** against the driver's own 240 s
   client budget, with **zero** completed requests above it. That is what
   establishes the observed E2E distribution is censored by construction, and it
   requires the raw per-request array — a percentile summary would not show it.

Upstream's own `actual_tpots` is appended only when
`actual_gen_length > 1 and generation_time >= MIN_GENERATION_TIME`, while every
other per-request list is appended unconditionally. Slicing it with the `new_ttfts`
cursor therefore **silently misaligns** — request *i*'s TTFT paired with a later
request's TPOT, plausible-looking and wrong. The patch leaves `actual_tpots`
untouched and adds a parallel, always-appended array.

**The `0.0` convention.** `actual_tpots_aligned` writes `0.0` where the upstream
gate rejected the sample. That is a **marker, not a value**.
`scripts/analyze_solo.py` **drops** those entries rather than averaging them in.
If you write your own analysis, drop them too.

### How to apply it

```bash
git clone <Optimus-AgenticBench> agbench && cd agbench
git checkout 1cf01cbf169d9370a0bc8fe574055c5e975d1be9
patch -p1 --dry-run < <this packup>/patches/solo_m1_per_request_e2e_tpot.patch  # verify first
patch -p1          < <this packup>/patches/solo_m1_per_request_e2e_tpot.patch
```

**The baseline is provable, not asserted.** The staged tree preserves the pristine
file beside the patched one:

| file | md5 |
|---|---|
| `agent_throughput.py.orig` (staged) | `2aa74d1d983984c1b53a3f27d51ebbaa` |
| `agent/agent_throughput.py` in the pristine checkout @ `1cf01cb` | `2aa74d1d983984c1b53a3f27d51ebbaa` — **identical** |
| `agent_throughput.py` (staged, patched) | `8f482b8fba8ba69d02c767a3618d1a36` |

Because `.orig` byte-matches the upstream commit, `diff -u .orig <patched>` is
exactly this patch and nothing else — no accumulated local drift.

### How to verify it took

The failure mode is **not** a crash — it is that the E2E column silently does not
exist:

```bash
python3 -c "
import json
r=json.loads(open('<results>/metrics.jsonl').readline())
print('SOLO_M1 present' if 'new_e2es' in r and 'new_tpots' in r else 'PATCH MISSING')"
```

Against this packup's own shipped result (gunzip
`results/load/metrics.jsonl.gz` first) it prints `SOLO_M1 present`.

### Upstreamability

The patch is additive: two new fields, two new emitted keys, no change to existing
behaviour or output. It has **not** been proposed upstream, and no upstream issue
or PR was searched for — stated as a gap, not a judgement about whether it should
be.
