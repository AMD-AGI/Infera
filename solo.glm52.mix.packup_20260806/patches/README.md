# Patches

One patch. It is **load-bearing** — the E2E column in the results table cannot be
measured without it.

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
| 1 | Adds `RequestMetrics.actual_e2es` — per-request end-to-end latency, seconds. |
| 2 | Adds `RequestMetrics.actual_tpots_aligned` — TPOT with **one entry per request**, `0.0` where the sample was filtered. |
| 3 | `add_prefill()` takes a new `e2e: float = 0.0` kwarg and appends to both new lists on every call. |
| 4 | The per-tick emitter slices `actual_e2es` and `actual_tpots_aligned` with the **same bounds** as `new_ttfts`, and emits them as `new_e2es` / `new_tpots`. |
| 5 | The one real call site passes `e2e=total_time`. |

### Why it was needed

Upstream records **neither** of the two things a concurrency-1 latency
measurement is about:

1. **No per-request end-to-end latency at all.** Nothing in the upstream metrics
   object holds it. Without this patch, E2E has to be *back-solved* from TTFT and
   TPOT — an estimate, not a measurement, and one that silently absorbs any
   overhead that is neither TTFT nor per-token time.

2. **`actual_tpots` is filtered and therefore NOT index-aligned.** Upstream only
   appends to it when the sample passes a quality gate:

   ```python
   if actual_gen_length > 1 and generation_time >= MIN_GENERATION_TIME:
       self.actual_tpots.append(generation_time / (actual_gen_length - 1))
   ```

   Every other per-request list is appended unconditionally. So `actual_tpots` is
   **shorter** than `actual_ttfts`, and slicing it with the same cursor as
   `new_ttfts` **silently misaligns** — request *i*'s TTFT gets paired with some
   later request's TPOT. It produces plausible-looking numbers and no error.

   The patch keeps `actual_tpots` untouched (nothing that reads it changes
   behaviour) and adds a parallel, always-appended array.

**The `0.0` convention.** `actual_tpots_aligned` writes `0.0` where the upstream
gate rejected the sample. That is a **marker, not a value**.
`scripts/analyze_solo.py` **drops** those entries rather than averaging them in —
treating them as zero-latency tokens would drag the mean toward zero and make TPOT
look better than it is. If you write your own analysis, drop them too.

### How it was applied

Copy-and-patch on the shared mount. The pristine file was preserved beside the
patched one as `agent/agent_throughput.py.orig`, which is what makes the patch
provable rather than asserted:

| file | md5 |
|---|---|
| `agent_throughput.py.orig` (staged) | `2aa74d1d983984c1b53a3f27d51ebbaa` |
| `agent/agent_throughput.py` in the pristine checkout @ `1cf01cb` | `2aa74d1d983984c1b53a3f27d51ebbaa` — **identical** |
| `agent_throughput.py` (staged, patched) | `8f482b8fba8ba69d02c767a3618d1a36` |

Because `.orig` byte-matches the upstream commit, `diff -u .orig <patched>` is
**exactly** this patch and nothing else — no accumulated local drift.

To apply it yourself:

```bash
cd <Optimus-AgenticBench checkout at 1cf01cb>
patch -p1 --dry-run < solo_m1_per_request_e2e_tpot.patch   # verify first
patch -p1          < solo_m1_per_request_e2e_tpot.patch
```

Verified while assembling this packup: the dry run applies cleanly against the
pristine checkout.

### Context — the symptom it cures

Without SOLO_M1, `metrics.jsonl` carries `new_ttfts` but no `new_e2es` and no
index-aligned TPOT. The failure is **not** a crash or an error — it is that the
E2E column of the results table does not exist, and any attempt to reconstruct it
by pairing TTFT with `actual_tpots` is quietly wrong for every request after the
first filtered one.

Check whether the patch is live in any result directory:

```bash
python3 -c "
import json
r=json.loads(open('results/solo/p50/metrics.jsonl').readline())
print('SOLO_M1 present' if 'new_e2es' in r and 'new_tpots' in r else 'PATCH MISSING')"
```

### Upstreamability

The patch is additive: it introduces two new fields and two new emitted keys, and
changes no existing behaviour or output. Nothing that reads `actual_tpots` is
touched. It has **not** been proposed upstream, and no upstream issue or PR was
searched for while assembling this packup — stated as a gap, not a judgement
about whether it should be.
