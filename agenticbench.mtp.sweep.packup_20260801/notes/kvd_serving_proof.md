# kvd is SERVING, not merely wired — proven, after one false negative

**Result:** `gets 0 → 11,250`, `hits 0 → 11,250`, `misses 0`, **`sets` flat**.
The read count matches the replayed prompt volume exactly.

## Why this test at all

A latency win proves nothing about kvd. SGLang's **in-GPU radix cache** serves a
repeated prefix without ever consulting L3 — which is exactly why the steady-state
counters read tens of thousands of `sets` and **zero** `gets`. The only clean
attribution is to empty the tier above L3 and replay.

## The false negative, and what it actually measured

The first attempt reported the opposite and looked conclusive:

    counter        warm     after "restart"   after replay   delta
    sets_total   22,048          22,048            37,829    +15,781
    gets_total        0               0                 0         +0
    hits_total        0               0                 0         +0

Replay took the same ~18 s as the cold warm-up, and the engine **re-stored**
every page. That reads as "L3 is write-only".

It was wrong, and the give-away is `misses_total: 0` **with** `gets_total: 0`.
A key mismatch would show up as *misses*, not as silence. Zero of both means kvd
was never asked — the query never left the engine — so no amount of reasoning
about hash keys, `PYTHONHASHSEED`, or the bigram view could explain it.

**The defect was in the test, not the deployment.** `boot.sh` restarts the
*engine process* inside a container that keeps running. That empties the GPU
tier, but the replay then re-populated it before any of the five prompts came
back — so nothing ever needed L3. The predecessor kit's procedure works because
it polls VRAM to 0 % and replays *byte-identical* prompts; my resumed run
satisfied the first condition and not, in effect, the second.

## The instrumented round that settled it

Rather than guess among the four early returns in
`hiradix_cache.py::prefetch_from_storage`, all four were instrumented at once
(`probe_prefetch.py` wraps the method, logs the deciding values, forwards to the
original — no behaviour change):

| # | early return | measured |
|---|---|---|
| 1 | `not self.enable_storage` | `enable_storage=True` — not this |
| 2 | `prefetch_length < prefetch_threshold` | **fires on the short requests** (`0 < 64`) |
| 3 | `cache_controller.prefetch_rate_limited()` | `rate_limited=False`, `occupied=0`, `capacity=356,128` — not this |
| 4 | `host_indices is None` | `host_avail=712,256` of `712,256` — not this |

Two populations, cleanly separated:

    prefetch_len=0       ntok=0    x2     short/administrative requests
    prefetch_len=0       ntok=3    x8     short requests -> EXIT below_threshold
    prefetch_len=120000  ntok=120022 x3   -> "PROCEEDS to storage query"

The three real prompts sailed past the threshold with `#cached-token: 0` at that
moment, proceeded to the storage query, and kvd's counters moved.

**A hypothesis this refuted.** Infera's own guard (`hicache_validate.py`)
documents `prefetch_capacity_limit = 0.8 * (host_pool - device_pool)`, which
would collapse to **0** here (host 712,256 tokens vs device 3,260,992) — a
perfect-looking explanation, and SGLang even prints
`host KV pool ... is smaller than the device pool; L2 cache effectiveness is
reduced`. But **this** sglang build computes `int(0.5 * mem_pool_host.size)`
(`cache_controller.py:467`), measured as `capacity=356,128`, not 0. The comment
describes a different version. The guard would also never have fired anyway: it
returns early whenever `--hicache-size` is set, which it is. Measured, not
assumed.

## The proof

kvd counters across the replay of 3 × 120,000-token prompts:

| counter | before | after | delta |
|---|---:|---:|---:|
| `gets_total` | 0 | **11,250** | **+11,250** |
| `hits_total` | 0 | **11,250** | **+11,250** |
| `misses_total` | 0 | 0 | 0 |
| `sets_total` | 40,798 | 40,798 | **0 during the replay** |

`sets` staying put is the load-bearing part: **reads, not re-writes**.

And the count is not merely non-zero — it is the *right* number:

    360,000 tokens replayed  ÷ 64 (page_size)      =  5,625 pages
    × 2 pools (KV + INDEXER)                       = 11,250
    observed gets                                  = 11,250   ✓

The ×2 is corroborated independently by the engine's own line:
`Attached hybrid DSA pool stack to HiRadixCache: pools=KV + INDEXER,
transfer_layer_num=78`. A cache returning pages for the wrong reason could not
land on that identity, and 100 % hits with 0 misses says every page asked for
was present.

## Caveats, stated

- The `sets_total` delta of **+2,969** between the two snapshots occurred during
  the engine **boot** that preceded the replay, not during it. The replay itself
  added no stores. Both snapshots are kept
  (`kvd_after_replay.json`, `kvd_after_probe_replay.json`).
- This proves the L3 **read path** end-to-end at 120K-token scale. It does not
  measure a performance benefit, and none is claimed — see the Case A limitation
  about there being no kvd-off A/B.
- The probe is a debug instrument and is **removed** before any measured window
  (`probe_prefetch.py uninstall`), since it logs on a hot path.

Raw: `logs/restart_replay.log`, `logs/restart_replay_resume.log`,
`logs/probe_prefill.log.gz`, `results/kvd_*.json`.
