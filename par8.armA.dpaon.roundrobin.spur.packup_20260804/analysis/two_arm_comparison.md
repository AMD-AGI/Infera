# The two arms, side by side

Both arms ran the **par8 workload byte-identical** (md5
`968b1543155839135dc9eaf6dd142626`; only the tokenizer path is retargeted to this
cluster). Everything below is read out of each arm's `results/summary.json`.

| | arm A (this kit) | arm B (`../../par8.armB.dpaoff.kvaware.spur.packup_20260804/`) |
|---|---|---|
| prefill DP-attention | **on** (dp8) | off (pure TP8) |
| router policy | **round-robin** | kv-aware (pw 20.0 / dw 2.0) |
| prefill `mem-fraction-static` | 0.70 (forced by round-robin) | 0.70 (forced by DPA-off) |
| prefill `chunked_prefill_size` | 65,536 global → **8,192/rank** | 65,536 global → **65,536** (no division) |
| decode | dp8 + MTP, 0.85 | dp8 + MTP, 0.85 |
| ran | 2026-08-04 03:16–04:23 UTC | 2026-08-03 12:44–13:51 UTC |
| nodes | crsuse2-m2m-010 / -081 | crsuse2-m2m-250 / -251 |

## Measured

| metric | **arm A** DPA on + round-robin | **arm B** DPA off + kv-aware |
|---|---|---|
| duration | 4006.7 s | 4007.7 s |
| sent / completed | 2,323 / 2,289 | 2,907 / 2,861 |
| **success rate** | **0.9854** | 0.9842 |
| errors (all client timeouts) | 17 | 25 |
| **achieved QPS** | 0.5798 | **0.7253** |
| **TTFT p50** (sustain) | 3,504 ms | **2,239 ms** |
| **TTFT p90** (sustain) | 7,079 ms | **6,389 ms** |
| TTFT p99 (sustain) | 16,602 ms | **10,606 ms** |
| TPOT p50 | 16.5 ms | 16.0 ms |
| TPOT p99 | 38.3 ms | 35.0 ms |
| cache actual / ideal | 0.8819 / 0.8899 | 0.8865 / 0.8899 |
| cache efficiency | 0.9911 | **0.9962** |
| input p50 / p90 / p99 | 75,668 / 155,601 / 237,743 | 74,853 / 154,176 / 225,679 |
| output p50 / p90 / p99 | 322 / 2,714 / 10,508 | 314 / 2,698 / 10,249 |

## ⚠ Two variables move at once — no attribution is available

The arms differ in **prefill DP-attention** *and* **routing policy**, because that
is what was specified. Arm B is ahead on every latency percentile and carries 25 %
more throughput, but that gap is the **combined** effect of the two changes and
**this data cannot split it**. Anyone quoting "DPA off is 1.6× faster" or
"kv-aware beats round-robin" from this pair is misreading it.

The 2×2 has two empty cells:

| | round-robin | kv-aware |
|---|---|---|
| **DPA on** | **arm A** ✓ | *(empty at par8 load)* |
| **DPA off** | *(empty)* | **arm B** ✓ |

The nearest thing to the top-right cell is the earlier acceptance run of this same
branch and image — DPA on + kv-aware — but at **Case A** load (32/128/48), not
par8's (8/32/24), so it is not comparable as a cell of this table.

### The one place the DPA variable IS isolated

`../../agenticbench.mtp.nodpa.packup_20260802/` measured prefill DP-attention on
vs off as a **single-variable** change at concurrency 1 on this cluster:
**1.65–1.93× TTFT**, bin-matched across input sizes, with an 8× chunk-size control
arm proving chunk contributed nothing (0.98–1.06×).

Direction consistent with the gap here. **Magnitude not transferable** — that run
is at 1/24th the concurrency, and DP-attention exists precisely to pack
concurrent requests, so its cost/benefit at N=1 is the worst case for it.

## Where the arms genuinely differ in kind, not degree

| | arm A | arm B |
|---|---|---|
| prefill ranks doing work | **all 8** (475–515 batches each) | **1** (`TP0`, 2,513 batches — dp_size=1 by design) |
| router steering observable? | **yes** — 16 targets, 290–291 picks each | **no** — 1 prefill target exists, so there is nothing to choose between |
| kvd `gets_total` | **14,864** (all hits) | not captured (node reclaimed) |
| kvd tier exercised | **host 84.6 GB + long 297 GB**, 121,835 evictions | wiring proven (`adapter connected: 8`), delta unmeasured |

The kvd row is the clearest asymmetry, and it is **caused by** the routing
difference rather than being independent of it: spreading the shared prefix across
8 ranks means no single rank's GPU radix holds the whole hot path, so the L3 host
tier is actually consulted. See `feature_evidence.md` §5.

## Evidence completeness — the kits are not equal

| | arm A | arm B |
|---|---|---|
| `summary.json` + metrics | ✓ | ✓ |
| kvd before **and after** | ✓ | before only |
| router pick log | ✓ | ✗ (container died with the node) |
| `collect_env.sh` snapshots | ✓ both nodes | ✗ (reconstructed from logs) |
| failed-attempt log | ✗ (overwritten by its own restart) | n/a |

Arm B's three gaps all trace to one cause: its allocations were reclaimed at the
wall clock ~11 h after the run, before the after-state was pulled. Arm A's
`REPRODUCE.md` step 11 exists to prevent a repeat, and this arm followed it.
