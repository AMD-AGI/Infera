# The two config defects that cost three attempts — what / why / how / context

Case A was attempted **three times** on this cluster. Attempts 1 and 2 died on
the prefill leg; attempt 3 completed the full 4,007 s window. Nothing about the
model, the fabric, or the workload changed between them. Both defects were
**already documented in the vultr sibling kit**
(`caseA.glm52.fullfeature.packup_20260801`) before attempt 1 was launched.

---

## Defect 1 — prefill `--mem-fraction-static` 0.88 (load-bearing)

### What

```
rocdevice.cpp:3582 ... HSA_STATUS_ERROR_OUT_OF_RESOURCES: Code: 0x1008
                       Available Free mem : 78 MB
Fatal Python error: Aborted
```

DP0 aborts; the remaining 7 ranks hang until `watchdog_timeout=3600` fires 87
minutes later. Downstream, the decode leg's 8 ranks go silent and the prefill
ranks log `TCPStore recvValue failed` — **both are consequences, not the first
failure.**

### Why

It is **not** KV exhaustion. The scheduler lines immediately before the abort
read `token usage: 0.01–0.05` — the KV pool is essentially empty. It is
**DP-attention runtime activation memory**: at dp8 every rank holds its own
8,192-token chunk activations, and a 155K-token prompt is 19 chunks. Under the
prefill delayer's batching the transient peak exceeds what
`1 − mem_fraction_static` leaves outside the static reservation.

### How

Lower it. `0.88 → 0.80` on prefill only; decode stays 0.85.

| | 0.88 | 0.80 |
|---|---:|---:|
| per-rank free after init | 32.9–33.8 GB | **284.0 GB** |
| KV pool (tokens/rank) | 3,260,992 | 2,939,264 (−10 %) |

**The direction is the counter-intuitive part**, and getting it backwards costs a
run. Diagnose by phase:

| phase | symptom | fix |
|---|---|---|
| decode | retract / `get_cpu_copy NotImplementedError` | **raise** mem-fraction-static |
| prefill | `HSA_STATUS_ERROR_OUT_OF_RESOURCES` / `Aborted` | **lower** it |

### Context — three wrong root causes before the right one

Recorded because the failure pattern is instructive, not to be self-flagellating:

1. **"prefill `TCPStore recvValue failed` is the first failure."** Wrong — I read
   `g1_prefill.log` (a gate round) instead of `probe_prefill.log` (what the
   attempt actually wrote). The HSA abort precedes TCPStore by 3 seconds.
2. **"the direct cause is the missing `--disable-custom-all-reduce`."** Wrong,
   though the flag *was* genuinely missing via a two-variable trap
   (`CUSTOM_AR="${CUSTOM_AR:-$([ "$MTP" = "1" ] && echo 0 || echo 1)}"` → MTP=0
   on prefill → custom AR silently re-enabled). Fixing it freed 0.85 GB/rank and
   **attempt 2 crashed identically.** A contributing consumer, not the cause.
3. **"the `fp8_mqa_logits` quadratic `[chunk, seq_len_kv]` fp32 buffer."** Wrong
   — the Phase-1 sweep ran ISL 155K (a 4.73 GiB buffer) 8/8 clean, while the
   attempt-2 crash occurred on a 120K needle (3.66 GiB, *smaller*). Buffer size
   alone does not separate pass from fail.

The actual discriminator is total activation headroom, which is what
`mem-fraction-static` sets — and which the vultr kit had already measured.

**Upstream status.** Not filed. This is a deployment tuning value, not a defect.

---

## Defect 2 — `GLM52_P1V3`, the DSA indexer IDLE-rank row underflow (load-bearing)

### What

```
RuntimeError: Expected lengths.size(0) == B to be true, but got false.
```

in the DSA indexer, reached through the **MTP draft model** (`deepseek_nextn`).
Kills a decode scheduler rank; the router drops to `active_workers: 1`.

**Not observed on this cluster** — because the patch was applied *before* attempt
3, on the strength of the vultr evidence. Vultr reproduced it twice, at 125 s and
766 s into two independent Case A runs.

### Why

The image already ships `GLM52_P1V2`, which trims DP-attention padding. It guards
only one direction:

```python
_p1v2_trim = _p1v2_real < _p1v2_padded     # assumes padding makes q_fp8 LONGER
```

On a DP-attention **IDLE** rank under MTP draft-extend the inequality inverts.
Captured live on vultr with the image's own `SGLANG_DEBUG_DSA_ROWS=1`:

```
mode=IDLE q_fp8=(1,32,128) q_offset=2 ntnp=0 agree=False lengths=(2,) -> mqa_q=(1,32,128)
```

`q_offset` (2) *exceeds* the rows present (1), so no trim runs, and
`fast_topk_v2` gets 1 score row against 2 lengths entries. A trim cannot fix it —
there are *fewer* query rows than lengths entries, so both sides must be
reconciled down to `min(real, padded)`.

### How

`patches/apply_p1v3.py`, run inside the **decode** container. Three edits in
`_get_topk_paged`. Idempotent, anchors on exact source text, fails loudly if the
image differs.

**Applicability was verified by digest, not assumed:** this cluster's in-image
`dsa_indexer.py` md5 is `632f17acd38737459b43f830ee60ee89` — byte-identical to
vultr's pre-patch file.

Verify the **loaded** module, not the file on disk:

```bash
docker exec agbench_mtp python3 -c "
import sglang.srt.layers.attention.dsa.dsa_indexer as m, inspect
print(inspect.getsource(m).count('GLM52_P1V3'))"   # want 3
```

### Context — why the fixed-length gates never caught it

The bug needs MTP **and** DP-attention **and** an idle rank simultaneously.
Phase 1's 8-round sweep ran MTP + DPA for 660 requests and never hit it, because
`--random-range-ratio 1.0` makes every request in a round the same length —
batch shapes stay homogeneous and ranks rarely go idle mid-flight. Case A's
breathing session population produces ragged batches constantly.

This is the general lesson from all three attempts: **a fixed-length sweep at
higher ISL and higher concurrency is not a superset of an agentic workload.** It
is a different axis. The sweep proves peak throughput and correctness; only the
agentic profile exercises ragged batches, prefix reuse, and idle ranks.

**Upstream status.** Not filed. It should be — the image already ships the
instrumentation that proves the bug (`SGLANG_DEBUG_DSA_ROWS`, `dsa_indexer.py:63`)
and a comment conceding the two bookkeeping sources were never verified to agree
on this path.

---

## Defect 3 — kvd `--long-bytes` (NOT applicable here)

Vultr's third patch lowers kvd's L3 spill budget 512 G → 64 G because
`--long-path /tmp/kvd-long` lands on the container's writable layer, and their
node's root disk was smaller than the budget.

**Checked and deliberately not applied on spur.** Here `/` inside the container
is an overlay on `/mnt/m2m_nobackup` with 23 TB free; `/tmp/kvd-long` held 514 GB
during the run with no pressure. Copying the patch would have been cargo-culting.

---

## Things this run deliberately did NOT change

Recorded so the next person does not re-derive them:

- **`new_session_rate` stays 0.10.** An earlier spur attempt raised it to 0.115
  on a linear-scaling assumption; it pinned `max_inflight` and had to be aborted.
  Vultr's Step-3 re-solve landed 0.0941 — inside noise of the shipped value.
- **`--chunked-prefill-size` stays 65536** (8,192/rank at dp8). This was a
  candidate mitigation for the OOM before the real cause was found; once
  headroom went to 284 GB it was unnecessary, and changing it would have made the
  fix a two-variable change.
- **`--context-length` stays 262144.** Dropping to 131,072 would have "fixed" the
  OOM too — by truncating **15.4 %** of the input distribution, i.e. by measuring
  a different workload. (Measured on the reference run's raw samples: 428 of
  2,781 prompts exceed 131,072; p90 alone is 152,264.) The MTP-off reference run
  also used 262144 — an earlier draft of this kit said otherwise and was wrong.
