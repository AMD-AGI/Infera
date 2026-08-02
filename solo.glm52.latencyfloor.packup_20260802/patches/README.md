# Patches

Two, and they sit at different layers. **Both are load-bearing, for different
reasons:** without 0004 the run does not complete; without 0005 it completes and
measures nothing useful.

| | target | applied where | without it |
|---|---|---|---|
| **0004** `GLM52_P1V3` | engine, inside the decode container | runtime, per boot | decode leg **crashes** within minutes |
| **0005** `SOLO_M1` | driver, on the jump host | once, to the staged repo | **no E2E / TPOT ladders** |

---

## 0004 — `dsa_indexer.py`: handle the REVERSED padding case (`GLM52_P1V3`)

**Inherited from Case A, unchanged.** Full write-up:
`../caseA.glm52.fullfeature.packup_20260801/patches/README.md` and
`../notes/notes.dsa.mtp.crash.md`.

**Applied by** `../scripts/apply_p1v3.py`, run inside the **decode** container.
Already live on leg `p6` when this run started; verified `P1V3: 3` in the loaded
module immediately before the window opened.

**One-line why.** The image's own `GLM52_P1V2` trim guards only
`real < padded`. On a DP-attention **IDLE** rank under MTP draft-extend the
inequality inverts (`q_fp8` has 1 row against `lengths` of 2), no trim runs, and
`fast_topk_v2` raises `Expected lengths.size(0) == B`. P1V3 reconciles both sides
to `min(real, padded)` and clips `lengths` through the existing `ke_offset`
extension point.

**Relevance to this run.** The solo workload draws from the same output
distribution as Case A and runs against the same MTP-enabled decode leg, so it
sits on exactly the same code path. Stock `merged-e` cannot produce these numbers.

---

## 0005 — `agent_throughput.py`: persist per-request E2E and TPOT (`SOLO_M1`)

**Applied by** `../scripts/apply_solo_metrics.py` on the **jump host** — the
driver runs there, not in a container. Idempotent, anchors on exact source text,
`sys.exit(2)`s loudly if any anchor is missing. Not a `diff -u` because it is
applied to a staged working copy whose provenance is a git SHA plus this patch.

**Target:** `$W/agbench/agent/agent_throughput.py`
(Optimus-AgenticBench @ `1cf01cb`; md5 before patch
`2aa74d1d983984c1b53a3f27d51ebbaa`; original preserved at
`agent_throughput.py.orig`).

### What

Four edits, all additive — no existing value is altered:

1. `BenchMetrics` gains `actual_e2es` and `actual_tpots_aligned`.
2. `add_prefill()` takes an optional `e2e=` kwarg and appends to both lists.
3. The realistic-mode call site passes `e2e=total_time`.
4. `save_metrics_loop` emits `new_e2es` and `new_tpots`, sliced with the **same**
   cursor as `new_ttfts` so the arrays concatenate row-wise.

The two other `add_prefill` call sites (`run_replay`, dataset replay) are
untouched and take the `e2e=0.0` default — they are not on this run's path.

### Why

The driver discards both quantities:

- **E2E is never recorded.** `total_time` is computed at line 2310, handed to a
  rate tracker, and dropped. Case A had to *back-solve* E2E from
  `TTFT + (gen−1) × TPOT`.
- **TPOT is recorded but not persisted.** `metrics.actual_tpots` lives in memory
  and is reduced to p50/p90/p99 in the final summary; the samples die with the
  process.

For a throughput study that is tolerable. For a **latency-floor** study those two
*are* the subject, so without this patch a 37-minute run produces output that
cannot answer the question it was run to answer.

### The subtlety that would have produced wrong numbers

`actual_tpots` is **filtered** — appended only when
`gen_len > 1 and gen_time >= MIN_GENERATION_TIME`. It is therefore **not**
index-aligned with `actual_ttfts` / `actual_prompt_lengths`, and slicing it with
`last_distributions_index` would misalign every row after the first filtered
request. That yields plausible, wrong per-request pairings — the worst kind of
bug, because nothing errors.

The patch adds a *separate* `actual_tpots_aligned` that appends on **every**
request, writing `0.0` for filtered samples. Alignment is preserved by
construction; the analyzer drops zeros rather than reading them as fast tokens.

### Verification

Offline, before staging (`/tmp/solotest/stub_test.py`):

| check | result |
|---|---|
| module imports, `py_compile` clean | ✅ |
| idempotent (second apply is a no-op) | ✅ |
| all four per-request lists same length | ✅ |
| filtered request writes `0.0` in aligned, absent from filtered | ✅ |
| unfiltered values match `gen_time/(gen_len−1)` exactly | ✅ |
| untouched call sites default to `e2e=0.0` | ✅ |

On the jump host, in the **loaded** module (not the file on disk — stale
`__pycache__` has invalidated an experiment in this tree before):

```bash
cd $W/agbench && $W/venv/bin/python -c "
import agent.agent_throughput as m, inspect
print('SOLO_M1:', inspect.getsource(m).count('SOLO_M1'))
M = m.BenchMetrics()
print(hasattr(M,'actual_e2es'), hasattr(M,'actual_tpots_aligned'))"
# 8 / True True
```

In the run output: `new_ttfts`, `new_e2es`, `new_tpots` all n=102, row-aligned.

### The cross-check it bought

With a *measured* E2E alongside TTFT and TPOT, the composition identity becomes
falsifiable rather than assumed:

    predicted = TTFT + (gen_len − 1) × TPOT
    predicted − measured = +0.0 ms at every percentile, 102/102 requests

This is a stronger statement than it looks. It says the three quantities share
one time base, and it **retroactively validates the back-solve Case A had to
rely on** — where the same arithmetic could only be sanity-checked against a
second, independent route (duty cycle), and agreed to within 3 %.

### Scope and risk — stated plainly

- This modifies the **measurement instrument**, not the system under test. No
  engine behaviour changes; the server cannot tell the difference.
- Purely additive: every pre-existing field in `metrics.jsonl` and
  `summary.json` keeps its exact prior meaning, so old analysis scripts still
  work on new output.
- The `0.0` sentinel in `new_tpots` is a real ambiguity if consumed naively —
  a genuine TPOT of exactly 0.0 is impossible, so dropping zeros is safe, but
  the convention must be honoured. `scripts/solo_analyze.py` does; it also
  reports how many were filtered.
- The original is preserved at `agent_throughput.py.orig` for a revert-based A/B.

### Upstream status

**Not filed.** Worth filing — persisting per-request E2E is a one-line
improvement to a benchmark driver whose entire purpose is measuring latency, and
the filtered-vs-aligned TPOT hazard is a live trap for anyone who tries to add
it themselves.
