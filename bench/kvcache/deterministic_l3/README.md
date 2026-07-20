# Deterministic L3 (kvd) cache validation bench

## Why this exists

Our previous attempts to validate kvd L3 reads with `agent-bench`
(light-trace-benchmark) failed because the bench chains turns by
feeding the model's *generated* output back into the next turn's
prompt. Model output is not bit-reproducible across vLLM restarts
(sampling kernels diverge at FP rounding), so cross-restart
replay produces *different* prompts → *different* block hashes →
zero kvd hits regardless of how well kvd retained content.

See PD design §6.4 "Validation plan" and the bench writeup at
`/tmp/cache-bench-results/l2-ab/SUMMARY.md` (session-local) for
the postmortem.

## What this bench does differently

The corpus is generated from a **fixed function of (session_id,
turn_idx)**. Every turn's user message is a deterministic
pseudo-random word salad; every assistant slot in the chat
history is a fixed 12-byte string. We send each request with
`max_tokens=1` so the engine's actual generation is irrelevant
— what we measure is prefill TTFT, which is what kvd accelerates.

Bit-identical prompts across runs → identical block hashes →
kvd hits when content is retained.

## Files

- `corpus.py`  — generate a synthetic deterministic JSON corpus
  (growing-prefix word-salad sessions; good for L1-pressure tests).
- `corpus_swebench.py`  — convert the Inferact/codex_swebenchpro_traces
  HF dataset into the same corpus format. Real Codex agent traces;
  ~93.8 % of trials share an ~11.5 K-token system prompt → this is
  the L2-positive workload (RAG/multi-tenant pattern).
- `replay.py`  — async chat replay client; measures TTFT.
- `run_3phase.sh` — orchestrates the three-phase A/B run.

## Running

```bash
# from repo root
chmod +x bench/kvcache/deterministic_l3/run_3phase.sh
OUT=/tmp/l3-bench-results SESSIONS=32 TURNS=8 TURN_TOKENS=6000 \
  bench/kvcache/deterministic_l3/run_3phase.sh
```

Tunables (env vars with defaults):
- `OUT` (`/tmp/l3-bench-results`): artifact dir.
- `CORPUS` (`$OUT/corpus.json`): cache-hit if it already exists.
- `SERVER` (`http://localhost:30000`): engine endpoint.
- `MODEL` (`MiniMax-M2.5`): the `served_model_name`.
- `CONCURRENCY` (`8`): max in-flight requests at the bench layer.
- `SESSIONS` / `TURNS` / `TURN_TOKENS`: corpus shape.
- `LAUNCH_VLLM` (`/tmp/run_vllm_l2.sh`): the vLLM launcher with
  `$1` = `INFERA_KVD_L2_GB`. See parent session for the script.
- `KVD_CONTAINER` (`infera-kvd`).

## How to read the output

The `summary` section at the end prints one row per phase:

```
Phase 1: 256/256 ok  TTFT mean=...  p50=...  p90=...  wall ...s
Phase 2: ...
Phase 3: ...
kvd delta during phase 1: gets+0     sets+N     evict+0   ← should be writing
kvd delta during phase 2: gets+M     sets+0     evict+0   ← should be reading
kvd delta during phase 3: gets+M     sets+0     evict+0   ← same reads, but L2 catches repeats
```

The signal you're looking for:

- **Phase 1**: many sets, zero gets. Confirms the warmup fills kvd.
- **Phase 2**: many gets. Confirms cross-restart replay actually
  hits kvd (the precondition our prior bench failed).
- **Phase 3 vs Phase 2**: gets count similar (workload identical),
  TTFT improves *if and only if* L2 captures any repeated key
  within a single vLLM uptime. With the `replay.py` round-robin
  ordering, each session's growing-prefix path means turns 1..N-1
  re-issue prefix block lookups that turn N also needs — that's
  where L2 fires.

## Sizing for a real validation run

The relevant constraint: the corpus's total KV footprint must
exceed L1 (forces eviction) and fit within kvd's host-pinned
tier (so kvd doesn't thrash). On the MI355X test setup
(`infera-kvd --max-bytes 128G`):

- 32 sessions × 8 turns × 6000 tokens = ~1.5M tokens working set
- @ ~150 KB / token per layer × ~80 layers = enormous; in practice
  the packed-multi-layer blob format compresses per-block to
  ~4 MB, so 1.5M tokens / 16 tokens per block × 4 MB = ~370 GB.
  That overflows 128 GB. Pick smaller numbers.

A working starting point: `SESSIONS=16 TURNS=4 TURN_TOKENS=4000`
gives ~256K tokens working set, ~64 GB packed kvd footprint — fits
comfortably in a 128 GB kvd host tier with headroom for repeats.

## Using the SWE-bench-Pro Codex traces (L2-positive workload)

The synthetic `corpus.py` produces growing-prefix sessions with
*zero* cross-session prefix sharing — L2's behavior on that
workload is correctly "near-zero hits" (PD §6.4). To validate
L2's hit path, use `corpus_swebench.py` which converts
`Inferact/codex_swebenchpro_traces` (610 real Codex agent trials,
~93.8% sharing an ~11.5K-token system prompt) into our corpus
format.

```bash
# One-time download (~210 MB, cached under ~/.cache/huggingface/)
uv run python3 -m bench.kvcache.deterministic_l3.corpus_swebench \
  --num-trials 16 \
  --strategy smallest \
  --max-turns-per-trial 16 \
  --out /tmp/l3-bench-results/swebench-corpus.json

# Then point the 3-phase runner at it
CORPUS=/tmp/l3-bench-results/swebench-corpus.json \
OUT=/tmp/l3-bench-swebench-results \
  bench/kvcache/deterministic_l3/run_3phase.sh
```

Selection strategies:

- `--strategy smallest`: cheap smoke run. Picks the N shortest
  trials by total content bytes. 16 trials × ~80 KB each = ~1.3 MB
  prompts; replay completes in ~5 min on TP=1 MiniMax-M2.5.
- `--strategy largest`: stress test. The largest trial in the
  dataset reaches 9 B input tokens (P99); cap with
  `--max-turns-per-trial` to keep replay tractable.
- `--strategy random` (+ `--seed`): unbiased sample.
- `--strategy first`: dataset natural ordering — deterministic
  without needing a seed.

Expected L2-positive signature (Phase 3 vs Phase 2):

- Phase 2 (L2 OFF): `kvd_gets +N` where N is roughly proportional
  to the number of cross-trial shared-prefix block lookups vLLM
  issues after L1 evicts each shared prefix slot.
- Phase 3 (L2 ON): `kvd_gets +M` where **M < N** because L2 serves
  the second-and-subsequent fetch of every shared kvd_key within
  the vLLM uptime. The gap (N − M) × 1117 µs per kvd UDS GET is
  the absolute time saved by L2 — should also show up as TTFT p50
  improvement.

If Phase 3 and Phase 2 land at the same `kvd_gets` (the L2-null
result documented in PD §6.4), the SWE-bench corpus didn't
trigger enough cross-trial L1 evictions to expose repeats — try
a larger `--num-trials` or `--strategy largest` to push pressure.

Tokenizer note: Codex tokenized the original ~11.5 K shared
prefix under cl100k_base. We replay against MiniMax-M2.5 (or
whatever model is up). MiniMax's tokenizer will produce a
different token *count* for the same bytes, but identical
*byte* prefix → identical MiniMax block hashes → kvd keys still
collide as intended. The "11.5 K cached tokens" number from the
dataset card won't reproduce; the structural shared-prefix
pattern will.

## Caveats

- We send `max_tokens=1`. Throughput-style metrics (TPOT, gen tok/s)
  are *not* meaningful here — only TTFT is. This bench validates
  **prefill cache reuse**, not generation pipeline performance.
- vLLM's `external_prefix_cache_queries_total` / `_hits_total`
  Prometheus counters are the authoritative signal that the
  connector is firing. The CSV captured before/after each phase
  carries them.
- Concurrency at the bench layer (`CONCURRENCY`) is independent
  of the engine's `--max-num-seqs` and `--max-inflight`. Keep it
  ≤ the engine's inflight cap so we don't queue at the engine.
