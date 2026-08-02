#!/usr/bin/env python3
"""SOLO_M1 -- make per-request E2E and TPOT survive to disk.

WHY THIS EXISTS
---------------
The solo (concurrency=1) run measures the *latency floor* of the service. Its
two headline quantities are end-to-end request latency and time-per-output-token
-- and the driver persists neither per-request:

  * E2E is never recorded at all. `total_time` is computed at
    agent_throughput.py:2310, used for the RateTracker, and dropped. Case A's
    analysis had to BACK-SOLVE E2E from TTFT + gen_len x TPOT. That is fine for
    a throughput study and unacceptable for a latency study.
  * TPOT lands in `metrics.actual_tpots`, but `save_metrics_loop` only writes
    `new_ttfts`; the TPOT list is reduced to p50/p90/p99 in the final summary
    and the samples are lost with the process.

So a solo run against the stock driver could not produce an E2E ladder or a TPOT
ladder -- exactly the two ladders it exists to produce.

WHAT IT CHANGES  (purely additive; no existing value is altered)
----------------------------------------------------------------
  1. BenchMetrics gains `actual_e2es` and `actual_tpots_aligned`.
     `actual_tpots` is FILTERED (gen_len>1 and gen_time>=50ms), so it is not
     index-aligned with the other per-request lists and cannot be sliced by
     `last_distributions_index`. `actual_tpots_aligned` appends on EVERY
     request, writing 0.0 where the sample was filtered out, which keeps it
     lock-step with new_ttfts / new_prompt_lengths / new_generation_lengths.
  2. `add_prefill()` takes an optional `e2e=` kwarg and appends both lists.
  3. The realistic-mode call site passes `e2e=total_time` (TTFT + generation,
     measured wall-clock across the whole streamed response).
  4. `save_metrics_loop` emits `new_e2es` and `new_tpots` alongside `new_ttfts`,
     sliced with the same index so the arrays concatenate row-wise.

The other two call sites (run_replay, dataset replay) are left alone -- they get
the default e2e=0.0 and are not on the solo path.

SAFETY
------
Idempotent (bails if SOLO_M1 is already present). Every edit anchors on exact
source text and sys.exit(2)s loudly if the anchor is missing, so a driver that
has drifted fails here rather than silently producing a file with no E2E column.

Usage:  python3 apply_solo_metrics.py [/path/to/agent/agent_throughput.py]
"""
import sys

P = sys.argv[1] if len(sys.argv) > 1 else \
    "/mnt/vast/c_huggingface/bench_20260801/agbench/agent/agent_throughput.py"

src = open(P).read()
if "SOLO_M1" in src:
    print("already patched (SOLO_M1 present) - no-op")
    sys.exit(0)


def sub(old, new, label):
    global src
    n = src.count(old)
    if n != 1:
        print(f"ANCHOR FAIL [{label}]: expected 1 occurrence, found {n}")
        sys.exit(2)
    src = src.replace(old, new)


# ---- 1. new fields on BenchMetrics ------------------------------------------
sub(
    "    actual_acceptance_lengths: List[float] = field(default_factory=list)  "
    "# Track per-request average acceptance length",

    "    actual_acceptance_lengths: List[float] = field(default_factory=list)  "
    "# Track per-request average acceptance length\n"
    "    # SOLO_M1: per-request end-to-end latency (TTFT + full generation), seconds.\n"
    "    # Never recorded upstream -- Case A had to back-solve E2E from TTFT and TPOT.\n"
    "    actual_e2es: List[float] = field(default_factory=list)\n"
    "    # SOLO_M1: TPOT with one entry per REQUEST (0.0 where the sample was\n"
    "    # filtered out), unlike actual_tpots which is filtered and therefore not\n"
    "    # index-aligned with the other per-request lists.\n"
    "    actual_tpots_aligned: List[float] = field(default_factory=list)",
    "fields",
)

# ---- 2. add_prefill: accept e2e, append both --------------------------------
sub(
    "def add_prefill(self, tokens: int, duration: float, cached_tokens: int = 0, "
    "generation_tps: float = 0.0, generation_tps_mtp: float = 0.0, "
    "actual_gen_length: int = 0, generation_time: float = 0.0, "
    "prefix_size: int = 0, reasoning_tokens: int = 0):",

    "def add_prefill(self, tokens: int, duration: float, cached_tokens: int = 0, "
    "generation_tps: float = 0.0, generation_tps_mtp: float = 0.0, "
    "actual_gen_length: int = 0, generation_time: float = 0.0, "
    "prefix_size: int = 0, reasoning_tokens: int = 0, e2e: float = 0.0):",
    "add_prefill signature",
)

sub(
    """        if actual_gen_length > 1 and generation_time >= MIN_GENERATION_TIME:
            self.actual_tpots.append(generation_time / (actual_gen_length - 1))""",

    """        if actual_gen_length > 1 and generation_time >= MIN_GENERATION_TIME:
            self.actual_tpots.append(generation_time / (actual_gen_length - 1))
            _tpot_aligned = generation_time / (actual_gen_length - 1)
        else:
            # SOLO_M1: keep the array index-aligned. 0.0 marks "filtered", and
            # the analysis drops zeros rather than treating them as fast tokens.
            _tpot_aligned = 0.0
        self.actual_tpots_aligned.append(_tpot_aligned)
        # SOLO_M1: e2e defaults to 0.0 on the call sites that do not pass it.
        self.actual_e2es.append(e2e)""",
    "add_prefill body",
)

# ---- 3. realistic-mode call site passes the measured total ------------------
sub(
    """                metrics.add_prefill(tokens_to_record, ttft, cached_tokens, generation_tps, generation_tps_mtp,
                                    completion_tokens, generation_time, current_prefix_tokens,
                                    reasoning_tokens=reasoning_tokens)

                if chunk_token_counts:
                    avg_acceptance_length = sum(chunk_token_counts) / len(chunk_token_counts)
                    metrics.add_acceptance_length(avg_acceptance_length)
                else:
                    # Request completed but no content returned
                    metrics.requests_completed += 1""",

    """                metrics.add_prefill(tokens_to_record, ttft, cached_tokens, generation_tps, generation_tps_mtp,
                                    completion_tokens, generation_time, current_prefix_tokens,
                                    reasoning_tokens=reasoning_tokens,
                                    e2e=total_time)  # SOLO_M1

                if chunk_token_counts:
                    avg_acceptance_length = sum(chunk_token_counts) / len(chunk_token_counts)
                    metrics.add_acceptance_length(avg_acceptance_length)
                else:
                    # Request completed but no content returned
                    metrics.requests_completed += 1""",
    "realistic call site",
)

# ---- 4. emit the two arrays -------------------------------------------------
sub(
    """        new_ttfts = metrics.actual_ttfts[last_distributions_index:current_distributions_len]
        last_distributions_index = current_distributions_len""",

    """        new_ttfts = metrics.actual_ttfts[last_distributions_index:current_distributions_len]
        # SOLO_M1: same slice bounds -> these concatenate row-wise with new_ttfts.
        new_e2es = metrics.actual_e2es[last_distributions_index:current_distributions_len]
        new_tpots = metrics.actual_tpots_aligned[last_distributions_index:current_distributions_len]
        last_distributions_index = current_distributions_len""",
    "metrics slice",
)

sub(
    """            "new_ttfts": new_ttfts,
            "new_acceptance_lengths": new_acceptance_lengths,""",

    """            "new_ttfts": new_ttfts,
            "new_e2es": new_e2es,      # SOLO_M1
            "new_tpots": new_tpots,    # SOLO_M1 (0.0 = filtered, drop in analysis)
            "new_acceptance_lengths": new_acceptance_lengths,""",
    "metrics record",
)

open(P, "w").write(src)
print(f"patched OK - SOLO_M1 occurrences: {src.count('SOLO_M1')} (want 8)")
