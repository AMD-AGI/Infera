# Results

## Start here

- **`matrix.csv`** — the whole experiment in four rows: which knobs, how many
  replies were coherent, per-rank `spec_accept_length`, pass/fail against the
  2.0 bar.
- **`flags-by-round.txt`** — `disable_custom_all_reduce`,
  `enable_aiter_allreduce_fusion` and `speculative_algorithm` extracted from each
  round's own `server-info/decode-0.json`. This is the file that shows the aiter
  fusion path was **True in every round** — held constant, so custom all-reduce
  is the discriminator rather than a co-varying suspect.

## How the coherence counts were produced

**By reading every reply.** Not by a heuristic. `matrix.csv`'s
`requests_coherent` column is a human judgement over all 48 replies, and every
one of them is dumped verbatim in `<round>/rank-test-replies.txt` so the
judgement is checkable in about a minute.

This is deliberate. A first attempt at an automatic classifier scored R04/R05/R06
as failures: `max_tokens=12` truncates the correct replies mid-sentence
(`1.  **Analyze the Request:**`), and one of the two garbled modes is pure ASCII
word-salad (`1reis0obuf Erect bufferreisangan legacyu`) that no simple rule
separates from prose. A wrong number in a deliverable is worse than no number.

## Per-round directories

Each of `r03-fusion-fix/`, `r04-no-mtp/`, `r05-nocustomar/`,
`r06-nofix-nocustomar/` holds:

| path | what |
|---|---|
| `rank-test/r*.json` | the raw OpenAI responses, 16 or 8 identical requests at `temperature=0`, `max_tokens=12` |
| `rank-test-replies.txt` | the same replies as one readable list |
| `probe/` | the acceptance probe: R01's `max_tokens` 1/3/8 ladder, both target prompts, 8 concurrent 400-token generations, `/metrics` before and after load, and the decode-log `accept len:` series. Absent for R04, where MTP is off and the spec gauges are meaningless |
| `server-info/` | the engine's own `server_args` dump per leg — the authoritative record of what each round actually ran |
| `workers.json` | router view: two workers, correct roles |

## Reading the evidence

**The A/B is R03 vs R05.** Same image, same config, same real acceptance, MTP on
in both; `--disable-custom-all-reduce` on the decode leg is the only difference.
Compare `r03-fusion-fix/rank-test-replies.txt` against
`r05-nocustomar/rank-test-replies.txt` and the result needs no interpretation.

**R04 is the bisection**: MTP off, everything else as R03. 16/16 coherent, which
exonerates the target model at TP4/DP4, the PD KV handoff including same-rail NIC
pinning and router DP-rank affinity, the plain decode attention path, and the
`index_share_for_mtp_iteration=false` override.

**R06 is the corner that makes the finding precise**: the NextN fusion fix
removed, custom all-reduce still off. Still 8/8. So custom all-reduce alone is
the whole cure and the fusion fix is not required.

### Two traps

1. `probe/metrics-idle.txt` shows `spec_accept_length 0.0` on ranks that have
   served nothing. That is **not** a measurement of zero acceptance. It is kept
   in the packup precisely so the trap is visible; the measurement is
   `metrics-loaded.txt`, and only ranks whose counters moved count.
2. `accept_length` and `accept_rate` are different metrics. The bar is on
   **length**. The failing baseline read length 1.25 / rate 0.05.

## The numbers

| round | coherent | `spec_accept_length` per active rank | min | verdict |
|---|---|---|---|---|
| R03 | 0/16 | dp0 1.00, dp1 1.00, **dp2 2.8125**, dp3 1.00 | 1.00 | FAIL |
| R04 | 16/16 | n/a — MTP off | n/a | PASS |
| R05 | 8/8 | 2.8875 / 3.0500 / 2.8684 / 2.8625 | 2.8625 | PASS |
| R06 | 8/8 | 2.8000 / 2.8125 / 2.5500 / 2.6250 | 2.5500 | PASS |

R03's split — one healthy rank, three reading *exactly* 1.00/0.00 — is the
bimodality that disappears in R05 and R06, where all four ranks land in a tight
band. `notes.md` §6 explains why that split was recorded as a clue rather than a
localisation, and why that caution turned out to be correct.
