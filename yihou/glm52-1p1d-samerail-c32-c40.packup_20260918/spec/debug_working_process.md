# Debug loop — garbled output on P4D4

## Target (pass/fail)

With `temperature=0` and a trivial prompt, the service returns coherent text.
Concrete check: `"What is 2+2? Answer with a single digit."` must not degenerate
into a repeated filler token, and
`"Reply with exactly: BASELINE_OK_<nonce>"` should reproduce the nonce.

Current state: FAIL.

## Configuration under test

GLM-5.2 MXFP4, 1P1D, prefill crsuse2-m2m-135 / decode crsuse2-m2m-138, both on
GPUs 2,3,4,5. **P4+DPA / D4+DPA (TP4/DP4)** — the bench's validated baseline is
TP8/DP8, so this shape is new and unproven. Image
`infera-sglang:v0519-yihou-0917`. `index_share_for_mtp_iteration=false` applied.
Simulated MTP acceptance OFF.

## Materials

1. `bench/glm5p2_pd/issue.md` — prior campaign history. §2.4 and Appendix A both
   document garbled output; §2.4's trigger (removing the IndexShare workaround)
   is **absent** here because we keep the workaround.
2. `infera.glm52.view/rdma.survey.8node.packup_20260917` — fleet RDMA survey.
   Already used to establish the same-rail fix.
3. No known-good reference program running yet. **A TP8 single-node run is the
   obvious reference to build** — see R02 note.

## Rounds

| # | Hypothesis | Change | Result |
|---|---|---|---|
| R01 | Is the first token already wrong? | none — probe only | First token OK, all later tokens degenerate |
| R02 | MTP draft/verify is the fault | `DECODE_MTP=0` | pending |

---

### R01 — where does it first go wrong? (`r01-first-token/`)

Probe at `temperature=0`, varying `max_tokens`:

| max_tokens | reasoning_content |
|---|---|
| 1 | `1` |
| 3 | `1!!` |
| 8 | `1!!!!!!!` |

**The first token is correct; every subsequent token degenerates to `!`.** A
repeated single token is the signature of degenerate logits (argmax landing on
a fixed low id), not of a tokenizer or parser fault — the earlier long probe
also returned real, individually-valid tokens in nonsense order, so bytes and
decoding are fine and the *numbers* are wrong.

In PD the first token comes out of prefill; everything after it is produced by
decode. So the fault is on the **decode** side, not in prefill and not in the
prompt path.

Corroborating evidence from `decode-0.log` during the same window:

- `transfer failed | tx_rdma_ack_timeout | retry_excd | blacklist` → **0
  occurrences**. The KV transport is clean; same-rail pinning is working. This
  rules the Mooncake layer out rather than leaving it a suspect.
- `Decode batch ... accept len: 1.25, accept rate: 0.05` — MTP acceptance is
  **5%**. The bench's simulated stand-in for a healthy run is 3.61.

That last number is the strongest lead. Note what it does *not* by itself
explain: a draft model producing rubbish should be *rejected*, and rejection
should fall back to the target model's own token, giving correct-but-slow
output. Getting both a 5% accept rate **and** wrong text means either the
target model is also wrong on the decode side, or the verify/fallback path is
not doing what it should.

Evidence sources used: program output (probe), component logs (decode worker),
prior-campaign document (`issue.md`).

### R02 — is MTP the fault? (`r02-no-mtp/`)

Single variable: `DECODE_MTP=0`, everything else identical.

- Coherent output → the fault is inside the MTP path (draft model at TP4, the
  verify/rejection path, or the MTP-specific `dsa_page_table_rows`
  `repeat_interleave` that only runs under speculative decode).
- Still garbled → the fault is in the plain decode path: the target model at
  TP4, or the handed-off KV being mis-mapped once decode starts reading it.

Either outcome halves the search space, which is why this is the next step
rather than a code read.
