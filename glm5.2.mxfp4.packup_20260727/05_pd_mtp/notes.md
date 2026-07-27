# Notes — 05 PD MTP

## Why MTP is on the decode leg only

Speculative decoding accelerates **token generation** (draft N tokens, verify in one parallel pass).
In PD disaggregation, generation happens on the **decode** leg; the prefill leg only turns the prompt
into KV and hands it off — it never generates tokens sequentially. Putting spec-dec on prefill would
(a) load a second (draft) model into prefill's memory, (b) add draft compute to every prefill forward
pass, throttling the pipeline — the reference program measured per-user rising but aggregate collapsing
and TTFT ballooning. So: **prefill leg = plain (MTP=0), decode leg = EAGLE (MTP=1).**

## Decode-leg MTP tuning (to avoid the PD high-conc KV-pool crash)

The reference program found that MTP on a PD decode leg OOMs the decode KV pool at conc ≥ 32 with
5 draft steps: EAGLE's draft-extend needs extra KV on top of each request's KV + the PD transfer
reservation, and the decode pool (PD decode disables radix cache → nothing evictable) fills up.

We used the reference's stable knobs and saw **no crash through conc=64**:
- `--speculative-num-steps 3` (not 5)
- `--speculative-num-draft-tokens 4`
- `--num-reserved-decode-tokens 256` (not 512)
- decode `--mem-fraction-static 0.80`

(These are env-overridable in `engine.sh`: `SPEC_STEPS`, `SPEC_DRAFT`, `RESERVED_TOK`, `MEMFRAC`.)
Trade-off: lower steps → lower accept len (2.75–2.88 here vs 3.5–4.8 for single-node 5-draft) but
stable under PD concurrency.

## The two rc6 MTP fixes apply unchanged in PD

Same as 04: mount the 1-line-patched `deepseek_nextn.py` (else `3072 vs 6144` shape crash at draft
load) + set `SGLANG_DSA_ENABLE_MTP_PRECOMPUTE_METADATA=0` (else decode hangs on the CUDA
`fused_metadata_copy` kernel). Only the **decode** leg needs them (it's the only leg with MTP). Both
verified working across the RDMA PD boundary — the draft head loads and accepts tokens on the decode
node while KV arrives over MoRI from the prefill node.

## Result vs no-MTP PD (02)

Same workload, same transport, MTP on decode adds:
- total throughput 5167 → **7444 tok/s** (+44%)
- median TPOT 20.9 → **12.1 ms** (1.7× faster per-token — the spec-dec win)
- median TTFT 535 → 412 ms
- still 256/256, no crash.

## Node-swap story (why prefill = chi2832, not chi2878)

02/03 used chi2878 as prefill. By the time we ran 05, chi2878 had been taken by another user's
`pd_uni` (DeepSeek-V4) container. Per the mission owner we moved prefill to **chi2832** (freed by
killing an unrelated container there). chi2832 has the same MI355X/ionic setup and routable GID at
idx1, so `MORI_IB_GID_INDEX=1` is correct there too. Nothing about the recipe changed — only the
prefill hostname/IP in `up.sh`.

## Logs

`logs/decode.log` — grep `accept len` for the spec-dec proof, `Decode batch` for throughput.
`logs/prefill.log` — the plain prefill leg (no MTP); `Load weight`, bootstrap, registration.
