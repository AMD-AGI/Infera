# Upstream PR draft (NOT submitted)

Target repo: sgl-project/sglang, base `main`. Written 2026-09-24 against `main` @ `4142235c2b`.

This draft is meant to be stacked on
[#40857](https://github.com/sgl-project/sglang/pull/40857), which fixes decode
radix KV ownership and allows EAGLE/EAGLE3. If #40857 merges first, the code
diff below reduces to adding `NEXTN` and the tests. If #40857 stalls, see
"Fallback scope" at the end.

---

## Title

[PD] Allow NEXTN with decode radix cache; add spec-decoding and decode-HiCache coverage (validated on GLM-5.2 DSA / ROCm)

## Motivation

`--disaggregation-decode-enable-radix-cache` has rejected every speculative algorithm since
[#19746](https://github.com/sgl-project/sglang/pull/19746). No technical reason was given, and
three questions about it in that PR's comments went unanswered. As a result, a decode leg that runs
EAGLE/NextN MTP is forced onto `ChunkCache`, which has three consequences:

- every turn re-transfers the full prompt KV from prefill;
- the decode leg cannot publish KV events for prefix-aware routing;
- `--enable-hierarchical-cache` on decode fails with "enable-hierarchical-cache and
  disable-radix-cache are mutually exclusive", because the PD hook forces `disable_radix_cache=True`.

[#40263](https://github.com/sgl-project/sglang/pull/40263) allowed DSPARK, and #40857 /
[#40681](https://github.com/sgl-project/sglang/pull/40681) allow EAGLE/EAGLE3 once an SWA
ownership bug is fixed. Two gaps remain after those PRs.

1. **`NEXTN` is still rejected.** `handle_pd_disaggregation` runs before
   `handle_speculative_decoding` (`arg_groups/pipeline.py`), and `NEXTN` is only normalized to
   `EAGLE` inside the speculative hook (`_resolve_speculative_algorithm_alias`). A user who launches
   an MTP model with the documented `--speculative-algorithm NEXTN` still hits the `ValueError`,
   even though that configuration is identical to `EAGLE` at runtime.
2. **No test covers spec decoding with decode radix, and no report has real acceptance.**
   [#39150](https://github.com/sgl-project/sglang/pull/39150) was closed partly because its runs
   accepted zero draft tokens, which left multi-token acceptance unvalidated. No test covers decode
   HiCache + MTP either.

This PR closes both gaps and adds GPU evidence from a DSA model (GLM-5.2) with real MTP acceptance,
DP attention, Mooncake, and decode HiCache with host load-back, on AMD MI355X.

## Modifications

1. `python/sglang/srt/arg_groups/pd_disaggregation_hook.py`: add the raw `NEXTN` alias to the
   speculative-algorithm allowlist, since the PD hook sees the pre-normalization CLI value. Keep
   DSPARK, EAGLE and EAGLE3 as in #40263 / #40857. Every other algorithm (NGRAM, STANDALONE,
   DFLASH, ...) is still rejected, with the same error text.

   ```python
   # Raw CLI values: this hook runs before NEXTN is normalized to EAGLE.
   _DECODE_RADIX_SPEC_ALGORITHMS = (None, "DSPARK", "EAGLE", "EAGLE3", "NEXTN")

   if cfg.speculative_algorithm not in _DECODE_RADIX_SPEC_ALGORITHMS:
       raise ValueError(
           "--disaggregation-decode-enable-radix-cache is incompatible "
           "with speculative decoding "
           f"(--speculative-algorithm {cfg.speculative_algorithm})"
       )
   if cfg.speculative_algorithm is not None and (cfg.speculative_eagle_topk or 1) > 1:
       logger.warning(
           "EXPERIMENTAL: Decode radix cache with tree drafts "
           "(--speculative-eagle-topk > 1) has not been validated."
       )
   ```

   (`speculative_eagle_topk` may still be `None` at this point; it is auto-resolved later by the
   speculative hook, so the warning only fires on an explicit topk > 1.)

2. `python/sglang/srt/arg_groups/fields/disagg.py`: update the help text to list NEXTN.

3. No tree or scheduler changes beyond #40857. For full-attention, MLA and DSA models the tree has
   only the FULL component. There, the post-insert re-match in `cache_unfinished_req` already covers
   the whole inserted range, so the SWA problem #40857 fixes does not arise.

## Gating

- **No environment variable.** Upstream precedent for this guard is a plain allowlist (DSPARK in
  #40263; EAGLE/EAGLE3 in #40857, #40681 and #39150). Our local build used an opt-in env
  (`SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1`) plus a hard `topk == 1` check; for upstream we drop
  the env and downgrade the topk check to the warning above.
- The existing `EXPERIMENTAL: Radix cache is enabled for decode server` warning, and the DP
  attention warning, still fire.
- Reviewers may prefer to reject `--speculative-eagle-topk > 1` outright until it has a test. That
  is a one-line change, and we are happy to make it.

## Test plan

Unit tests on CPU, in `test/registered/unit/server_args/test_server_args.py`:

- `test_pd_decode_radix_cache_allows_chain_speculative_algorithms`: for EAGLE, EAGLE3, NEXTN and
  DSPARK, the decode leg with `--disaggregation-decode-enable-radix-cache` (mooncake) resolves
  `disable_radix_cache=False`.
- `test_pd_decode_radix_cache_rejects_other_speculative_algorithms`: NGRAM and STANDALONE still
  raise the existing `ValueError`.
- `test_pd_decode_hicache_with_eagle_and_decode_radix`: running `handle_pd_disaggregation`, then
  `handle_cache_compatibility`, on a decode leg with EAGLE + decode radix +
  `--enable-hierarchical-cache` does not raise "mutually exclusive". Without the decode radix flag
  it still raises.

Unit tests on CPU, in `test/registered/unit/mem_cache/test_unified_radix_cache_unittest.py`: the
suite is already parameterized on `is_eagle`. Add a FULL-only case covering decode prealloc
(`match_prefix` + lock), then `cache_unfinished_req` / `cache_finished_req`, with a bigram key and
`page_size=64`. It should assert that the protected length covers the page-aligned insert and that
`available + evictable == total` after release.

End-to-end tests on GPU, in `test/registered/disaggregation/test_disaggregation_decode_radix_cache.py`:

- `TestDisaggregationDecodeRadixCacheEagle3Mooncake`: uses `DEFAULT_TARGET_MODEL_EAGLE3` /
  `DEFAULT_DRAFT_MODEL_EAGLE3` with EAGLE3 steps 3 / topk 1 / draft 4 on both legs, using the same
  launch arguments as `TestDisaggregationMooncakeSpec` in `test_disaggregation_basic.py`. It reuses
  the mixin's hit test and two-pass GSM8K test, and adds
  one more assertion: mean `spec_accept_length` > 1.5 on the second (cache-hit) pass, and no more
  than a small margin below the first pass.
- Optional: `...NextNMooncake` with `DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN`, to exercise the raw
  `NEXTN` alias end to end.
- Optional: a decode-HiCache variant (`write_through`, small `--max-total-tokens` on decode) that
  forces eviction to host and asserts a load-back on reuse.
- Optional: an AMD registration next to the existing MI35x disaggregation job.

Locally we will also re-run the GLM-5.2 validation below on a `main`-based image before marking
the PR ready.

## Validation results (GLM-5.2, AMD MI355X)

**Setup, common to both runs:**

- Model: GLM-5.2-MXFP4 (`GlmMoeDsaForCausalLM`; draft `GlmMoeDsaForCausalLMNextN`).
- Topology: PD 1 prefill + 1 decode, each TP4 / DP4 with `--enable-dp-attention`.
- Transfer: Mooncake over RDMA.
- Speculative decoding: decode only, EAGLE (NextN MTP) steps 5 / topk 1 / draft tokens 6. Prefill
  runs without speculative decoding.
- KV cache: page size 64, fp8 KV; prefill HiCache ratio 1.5.
- Router: prefix-aware routing with prefill/decode DP-rank affinity.
- Software: SGLang `a9fb1c3238` (2026-09-17) plus a local env-gated patch that relaxes only this
  guard (EAGLE/NEXTN, topk 1). There are no other changes to the radix, decode or HiCache paths.
- Accuracy: InferenceX lm-eval GSM8K, full 1319 questions, real MTP acceptance (no simulated
  acceptance), concurrency 64.

**Run B: decode radix cache on**

- Startup: every decode DP rank uses `UnifiedRadixCache` (FULL component), and "KV cache is forced
  as chunk cache" no longer appears. `/server_info` shows
  `disaggregation_decode_enable_radix_cache=true` and `disable_radix_cache=false`.
- GSM8K with random few-shot (stock task): strict-match 0.9682, flexible-extract 0.9704. Few
  requests share a prefix, so this is mostly a no-regression check.
- GSM8K with fixed few-shot (sampler `first_n`, so requests share the few-shot prefix):
  strict-match 0.9757, flexible-extract 0.9765. 1255 of 1319 requests (95.1%) hit a decode-side
  prefix, covering 82% of blocks. The misses are essentially the first 64 concurrent requests,
  which were issued before anything was cached.
- Two ~11k-token shared-prefix probes: the second request hit 169 of 170 blocks on decode, and
  both answers were correct. Flushing only the decode leg dropped hits to 0, which confirms that
  the reuse happens on decode.

**Run C: decode radix + decode HiCache, both legs with HiCache**

- Settings: `write_through`, `kernel` IO, `page_first` layout. The decode device pool was capped at
  200k tokens/rank to force eviction; the host pool was 60 GB/rank (1,318,592 tokens).
- Startup: the MTP draft layer is packed into the host pool (`target_layers=78, draft_layers=1,
  total_layers=79`), and the DSA indexer host pool also has 79 layers. `hicache_attached=True`.
- GSM8K with fixed few-shot: strict-match and flexible-extract both 0.9727, within the error bar of
  Run B. Decode wrote 997,952 tokens to host and evicted 199,104 from device. MTP accept length in
  the decode logs was about 3.5–4.0.
- Host load-back probe:
  - Write phase: 8 documents of about 20k tokens each.
  - Flood phase: 80 other documents, about 2x the decode device capacity per rank.
  - Reuse phase: one new question on each of the 8 documents.

  All 96 answers were correct. The reuse phase loaded 162,304 tokens back from host, which is 99.8%
  of the reused prompt tokens; the remainder (partial last page and new question) came from
  prefill. There were no retractions, no load-back failures, and no eviction-insufficient warnings.

**Performance:** an A/B run is in progress; results will be added before this PR is marked ready.

## Known limitations

- **Coverage.** Our runs cover only topk 1 and EAGLE with a NextN draft. We have not run EAGLE3 or
  CUDA ourselves; the proposed CI test covers CUDA with EAGLE3. Tree drafts (topk > 1) are
  unvalidated by anyone, which is why they get the warning.
- **Base commit.** The GPU numbers come from `a9fb1c3238` plus a guard-only patch, not from this
  branch. They will be re-run on a `main`-based image; `decode.py` and the DSA/MTP decode path
  changed after that commit.
- **Acceptance on hits vs. misses.** We report accept length for the HiCache run but have not yet
  compared it between decode-prefix hits and misses; this will be added.
- **Prompt-position draft KV.** In our deployment prefill has no draft model, so draft KV at prompt
  positions never comes from prefill. Radix reuse makes it come from KV the decode leg computed for
  an earlier request. In the common deployment where prefill also runs EAGLE, draft KV is
  transferred for the suffix and reused for the prefix.
- **Retraction.** Retraction was not exercised in either run. The re-bootstrap after a retraction
  does not re-match the decode prefix (existing TODO), and on HIP the retraction backup is always
  `cpu_tensor`.
- **Models.** SWA hybrids need the tree fix from #40857 or #40681. This PR relies on that fix and
  adds nothing for SWA or DSV4.
- **DP attention.** Each attention DP rank keeps its own tree, so hit rate depends on prefix-aware,
  rank-stable routing (same caveat as the existing warning).
- **Where the benefit comes from.** Decode prefix hits save prefill→decode transfer and duplicate
  decode HBM. They do not save prefill compute: prefill still computes the full prompt locally and
  sends only the delta (confirmed in #19746).
- **Cache-key overshoot.** Speculative overshoot can still enter the radix key (see
  [#35694](https://github.com/sgl-project/sglang/pull/35694)). This lowers hit rate, not
  correctness.
- **Decode HiCache scope.** We did not measure long-running accept-length drift under HiCache (see
  [#28771](https://github.com/sgl-project/sglang/issues/28771)) or test an L3 storage backend.
- **Greedy parity as a test oracle.** Token-by-token greedy parity was not usable: requests on
  different DP ranks diverged even without any reuse. Correctness was therefore judged with
  full-GSM8K statistics.

## Fallback scope (if #40857 does not land)

Open a standalone PR that allows EAGLE, EAGLE3 and NEXTN with decode radix cache only for models
without an SWA component. Detect this with the same `is_hybrid_swa` check that `kv_cache_builder.py`
already uses for the decode-radix model gate, and keep rejecting SWA models when speculative
decoding is on. This avoids the double-ownership bug without touching `UnifiedRadixCache`. Tests
and evidence stay the same.

## Checklist

- [ ] Rebase onto `main`, then re-run the Run B and Run C gates on a `main`-based ROCm image.
- [ ] Measure accept length separately for hit and miss requests.
- [ ] Attach the performance A/B.
- [ ] Reproduce with Llama-3.1-8B + EAGLE3 on CUDA with Mooncake.
- [ ] Run pre-commit, the server-args unit suite and the URC unit suite.
- [ ] Update the documentation for `--disaggregation-decode-enable-radix-cache`.
