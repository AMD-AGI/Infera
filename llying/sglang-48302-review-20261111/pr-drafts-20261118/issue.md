Title: [Bug] dsv41: Candidate graph variants can differ across attention-DP ranks (decode and DSpark verify)

## Problem

Under DP attention the ranks replay their CUDA/HIP graphs in lockstep, and some collectives captured in them (custom all-reduce, aiter all-gather / reduce-scatter) register that graph's input buffers. If two ranks replay different graph variants in the same step, a rank reads its peer's buffer from the other graph, which holds stale data.

`Dsv41CandidateGraphVariants` (DeepSeek-V4.1, SM100+ and gfx95) still selects from each rank's own lengths:

- Decode: ranks holding different requests can pick different candidates; an idle rank has no lengths and takes `candidate_filtered`.
- DSpark target verify: busy ranks select from their committed lengths plus `verify_extra_tokens`. An idle rank stays `IDLE` (no ragged layout) or runs `TARGET_VERIFY` with all-ones lengths (ragged layout), so it can pick a different variant. `SGLANG_OPT_DSV41_TARGET_METADATA_EARLY` is off under DP attention (`_wants_early_verify_metadata()`), so verify does not need its own lengths for early staging there.

#48302 fixes the same issue for `DsaGraphVariants` and `SeqLenBucketGraphVariants`: an upper bound of each rank's longest decode sequence rides the existing MLP-sync all-gather (`dp_decode_graph_max_seq_len`), and those classes select from the maximum over ranks in decode / idle. DSV4.1 was left out of that PR (review: https://github.com/sgl-project/sglang/pull/48302#discussion_r4595135900).

## Proposed fix

- Under DP attention, select DSV4.1 candidates from the global bound in decode, idle and target verify alike (`select_for_max_seq_len(bound)`, which adds `verify_extra_tokens`), including the ragged idle-verify path.
- Make sure the bound covers what verify uses locally: `_local_decode_graph_max_seq_len()` should be at least `candidate_request_length_bound(reqs, verify_num_draft_tokens)`.
- Validate on B200 and MI355X with DP attention and DSpark: every rank replays the same variant, and GSM8K / long-context output is unchanged.
