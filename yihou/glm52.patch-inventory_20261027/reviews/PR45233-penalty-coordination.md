Our upstream audit found that the accepted-token accounting portion here overlaps my older #33643 (bot-closed stale draft, not merged). I will keep #33643 closed and track this implementation instead of reopening a competing fix.

The old regressions compared sequential per-token accumulation with a multi-token speculative commit, including unequal accepted lengths, padded tails, frequency/presence/repetition state and `min_new_tokens`. Your `penalty_cumulated_output_len` and accepted-output-token path address the same lost-token issue and additionally handle within-verify-block state. Please preserve the variable-length/padding and min-new-token checks when separating/rebasing the core fix from the CORAL overlay.

This is code/coverage coordination, not a GPU or model-accuracy endorsement of this PR. Its outstanding distribution/serving validation remains required. Our GLM historical throughput used simulated acceptance and is not supporting evidence for sampling correctness.
