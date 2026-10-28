PD follow-up, kept as drafts until the backend/layout contracts here settle:

- #46824 handles PD relayout when the ROCm DSA draft shards its KV like the target. It consumes the canonical `dsa_draft_shards_under_dcp` policy proposed here; on today's main, which lacks that policy, it preserves replicated-draft behavior. It includes host-byte DP/PP/chunk/page regressions, not a current GPU certification.
- #46826 lifts only the ROCm decode-side PD guard and must land after this backend and #46824. It does not implement your backend or change its defaults.
- #46828 offers opt-in GLM kv_b BF16 retention for replicated-Q preparation, with numerical/performance checks still pending.

We also coordinated overlapping split-indexer work in #42618 instead of filing another implementation. Please flag any final-policy incompatibility before these PD drafts advance. Historical `pdBe-r1` passed limited math/passkey checks, but DP2×DCP2 graph garbling remains unbisected; historical simulated-acceptance throughput is not evidence that this combined stack is correct.
