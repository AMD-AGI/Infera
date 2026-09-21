# patches/

## 0001-engine-sh-extra-args-env.patch

- **What:** adds a `DECODE_EXTRA_ARGS` (and matching prefill) escape hatch to
  `bench/glm5p2_pd/engine.sh`, appending arbitrary flags to the engine command
  line.
- **Why:** the "corrected MTP config" needs `--disable-custom-all-reduce` on the
  decode leg, and sglang exposes no positive/negative flag pair for it; the only
  other way to disable it (`--enable-deterministic-inference`) changes much more.
- **How applied:** `git apply patches/0001-engine-sh-extra-args-env.patch` from
  the repo root before bring-up. It is uncommitted in the branch.
- **Context:** without it, `config.yihou.full.dcar.sh` cannot inject the flag and
  the run falls back to the custom-all-reduce-ON (t2e) cell. Same patch as the T1
  pack-up.

No image patch is included: **T2 ran the stock base image**
(`infera-sglang:v0519-yihou-0917`), so neither the NextN fusion fix nor
PR #37152 applies here. Those image patches live in the T1 pack-up
(`glm52-agentx-c40-realacc.packup_20260918/patches/`) for anyone reproducing the
`-nextnfix-hicache` layer instead.
