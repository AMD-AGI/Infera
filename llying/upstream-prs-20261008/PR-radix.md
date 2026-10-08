Infera already enables Decode radix cache for compatible non-speculative Mooncake workers, but unconditionally skips it when speculative decoding is active. This adds an explicit `SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1` opt-in for workers running a compatible SGLang build.

The default remains unchanged. KV events, Decode role, Mooncake transport, and the existing model/topology rejection checks still gate automatic flag forwarding. Explicit engine flags remain the engine's responsibility. This does not enable Decode HiCache or bundle an SGLang patch.

The historically validated engine combination is GLM-5.2 EAGLE/NEXTN, top-k 1, using the [experimental SGLang patch](https://github.com/AMD-AGI/Infera/blob/3f0aa62af014becf56798ab2731fb67cd4cdd268/llying/decode-kv-aware-mtp-radix-20260924/patches/01-sglang-decode-radix-allow-eagle.patch). Unsupported stock builds can still reject the forwarded flag; this opt-in is not automatic capability detection or a claim of support for every speculative algorithm.

Validation:
- 14 CPU argv-forwarding tests pass, including default/off/exact opt-in, non-speculative behavior, preserved model/topology rejections, unrelated roles/transports, disabled KV events, and explicit-flag deduplication. These use a stub SGLang argument resolver and exercise the real Infera parser.
- Added real-SGLang guard tests; the engine test module is skipped locally because SGLang is not installed. No new GPU benchmark or accuracy run is claimed for this extraction.
- Ruff format/check and `git diff --check` pass.

This PR is independent of the Router lifecycle and session-affinity changes.
