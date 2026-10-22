# Custom logit processors in the EAGLE verify step

Stock SGLang v0.5.18 applies request-scoped custom logit processors in its
regular sampler but not in the EAGLE verify step (`eagle_sample` in
`speculative/eagle_utils.py`). The GLM-5.3 thinking-budget processor
(`infera/engine/sglang/thinking_budget.py`) needs that step.

`patch_eagle_custom_logit_processor.py` adds a call to the existing
`apply_custom_logit_processor` helper in `eagle_sample`, with
`num_tokens_in_batch=draft_token_num`, after the logit bias and grammar mask.
For the duration of the call each request's custom params carry the draft chain
(`__spec_candidates__`), the sequence length (`__spec_seq_len__`) and the block
size (`__spec_num_tokens__`).

The thinking-budget processor then handles one request per `draft_token_num` rows:

- Row `k` is evaluated on the committed outputs plus draft tokens `1..k`, so a
  `</think>` or newline inside the draft block is counted as it would be when
  decoded one token at a time.
- Committed tokens that are not yet in `req.output_ids` are counted through the
  sequence length.
- The draft chain is read back from the device only when the budget can be
  reached within the current block.

Scope: chain drafts (`--speculative-eagle-topk 1`), for example EAGLE MTP(3,1,4).
With a larger top-k the processors are skipped and one warning is logged.

The script fails on an unknown `eagle_sample` layout or a partially patched file,
is idempotent, removes stale module bytecode and recompiles with checked source
hashes. `Dockerfile.sglang.glm53` applies it. Drop the patch when the pinned
SGLang base applies custom logit processors in the EAGLE verify step.
