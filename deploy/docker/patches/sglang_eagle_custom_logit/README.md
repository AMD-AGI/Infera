# Custom logit processors in the EAGLE verify step

Stock SGLang v0.5.18 applies request-scoped custom logit processors in its
regular sampler but not in the EAGLE verify step (`eagle_sample` in
`speculative/eagle_utils.py`). The GLM-5.3 thinking-budget processor
(`infera/engine/sglang/thinking_budget.py`) needs that step.

`patch_eagle_custom_logit_processor.py` adds a call to the existing
`apply_custom_logit_processor` helper in `eagle_sample`, with
`num_tokens_in_batch=draft_token_num`, after the logit bias and grammar mask.
Only processors that set `SUPPORTS_SPEC_VERIFY_ROWS = True` run in this step,
since each request owns `draft_token_num` rows there. For the duration of the call
each request's custom params carry the block size (`__spec_num_tokens__`), the
step (`__spec_step__`) and the request's row in it (`__spec_row__`). A step holds
the draft chains and sequence lengths of the whole batch and copies them to the
host once on demand. The last four steps are kept per request in
`__spec_history__` as `(step, row)` pairs.

The thinking-budget processor then handles one request per `draft_token_num` rows:

- Row `k` is evaluated on the committed outputs plus draft tokens `1..k`, so a
  `</think>` or newline inside the draft block is counted as it would be when
  decoded one token at a time.
- Committed tokens that are not yet in `req.output_ids` are rebuilt from the
  history: a step that grew the sequence by `c` committed `chain[1:c]` plus the
  bonus token that opens the next chain. Thinking markers in those tokens are
  therefore seen. When a committed token cannot be recovered, no row is forced.
- A block whose row count does not match the batch is skipped.
- A block of one row (zero draft steps) takes the same path.
- The draft chain and history are read back from the device only when the budget
  can be reached within the unseen span: the budget is not larger than that span,
  or the visible thinking block is already within it.

Scope: chain drafts (`--speculative-eagle-topk 1`), for example EAGLE MTP(3,1,4).
With a larger top-k the processors are skipped and one warning is logged; the
worker also logs a warning at startup for that configuration.

The script fails on an unknown `eagle_sample` layout or a partially patched file,
is idempotent, removes stale module bytecode and recompiles with checked source
hashes. A module without `eagle_sample` is left untouched. `Dockerfile.sglang.glm53`
applies it at build time; the overlay payload (`deploy/overlay`) ships it and
`infera-exec` applies it at container start. Drop the patch when the pinned
SGLang base applies custom logit processors in the EAGLE verify step.
