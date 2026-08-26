# sglang Responses API patches

Patches against the sglang source tree bundled in the ROCm engine images (sglang is an
editable checkout at `/sgl-workspace/sglang`, so `python/sglang/...` is what actually
runs — there is no separate site-packages copy to patch). Each is a self-locating,
idempotent Python script, applied at image build time by the patch loop in
`Dockerfile.sglang` / `Dockerfile.sglang.gfx942`.

This set holds fixes to `POST /v1/responses` that are **not** disaggregation-specific.
The PD-only half lives in `../sglang_disagg/patch_responses_pd_bootstrap.py`; a Kimi-K3
PD deployment needs both, and they apply in either order (disjoint anchors, different
functions).

## `patch_responses_custom_encoder_prompt.py`

**Makes `POST /v1/responses` answer at all on a model with a custom chat encoder.**
Without it, every Responses request to a Kimi-K3 server — aggregated or PD, it makes no
difference — comes back in ~5 ms as

```
HTTP 400  texts cannot be empty and tokenizer must be initialized
```

while the same prompt through `POST /v1/chat/completions` answers normally. The error
names the tokenizer, which is a red herring: the tokenizer is fine, the prompt handed to
it is the empty string.

### Root cause

`OpenAIServingResponses` subclasses `OpenAIServingChat` and reuses its
`_process_messages`, but not its prompt-selection ladder.

1. For a model whose `chat_encoding_spec` is `"kimi_k3"` or `"inkling"`,
   `_process_messages` takes the custom-encoder branch: the encoder returns
   **pre-rendered token ids**, and the local `prompt` string is never assigned, so
   `MessageProcessingResult.prompt` keeps its `prompt = ""` initial value. That is by
   design — the ids are the authoritative rendering, the text is not.
2. `OpenAIServingChat.create_chat_completion` knows this. Its `prompt_kwargs` ladder
   routes `kimi_k3` (and a non-empty-ids `inkling`) to `{"input_ids": ...}` **before**
   the generic `is_multimodal` arm that would take `{"text": ...prompt}`.
3. `OpenAIServingResponses._make_request` does not. It branches on `is_multimodal`
   alone. Kimi-K3 is multimodal (`KimiK3ForConditionalGeneration`), so `engine_prompts`
   becomes `[""]`, `create_responses` turns a `str` engine prompt into
   `GenerateReqInput(text="")`, and `managers/tokenizer_manager.py` raises on
   `if not texts or ...` — `""` is falsy.

The `ValueError` is swallowed by the blanket `except ValueError` in `create_responses`
and returned as a 400 with no traceback, which is why nothing in the engine log explains
it.

### What the patch does

| File | Change |
|---|---|
| `entrypoints/openai/serving_responses.py` | `_make_request` replicates the chat endpoint's selection: custom encoders take `prompt_ids`, everything else keeps today's behaviour |

Multimodal models **without** a custom encoder are untouched and keep the text path.
`request_prompts` is only read by `_generate_with_builtin_tools`, which ignores it on the
non-harmony path, so switching it to ids changes nothing else.

### Verifying

Source-level, at build time: the script itself fails the build if its anchor drifted.
Bytecode-level, because a stale `__pycache__` silently reverts a source edit:

```sh
f=$(python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__))")/srt/entrypoints/openai/serving_responses.py
d=$(dirname "$f"); rm -f "$d/__pycache__/serving_responses."*.pyc
python3 -c "import py_compile; py_compile.compile('$f', doraise=True)"
strings "$d"/__pycache__/serving_responses.*.pyc | grep -c _infera_responses_custom_encoder_ids   # expect 1
```

At runtime:

```sh
curl -s $EP/v1/responses -H 'Content-Type: application/json' \
  -d '{"model":"kimi-k3","input":"1+1=?","store":false,"max_output_tokens":32}'
```

200 with a non-empty `output[]` instead of the 400 above, and prompt token accounting
matching the equivalent chat request.

### Upstream

Worth filing, not yet submitted. It reads as an oversight: the custom-encoder arms were
added to `create_chat_completion`'s ladder and `_make_request` was not revisited, so the
two paths silently disagree about which field of the shared `MessageProcessingResult` is
authoritative. Drop the script once base sglang routes both through one helper — it then
reports "already present" and no-ops.
