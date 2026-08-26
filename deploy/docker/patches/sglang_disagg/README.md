# sglang PD disaggregation patches

Patches against the sglang source tree bundled in the ROCm engine images (sglang is an
editable checkout at `/sgl-workspace/sglang`, so `python/sglang/...` is what actually
runs — there is no separate site-packages copy to patch). Each is a self-locating,
idempotent Python script, applied at image build time by the patch loop in
`Dockerfile.sglang` / `Dockerfile.sglang.gfx942`.

## `patch_responses_pd_bootstrap.py`

**Makes `POST /v1/responses` usable at all under PD disaggregation.** Without it every
Responses request to a worker started with `--disaggregation-mode prefill|decode` comes
back as

```
HTTP 400  Invalid request: Disaggregated request received without bootstrap room id
```

no matter what the client or the proxy in front of it sends. Nothing logs that anything
was dropped.

### Root cause

Three steps, none of which is visible from the outside:

1. `entrypoints/openai/protocol.py` gives `CompletionRequest` and `ChatCompletionRequest`
   a `bootstrap_host` / `bootstrap_port` / `bootstrap_room` trio. `ResponsesRequest` has
   no such fields and declares no `model_config`, so pydantic's default `extra="ignore"`
   **silently discards** a proxy's annotation.
2. `entrypoints/openai/serving_responses.py` builds its `GenerateReqInput` without
   mentioning bootstrap, so `bootstrap_room` is `None`.
3. `managers/scheduler.py`, in any disaggregation mode with a non-FAKE transfer backend,
   sees `recv_req.bootstrap_room is None` and calls
   `prepare_abort(..., status_code=HTTPStatus.BAD_REQUEST)`.

### What the patch does

| File | Change |
|---|---|
| `entrypoints/openai/protocol.py` | `ResponsesRequest` gains the three bootstrap fields, types copied verbatim from `ChatCompletionRequest` |
| `entrypoints/openai/serving_responses.py` | `create_responses` forwards them into `GenerateReqInput` |

That is the whole fix. `_make_request` already converts a `ResponsesRequest` into a
`ChatCompletionRequest` and runs it through the same `_process_messages` → prompt path as
the chat endpoint, so the generation pipeline was always shared; only the bootstrap
plumbing was missing.

### Why we need it

The Codex CLI/SDK speaks the Responses API by default
(`model_providers.<id>.wire_api = "responses"`), so every Codex-driven agentic workload
— Hyperloom's inference-optimizer among them — is locked out of a PD deployment.
Infera's Rust router registers `/v1/responses` and threads the bootstrap trio through
both legs exactly as it does for chat; this patch is the engine half of that path.

### What it deliberately does not do

**The harmony builtin-tool loop is not patched.** `serving_responses` has a second
`GenerateReqInput`, in `_generate_with_builtin_tools`, for the follow-up turn after a
server-side tool call. Under PD that turn needs its own bootstrap room and its own
prefill leg, and no proxy has arranged either — reusing the first turn's room would hand
the decode worker a room the prefill side already consumed, and it would block on KVPoll
until the ~300 s timeout. Left unpatched it fails fast with the same 400 instead.
Non-harmony models never reach it: `SimpleContext.need_builtin_tool_call()` returns
`False`, so the loop breaks after the first turn.

**Stateless calls only**, independently of this patch. `store` /
`previous_response_id` and the `GET /v1/responses/{id}` and `.../cancel` routes read a
per-process `response_store` dict, so they only resolve when the follow-up request lands
on the same worker. Behind any multi-worker router, clients must send `store: false`.

### Verified

Anchors present exactly once in the sglang v0.5.17 tree shipped in the mi35x engine
image. Applied against that tree: both files patched, a re-run reports `already present`,
both modules `py_compile` clean, and `ResponsesRequest.model_validate` round-trips
`bootstrap_host=10.0.0.1 bootstrap_port=9000 bootstrap_room=12345` while an aggregated
request with no bootstrap keys still yields `None, None, None`.

The runtime check is end-to-end: a Responses request through the router against a 1P1D
pair returns 200 instead of 400, and prefill and decode log the **same** `bootstrap_room`
— which is what proves the KV actually moved over Mooncake rather than the decode leg
quietly recomputing the prompt.

### Upstream

Not submitted. The gap looks like an oversight rather than a decision: the two older
request models grew the fields and `ResponsesRequest` was added later without them.
Worth filing. Drop this script once base sglang carries the fields — it then reports
"already present" and no-ops.

## `patch_mooncake_early_send_wait_event.py`

**Fixes silent KV corruption for chunked prefill over the mooncake PD transport.**
Without it, every prefill chunk except the last can be transferred to the decode leg
while the forward that writes those pages is still running. The decode leg gets
half-written KV for that part of the context.

The symptom is not a crash and not an obvious wrong answer — it is *partially* wrong
output for prompts longer than one prefill chunk. On GLM-5.2 DSA, needle-in-a-haystack
retrieval returns the first few digits of a 7-digit needle and then degenerates into
repeated `</think>`:

```
want=2183762  got='2183</think>2183</think>218</think>218</think> the'
want=7544440  got='7549.8.8.</think>7549. The</think>7549.8.8759</thi'
```

The corruption boundary lands exactly on the chunk boundary: a needle in the final chunk
is retrieved correctly, a needle in any earlier chunk is not. It only reproduces under PD
— the same model, same DSA backends, same chunk size in an aggregated single-node server
passes 9/9.

### Root cause

`prefill.py` records a CUDA event as a barrier before handing over pages that may still
be under write, and the comment there says exactly what it is for:

> *Record a completion event now so the transfer worker can wait on those writes before
> the RDMA read, instead of racing them.*

But **only the `mori` backend ever read that event** (`mori/conn.py`). `mooncake/conn.py`
had no `wait_event` / `synchronize()` anywhere, so on mooncake the barrier had never
taken effect. And under overlap scheduling the transfer that actually moves non-final
chunks (`send_kv_chunk(..., last_chunk=False)` in `process_batch_result_disagg_prefill`)
did not even record an event. The final chunk is always correct because it goes through
the sampling path, which already has a real `copy_done.synchronize()`.

### What the patch does

Three changes, all mirroring what `mori` already does:

| File | Change |
|---|---|
| `disaggregation/common/utils.py` | `TransferKVChunk` gains a `wait_event` field so the barrier can travel with the work item |
| `disaggregation/mooncake/conn.py` | `send()` picks up and forwards the event; `add_transfer_request()` accepts it; `transfer_worker` `synchronize()`s **before** reading device memory |
| `disaggregation/prefill.py` | the overlap-path non-final-chunk transfer, which had no barrier at all, now records an event on `forward_stream` |

This is **not** DSA-specific. Any PD deployment running chunked prefill over mooncake
with overlap scheduling is affected; DSA's sparse retrieval just makes it visible in the
conspicuous "retrieved half the digits" way instead of a quiet quality drop.

### Applying

The image build runs it; a live container that was started from an unpatched image can
run the same script by hand:

```bash
python deploy/docker/patches/sglang_disagg/patch_mooncake_early_send_wait_event.py
```

It locates sglang through `importlib`, so it needs no path argument, and re-running it
is a no-op ("already present"). The edits are anchored on source text rather than line
numbers, so they survive the offsets between sglang releases: the anchors are present in
both v0.5.15.post1 and v0.5.16, and the result is byte-identical to the hand-cut diff
this patch started as. An anchor that goes missing or stops being unique writes
**nothing** and fails — the `wait_event` handover is only correct if all three files
land, and a half-patched tree still corrupts long prompts.

### Verified

2× 8×MI325X (gfx942), ROCm 7.2.0, sglang v0.5.16, GLM-5.2-FP8 1P1D, mooncake RDMA,
overlap scheduling **on**, `chunked-prefill-size 131072` with `--enable-dp-attention`
(so 16384 per DP rank), `MEM_FRAC=0.80`. Logs confirm the failing prompt is still really
split into 4 chunks after the fix:

| Test | Before | After |
|---|---|---|
| needle, 3 lengths × depth 10/50/90 | 5/9 | **9/9** |
| needle, 29k depth sweep (5,20,35,44,50,56,62,75,95) | 4/9 | **9/9** |
| full correctness suite (needle + humaneval-long + long-context delta) | needle 7/9 | all checks pass |

It also removes the reason for the `CHUNK=524288` + `MEM_FRAC=0.80` workaround: chunk
size goes back to being a pure performance knob, and prefill overlap scheduling no longer
has to be sacrificed.

### Upstream

Not submitted yet. The closest existing report,
[sglang#25583](https://github.com/sgl-project/sglang/issues/25583) (GLM-5-FP8 + NSA +
70k prompt, identical symptom), was auto-closed with no follow-up. The aggregated-vs-PD
A/B above is the piece that report was missing.

The cost of the fix should be measured when upstreaming: the new
`wait_event.synchronize()` in the transfer worker blocks that thread, which in principle
trades some transfer overlap for correctness.
