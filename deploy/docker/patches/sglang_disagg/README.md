# sglang PD disaggregation patches

Patches against the sglang source tree bundled in the ROCm engine images (sglang is an
editable checkout at `/sgl-workspace/sglang`, so `python/sglang/...` is what actually
runs — there is no separate site-packages copy to patch).

## `mooncake_early_send_wait_event.diff`

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

```bash
cd /sgl-workspace/sglang
git apply mooncake_early_send_wait_event.diff     # or: patch -p1 < ...
```

`examples/sglang_glm5.2/patch_sglang.sh` does this idempotently for both sglang patch
sets and is what the GLM-5.2 1P1D leg scripts expect to have been run.

### Verified

2× 8×MI325X (gfx942), ROCm 7.2.0, sglang v0.5.16, GLM-5.2-FP8 1P1D, mooncake RDMA,
overlap scheduling **on**, `chunked-prefill-size 131072` with `--enable-dp-attention`
(so 16384 per DP rank), `MEM_FRAC=0.80`. Logs confirm the failing prompt is still really
split into 4 chunks after the fix:

| Test | Before | After |
|---|---|---|
| needle, 3 lengths × depth 10/50/90 | 5/9 | **9/9** |
| needle, 29k depth sweep (5,20,35,44,50,56,62,75,95) | 4/9 | **9/9** |
| full `verify_correctness.py` suite | needle 7/9 | all checks pass |

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
