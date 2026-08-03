# The bigram kv-event path is NOT exercised by this deployment — measured

This corrects a claim in a sanctioned kit and prevents this run from repeating
it. It costs nothing to state and would be misleading to omit, because the
router's cache view here is large and healthy — which *looks* like proof of the
bigram fix and is not.

## The claim being corrected

`glm52.merged_branch_image.packup_20260801/README.md`, "The two numbers that
actually discriminate":

> **The bigram fix produces the *right* hashes.** `is_eagle` **is a global
> server arg**, so with MTP on the prefill leg's kv-events carry bigram pairs.
> Unfixed, the router hashes the pairs and the view reads **0**. It reads
> **51 blocks — identical to the plain-int path in G0**…

That kit's G1 launched **`MTP=0` on prefill and `MTP=1` on decode** — the same
split used here. So on its own configuration, the premise "the prefill leg's
kv-events carry bigram pairs" needs to be true for its 51-block reading to mean
what it says.

## What the wire actually carries

`kv_event_wire_probe.py` subscribes to the real ZMQ sockets on all 8 DP ranks
per leg and decodes with the router's own msgspec structs — so this is the exact
byte stream the router consumes, not a reconstruction. Driven with 4 × ~17K-token
requests through the router:

    PREFILL  tcp://10.245.157.89:17568
    rank   batches   stored  int-view  pair-view  first sample
    0           12     1055      1055          0  INT bs=64 [154822, 154824, 154826]
    TOTAL       12     1055      1055          0

    VERDICT: PLAIN-INT view on the wire
             -> the bigram path is NOT exercised by this leg.

1,055 `BlockStored` events, **every one plain `int`**, zero bigram pairs.

(The decode leg published no `BlockStored` at all — expected: under speculative
decoding infera deliberately does not append
`--disaggregation-decode-enable-radix-cache`, so the decode leg runs SGLang's
chunk cache and contributes nothing to the router view. Its
`policy_cache_view_size` reads 0 on all 8 ranks, consistent.)

## Why — the mechanism, read in the running image

`is_eagle` is **not** a global server arg. It is derived per engine from that
engine's own speculative configuration:

    sglang/srt/mem_cache/kv_cache_builder.py:211
        is_eagle=spec_algorithm.is_eagle(),

    sglang/srt/mem_cache/radix_cache.py:288
        self.is_eagle = params.is_eagle
    radix_cache.py:393,424
        key, _ = key.maybe_to_bigram_view(self.is_eagle)

In this deployment (and in the kit's own G1) the **prefill** leg runs
`speculative_algorithm=None` — confirmed in its `server_args` line — so its radix
cache is constructed with `is_eagle=False`, `maybe_to_bigram_view` is a no-op,
and its kv-events are plain ints. The leg that *is* EAGLE (decode) is precisely
the leg that publishes no `BlockStored` events under PD.

So the bigram view can only appear if the **prefill** leg itself runs MTP
(`PREFILL_MTP=1`), which no configuration here does.

## What this changes, and what it does not

**Does not change:** the fix is correct, present in this image's bytecode, and
covered by the branch's unit tests (`tests/unit/router/test_kv_event_bigram.py`),
whose behavioural coverage was verified upstream by reverting the fix and
watching the real-socket test fail `0 vs 2`.

**Does change:** what this run's cache view is evidence *for*. The measured
prefill view of **14,723 blocks** proves kv-aware routing is live and hashing
correctly — but against the **plain-int** path. It is **not** evidence about
bigram decode, because that code never executes here.

Stated the other way: had the bigram fix been absent from this image, **every
number in this experiment would be unchanged**.

## The kv-aware proof that this run *can* make

Not "the view is non-zero" (it would be non-zero for uninteresting reasons), but
an exact cross-check between two independent subsystems — the router's own hash
count and the engine's usage accounting:

| router pick log | engine `usage` | check |
|---|---|---|
| `cache_hits=937` | `cached_tokens=59,968` | 937 × 64 = **59,968** ✓ |
| `cache_hits=93`  | (early, partially warm)  | — |

`block_size=64`, from the worker registration. The router's block hashes chain
to the same prefix the engine independently reports as cached, to the token. A
router hashing the wrong view could not produce that identity.

Probe output: `results/wire_prefill.txt`, `results/wire_decode.txt`.
