# The PD decode leg writes to kvd and never reads back

**Status:** confirmed by source read + measurement. Acted on (decode leg no longer
wires kvd). **One follow-up left open** — see [Open question](#open-question).

**sglang:** `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` (release/v0.5.15)
**Found:** 2026-07-31, during the kvaware+kvd × MTP merge validation.

## Claim

On a PD **decode** worker, infera's kvd (SGLang hierarchical cache L3) is
**write-only**: the backup path runs and fills L3, but nothing on that leg ever
issues a storage prefetch, so the bytes are never read back. This is a property of
**PD decode**, not of MTP — it holds in every configuration we can run.

## Evidence 1 — source: the read path is not wired on the decode branch

`python/sglang/srt/managers/scheduler.py:2309`

```python
def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
    if not self._set_or_validate_priority(req):
        return
    if self.disaggregation_mode == DisaggregationMode.NULL:
        self._prefetch_kvcache(req)              # aggregated
        ...
    elif self.disaggregation_mode == DisaggregationMode.PREFILL:
        self._prefetch_kvcache(req)              # prefill leg
        ...
    elif self.disaggregation_mode == DisaggregationMode.DECODE:
        self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
        # <-- no _prefetch_kvcache call
```

`_prefetch_kvcache` (`scheduler.py:2284`) is the **only** caller of
`tree_cache.prefetch_from_storage(...)`, and it is itself called from exactly two
places, both inside `_add_request_to_queue`:

```
$ grep -n "_prefetch_kvcache" python/sglang/srt/ -r
scheduler.py:2284:    def _prefetch_kvcache(self, req: Req):
scheduler.py:2315:            self._prefetch_kvcache(req)   # NULL
scheduler.py:2319:            self._prefetch_kvcache(req)   # PREFILL
```

The decode branch has no equivalent. The write side (`HiRadixCache`'s backup /
`write_backup_storage`) is driven from the cache controller and keeps running,
which is why L3 fills.

## Evidence 2 — measurement: 180 sets, 0 gets on the decode leg

G0 of the merge validation (`work.merge_20260731`), two-node PD over mooncake
RDMA, DP-attention 8/8 both legs, kv-aware ON, kvd ON on **both** legs, each with
its own daemon. MTP **off** — so `--disaggregation-decode-enable-radix-cache` was
appended and the decode leg had a full `HiRadixCache`, the most favourable case:

| leg | node | sets | gets | hits | bytes |
|---|---|---:|---:|---:|---:|
| prefill | chi2879 | 102 | **102** | **102** | 180 MB |
| decode | chi2867 | 180 | **0** | **0** | 318 MB |

The prefill numbers are from the restart-and-replay attribution test (restart the
engine to empty the GPU radix cache, replay the same prefixes: 102 reads, zero new
writes). The decode leg was never restarted, so on its own `gets=0` would be
ambiguous — the in-GPU cache could simply have absorbed everything. Evidence 1 is
what removes the ambiguity: there is no code path that could have produced a get.

Cost of the write-only traffic on this run: 318 MB of host memory and the PCIe
D2H bandwidth to move it, for zero reads.

## Why this is not an MTP-specific finding

The merge hit it from the MTP direction, via a different (real) restriction:

1. infera appends `--disaggregation-decode-enable-radix-cache` on the decode leg
   when KV events are on (`infera/engine/sglang/args.py:255`).
2. SGLang rejects that flag under speculative decoding
   (`arg_groups/pd_disaggregation_hook.py:41`), so the leg dies at argument
   parsing → forces the flag off → `disable_radix_cache = True`
   (`pd_disaggregation_hook.py:56`).
3. SGLang then rejects `enable_hierarchical_cache and disable_radix_cache`
   (`server_args.py:5772`), which infera's kvd wiring always sets
   (`kvd_wiring.py:51`). Decode leg dies again.

That chain made it *look* like "kvd and MTP are incompatible on the decode leg".
Evidence 1 shows the weaker precondition already fails: even with MTP off and the
radix cache present, the decode leg cannot read L3. Restriction (3) traces to
upstream [#9452](https://github.com/sgl-project/sglang/pull/9452) — hicache is
implemented as a `RadixCache` subclass and its keys are radix-node hash values,
so a `ChunkCache` has nothing to key on. That is an implementation dependency,
not a law, but it is not the reason the decode leg reads nothing.

## What was done

`patches/patch_infera_decode_kvd_skip.py` — infera does not wire kvd on a PD
decode leg, and logs why. Prefill keeps kvaware + kvd fully on.

Superseded `patch_infera_decode_radix_vs_mtp.py`, which only narrowed the
`--disaggregation-decode-enable-radix-cache` append to exclude speculative
decoding. That patch fixed the crash but left the write-only traffic in place, and
was scoped to MTP when the underlying problem is not.

## Open question

`disaggregation_decode_enable_offload_kvcache` drives a *separate* decode-side
mechanism, `DecodeKVCacheOffloadManager` (`disaggregation/decode.py:1984`,
constructed at `scheduler.py:465`), which requires `hicache_storage_backend` to be
set (`server_args.py:5779`). **We never enable it**, and it was not exercised in
any run here.

Whether that manager reads back from L3 — and therefore whether "decode leg never
reads kvd" needs an exception carved for it — has **not** been checked. Our patch
skips kvd wiring on any decode leg, which would also disable that path. If someone
wants `--disaggregation-decode-enable-offload-kvcache`, read
`DecodeKVCacheOffloadManager` first and re-scope the patch.

## Upstream

Not reported. Worth raising as a question rather than a bug report — "is
write-only L3 on the PD decode leg intended?" — because the fix could reasonably
be either wiring the prefetch or documenting that decode should not enable
hicache storage.
