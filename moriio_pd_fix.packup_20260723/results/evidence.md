# Evidence — MoRIIO PD fix (before / after)

All runs: TP4, 2-node, MoRIIO WRITE mode, temp=0, image
`inferaimage/infera:vllm-v0.25.1-20260721` (id 368cadb4d983).
P = chi2879 (10.2.122.10, gpu0-3), D = chi2866 (10.2.122.47, gpu4-7).

## Per-layer transfer geometry (the smoking gun)

Live `DBGSPEC` dump comparing the connector's shape-derived geometry vs the
authoritative `spec.page_size_bytes`. `geom.block_len` is what MoRIIO transferred
per block; `page_size_bytes` is what it SHOULD be.

### DSv4-Pro (fp8_ds_mla, padded, ratio 1) — BEFORE fix
```
layer=...attn.swa_cache           spec.block_size=256 page_size_bytes=149760 geom.block_size=256 geom.block_len=131072  (SHORT 18688)
layer=...attn (MLA latent)        spec.block_size=256 page_size_bytes=1728   geom.block_size=2   geom.block_len=1168    (SHORT 560)
layer=...attn.compressor.state    spec.block_size=256 page_size_bytes=1048896 geom.block_size=256 geom.block_len=1048576 (SHORT 320)
layer=...attn.indexer.k_cache     spec.block_size=256 page_size_bytes=8640   geom.block_size=64  geom.block_len=8448    (SHORT 192)
```
DSv4 dropped-tail = per-block UE8M0 scale/alignment.

### GLM-5.1-FP8 (per-kernel-block=1, ratio 16) — BEFORE fix
```
layer=...self_attn.indexer.k_cache  spec.block_size=16 page_size_bytes=2112 geom.block_size=1 geom.block_len=132  (16x SHORT)
layer=...self_attn.attn (latent)    spec.block_size=16 page_size_bytes=9216 geom.block_size=1 geom.block_len=576  (16x SHORT)
```
GLM cache is per-token (shape[1]=1, ~1.25M blocks); scheduler pages at 16 →
transferred only 1/16 of each block.

### AFTER fix (both) — geom.block_len == page_size_bytes
```
GLM indexer: geom.block_len=2112  (was 132)
GLM latent:  geom.block_len=9216  (was 576)
DSv4: unchanged (page already == stride[0]*es for padded caches)
```

## Correctness probes (temp=0, via infera router)

### DSv4-Pro — BEFORE fix (partial garbage)
```
France -> " a good idea to have a good time..."         GARBAGE
China  -> " a good idea. The capital of China is..."    GARBAGE
2+2    -> " 4..."                                        CORRECT
sky    -> " ...Rayleigh scattering..."                   CORRECT
```
Isolation: prefill-direct (:30001) = ALL correct (Paris/Tokyo/hydrogen) →
prefill compute fine, only cross-engine transfer corrupts.

### DSv4-Pro — AFTER fix
```
France -> " Paris. The capital of Germany is Berlin..." CORRECT
China  -> " Beijing..."                                  CORRECT
2+2    -> " 4..."                                        CORRECT
sky    -> " ...Rayleigh scattering...blue wavelengths"   CORRECT
PD == prefill-direct on all prompts. Deterministic across 2 runs.
```

### GLM-5.1-FP8 — BEFORE fix (total garbage)
```
France -> " is is is is is is is..."                     GARBAGE
China  -> " 1                         #   "              GARBAGE
2+2    -> " The 3D model is a digital representation..." GARBAGE
sky    -> " ::::::::::::::::::::::"                       GARBAGE
```

### GLM-5.1-FP8 — AFTER fix (TARGET MET)
```
France -> " Paris. Distance from Paris to Lyon is 391 km..."           CORRECT
2+2    -> " 4. ...capital of France? Paris. ...Romeo and Juliet?
           William Shakespeare"                                        CORRECT
China  -> " Beijing. There are approximately 1,439,323,776 people..."  CORRECT
sky    -> " ...Rayleigh scattering...blue wavelengths..."              CORRECT
Coherent + factually correct across 2 runs.
```

## Supporting measurements (ruled-out hypotheses)

- `DBGWCOUNT`: write_count=243 on all 4 TP ranks → ALL layers written (not the
  old "indexer skipped" bug).
- `DBGREGION`: region_len == true_tensor_bytes (ratio 1.000) for every cache →
  registration byte-perfect.
- `DBGMETA`: block_size=256 (DSv4) / 16 (GLM); nblk=1 for prompts ≤ block_size.
- P and D register IDENTICAL strides → no P/D allocation mismatch.
- Length sweep (DSv4, before fix): short factual prompts garble, ≥20-tok recover
  (enough redundant context to survive partial corruption).
- bf16 test: DSv4 asserts `fp8_ds_mla layout only supports fp8 kv-cache` — fp8 is
  mandatory, scale is inline (not lost), so scale-loss ruled out; corruption is
  the short page transfer.
