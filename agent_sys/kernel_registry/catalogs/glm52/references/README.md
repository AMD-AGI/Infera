# PyTorch references

`naive.py` expresses six operator contracts without importing a GPU framework
beyond PyTorch. It is newly authored for this catalog; historical correctness
results belong to the exact preserved `payload/tasks/*/driver.py` oracles.

| Operator | Entry point | Inputs and output |
|---|---|---|
| sparse MLA | `sparse_mla` | Queries, flattened latent KV and selected physical slots; FP32 output plus log2 LSE |
| DCP sparse MLA | `sparse_mla(..., lengths=...)` | Per-row owned slot counts; empty rows return zero output and -2**30 LSE |
| DSA indexer | `dsa_indexer` | FP8 queries, preshuffled paged cache and head weights; top-k physical slots, -1 padded |
| DCP indexer | `dcp_indexer` | One rank's local inputs plus other ranks' packed candidates; merged global top-k slots |
| MXFP4 MoE | `moe_mxfp4` | Packed E2M1 weights/UE8M0 scales, routes and route weights; FP32 routed SwiGLU output |
| fused indexer | `fused_indexer` | Projection/normalization/RoPE inputs and cache; updates the selected cache slots and returns top-k slots |

The fused reference includes the BF16 rounding boundaries, normalized Hadamard,
power-of-two FP8 scales and physical cache layout. Its FP32 GEMMs and reductions
do not emulate HIP instruction accumulation order. PyTorch top-k tie ordering
also need not match production. Consequently this is a readable semantic
reference, **not a replacement for the historical strict equality oracle**.
GPU differential validation of these newly authored references is outstanding.

CPU controls cover hand-computable attention, empty rows, physical page mapping,
remote DCP candidate merging, MXFP4 ties, routed MoE, RoPE/Hadamard bases and
fused cache write boundaries:

```bash
python -m pytest references/test_naive.py -q
```

Run from the catalog directory in an environment containing PyTorch and pytest.
The controls require no GPU and do not benchmark or certify imported candidates.

The preserved source task drivers remain the reproduction entry points for the
historical microbenchmarks. Their external fixture locations and checksums are
listed in `../external_fixtures.json`.
