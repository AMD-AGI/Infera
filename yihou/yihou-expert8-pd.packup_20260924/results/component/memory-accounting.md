# Metadata-only checkpoint memory accounting

Read282 safetensors JSON headers (data_offsets and shapes), without materializing weight tensors or using GPUs. Exact bytes are in memory-accounting.json; values below use GiB=2^30 bytes.

| Category | Original GiB | Retained GiB |
|---|---:|---:|
| Target routed experts | 358.5938 | 11.2061 |
| Target router gate/bias | 0.2198 | 0.0069 |
| Target shared experts | 1.4008 | 1.4008 |
| Target attention | 24.3421 | 24.3421 |
| Target dense MLP | 1.2656 | 1.2656 |
| Target embedding/norm/head/other | 3.5467 | 3.5467 |
| Excluded draft | 18.5388 | 0 |

The retained target checkpoint totals44,848,135,008 bytes. Routed MoE retains3,600 of115,200 tensors, exactly1/32 of routed expert bytes; the shared expert is unchanged.

For a theoretical original256EP4 versus synthetic8EP4 comparison, even expert partitioning saves93,251,174,400 routed-weight bytes per rank. Replicated router gate/bias slicing adds228,631,200 bytes/rank, total93,479,805,600 bytes = **87.0599 GiB/rank**.

This is checkpoint tensor-byte accounting, NOT measured VRAM. It excludes runtime padding, weight repacking/dtype conversion, kernel workspaces, graphs and temporaries. Disabling draft is accounted separately and must not be attributed to reducing the target expert count. Actual weight/KV memory is reported by full engine startup.
