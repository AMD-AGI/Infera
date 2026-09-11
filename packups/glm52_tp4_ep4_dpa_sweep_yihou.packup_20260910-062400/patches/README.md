# Changes and how they apply

No SGLang/AITER core patch was made. Delivered bench/scripts already contain all changes; do not apply provenance/tracked_changes.diff on top of them.

| Change | Why | Application/context |
|---|---|---|
| DecodeTopology/local graph batch/ParallelState | Old wrapper onlyDP1EP1;DPA requires real4replica topology | Inline bench/topology.py and profile_decode.py |
| Base DP token metadata | Pinned ForwardBatch scales speculative phases internally | Inline before bootstrap/outer decode;do not multiply by6 again |
| Replica-aware aggregation | Avoid4x double-counting TP outputs or missing DP outputs | Global result plus separate rank/DP reports |
| Shared coin progress guard | Divergent DP completion can deadlock collective loop | 7-int collective each outer iteration,included timing |
| Explicit JOB_ID/NODE/CONTAINER | Old scripts pinned stale allocation/path | Inline scripts,forbidden-node/ownership guards,dry-run |
| Source+launch provenance | Track dirty/untracked code and full wall cost | Per-run HEAD,diff,hashes,bench snapshot,launch_status |

New-file originals are complete under scripts/original. Baseline tracked diff is in provenance for review;all frozen source versions are hashed. Existing fallback support limits were recorded rather than patched. Packaging only relocates files and writes documentation/audit utility;no runtime change or GPU rerun.
