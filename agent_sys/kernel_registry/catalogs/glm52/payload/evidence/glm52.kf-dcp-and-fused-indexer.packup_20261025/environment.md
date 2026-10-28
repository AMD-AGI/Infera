> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Environment (2026-10-23 12:50Z - 10-25 05:xxZ)

Image, sglang overlay, model, harness, bench: identical to `../glm52.kf-decode-kernels-e2e.packup_20261023/environment.md`
(`historical-sglang-image-a`, ID `sha256:5b6565ef3e4e...`, sglang overlay 93034eba91, AgentX C56 3600 s,
sim acceptance 3.61).

| node | slurm job | IP | rounds |
|---|---|---|---|
| compute-node-05 | REDACTED | node-05.invalid | kda1, kdc1, kde1, dcpk kernel run |
| compute-node-07 | REDACTED | node-07.invalid | kdg1, kfb3 |
| compute-node-02 | REDACTED | node-02.invalid | kdf1, kfc1, fik kernel run, dcpreal |
| compute-cluster compute-node-11 | REDACTED | - | KF campaigns (fused_indexer) |

[Private orchestration details omitted.]

## KF candidates (`candidates/`; overlay copies listed in `overlay/MD5SUMS`)
| kernel | in overlay | best KF-measured | e2e |
|---|---|---|---|
| DCP sparse MLA decode split-K (`kf_dcp_sparse.py`) | iter 5 6af13db | 1.691x (cabd158) | kdc1, kdg1 |
| DCP split indexer top-k (`kf_dcp_indexer.py`) | iter 4 09f754e | 2.389x (09f754e) | kdc1, kde1 |
| gfx950 fused decode indexer (`kf_fused_indexer.py`) | 1748c11 (1.33x) | 1.40x (8dcc5a1, not in overlay) | kfc1 |
| DPA sparse / indexer / MoE | as 10-23 pack-up | | kfb3, kfc1 |

External data (not packed): real DCP indexer inputs `REDACTED_LOCAL_PATH` (1.6 GB);
full round dirs `REDACTED_LOCAL_PATH`; KF campaign repos `REDACTED_LOCAL_PATH`.
[Private orchestration details omitted.]
