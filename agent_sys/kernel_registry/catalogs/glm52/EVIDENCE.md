# Measurement interpretation and remaining gaps

The catalog imports historical observations. No GPU benchmark or serving run was
performed during import. All historical runs remain `reproducibility: incomplete`;
published redacted bytes are hash-verifiable independently. Historical results
refer to the original bytes, not the renamed public adapters; see [REDACTION.md](REDACTION.md).

| Kernel | Integrated source | Historical KF speedup | Interpretation |
|---|---|---|---|
| sparse MLA | iter14 64fbeaa2 plus SGLang import adaptation | 3.165172x | Overlay source has its own version; the KF claim belongs to the pre-adaptation source |
| DSA indexer | iter15 c2cacf5f | 3.153284x | Synthetic campaign correctness; target-stack top-k checks also retained |
| MXFP4 MoE | iter14 9cd51e60 | 1.104899x | EP1; later iterations 17 and 18 were correct but reverted for performance |
| fused indexer | iter4 1748c11 | 1.328203x | Target-stack driver median ratio 1.359080x; iter10 1.399768x was not integrated |
| DCP sparse | iter5 6af13db | 1.639143x | Recovered from retained publication; best iter8 1.691238x was not integrated |
| DCP indexer | iter4 09f754e5 | 2.388587x | Synthetic-only claim; real-input replay has mismatched top-k sets and slow cases |

KF takes the minimum across repeated measurements of the mean of scored-case
latency ratios. Top-level aggregate case timings do not reproduce that score.
The imported run uses the selected repetition's actual pairs and retains all
repetitions as raw evidence. Unscored cases remain in the raw benchmark files.
Target-stack logs have rounded printed timings and use their own named metrics.

E2E results name the entire installed kernel bundle. DPA kfb1/kfb2 versus the
same-node kfa2/kfa1 controls report mean TPOT around 11.2–11.3 ms versus 14.7–15.3 ms.
They do not establish per-kernel attribution or full-model correctness. DCP sparse
alone was measured on another node from baseline. Fused kfc1 versus kfb3 also uses
different nodes and does not establish a gain.

The pd llr3-to-llr7 summary reports +10.4% score and TPOT 19.8 to 15.4 ms, but also
changes custom-all-reduce/top-k configuration and increases mean input length 3.6%.
The llr42 fused trial reports 47,674 versus baseline mean 48,408, a single trial;
live kernel CHECK was exact, while real-acceptance correctness of the underlying
overlap/custom-all-reduce stack was not established. The earlier llr22 fused trial
was a module-path no-op. These qualifications remain in the corresponding records.

Known reproduction gaps include complete package inventories, per-run host driver
and OS captures, model/checkpoint content identity, exact original target-image
source digests, and some historical commands/timestamp timezones. Full immutable
image IDs were recovered for the three e2e image variants. Image identity alone does not
ensure the image remains retrievable.

Large workload fixtures retain exact SHA256 and sizes but currently have
unhosted `urn:sha256:` identities. Real DCP captures were not available for import and have no recovered content digest. Later pd runs preserve relevant
configuration/results, but not every run has a separately proven adapter snapshot.

The newly authored PyTorch references provide readable operator semantics and
CPU controls. They have not been compared against GPU execution. In particular,
the fused reference does not emulate HIP accumulation order, and top-k tie order
can differ. Historical passes describe the original oracle. Its published redacted snapshot
is not a new correctness verdict.
