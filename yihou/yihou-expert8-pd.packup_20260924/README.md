# GLM-5.2 synthetic8-expert PD — reproduction kit

**Ran:**2026-09-23. **Packaged:**2026-09-24. **Status:**PASS operational smoke and fixed-length C32. No original files or running services changed during packaging; no git commit.

## Goal
Physically reduce routed experts256->8 to free weight memory for KV-cache experiments, while preserving attention/KV architecture and actually executing all8 retained experts. This is intentionally non-faithful synthetic modeling, not original GLM-5.2 quality or performance.

138 prefill +136 decode, GPUs0–3 each;TP4/DP4/DPA enabled/EP4,2 routed experts per rank; shared expert retained separately. MTP/HiCache/simulated acceptance OFF; triton DSA,FP8 KV, memory fraction0.85.

## Results
| Metric | Result |
|---|---:|
| Post-weight allocation per rank,P/D |31.43 /31.44 runtime GB|
| KV pool per rank,P/D |205.23 /210.74 runtime GB|
| KV tokens per rank,P/D |3,990,400 /4,097,408|
| Fixed-length measured requests |256/256,0failures;32warmup|
| Input/output length |4096/1024,each verified from server usage|
| Peak client concurrency |32|
| Measured duration |184.09s|
| Output throughput |1,424.02tok/s|
| Input+output throughput |7,120.09tok/s|
| MeanTTFT /TPOT |1.126s /20.87ms|

Equal-EP4 theoretical expert+gate byte saving87.06GiB/rank is **not a measured full256EP4 A/B**. No long-context/full-KV occupancy soak performed.

## Success criteria and evidence
Exact original criteria copied in `spec/mission.md`.
1. Actual8 allocation and route0–7:PASS component tests, physical layer logs and successful full forward. No discarded expert tensor materialized by synthetic iterator.
2. Matched-rail GPU RDMA bytes:PASS8/8 directions,40.11–42.44GB/s.
3.1P1D health, single and concurrent smoke:PASS. No native fault/traceback/NCCL error/NaN marker in successful engine logs; no semantic quality claim.
4. Fixed C32,256 successes,exact lengths,client exit0:PASS; archived288 per-request records include warmup.
5. Weight/KV capacity report:PASS; measured allocation and theoretical accounting clearly separated.

## Reproduce
Read `REPRODUCE.md`. Restore preserved scripts to a NEW shared workspace, obtain the exact image, prepare the model overlay, check free GPUs and fresh RDMA, launch, smoke, then one C32 point. Do not launch directly from payload or overwrite the original workspace.

## Navigation
- `REPRODUCE.md`: ordered cold-start commands and safety checks.
- `environment.md`, `environment/`: pinned image/source/driver/fabric evidence; later captures labeled.
- `payload/scripts/`: verbatim scripts that ran; `scripts/restore_workspace.yihou.py`: safe workspace relocation.
- `payload/source/`, `patches/`: exact source snapshots, minimal patch, image build provenance.
- `payload/tests/`: focused component/client tests.
- `results/`: benchmark,smoke,component,memory accounting,transfer evidence; per-request data gzip.
- `logs/`: approved gzip key logs, including failed setup and successful run.
- `spec/`: original mission/plan/narrative copied verbatim; some chronological notes retain earlier pending states.
- `notes.md`: fixes, rejected hypotheses, limitations.
- `copy-manifest.json`: source-copy mapping; `SHA256SUMS`: package integrity.

## Dependency gap
Final image has a local ID, not a pullable registry digest. Verified copies existed on136/138 at packaging; image export/load is documented. Historical Dockerfiles are included as provenance but do not guarantee a bit-identical rebuild. Models and image layers intentionally excluded; credentials must come from normal authorized cluster access.
