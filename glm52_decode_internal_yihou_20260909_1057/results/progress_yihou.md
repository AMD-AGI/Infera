# Historical benchmark progress — superseded by report.md

## Verified setup
- Existing allocation126175, nodecrsuse2-m2m-055; no new allocation and no node234/036 access.
-8 AMD Instinct MI355X; driver6.14.14; torch2.9.1+rocm7.2.0.git7e1940d4.
- Image sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d; SGLang402df1e1e453e1e85ec0f5ac4052d36598cc691a; AITER2c71811b32c8ce2e1266aedaec199df7d90f597d.
- Model config hash8b46225f7afd7181735c2dd97b925bcb70dfebb72b1a105d4ac0be7583a5c726; index hashfd42188894abe9196fb70a2113c4fd1d0569b29a307a89c0e18e200111456fc6. Full weight-file identity not hashed.
- Packup7993-file manifest verification passed unchanged at2026-09-09T12:23Z.

## Completed experiment
Iteration002, eager smoke: bs16,ISL1024,OSL16,maxsteps2,warmup0. All8 ranks agree.128 useful output tokens,2 verify iterations,8tokens/request,final context1032,complete=false. Finite bootstrap asserted.8 FlyDSL decode and8 fused gfx950 DSA indexer engagement markers. No Scheduler/PD instances.

The99.52s loop duration contains first-use JIT and must not be used as steady-state performance. Two all-4 accept coins give realized length4.0, not an empirical estimate of the expected3.61 target. Native decode and draft graph runners were disabled for this smoke.

Observed pool capacity3,985,920 tokens exceeds1,280,000+reserve needed for the target workload. Actual target+draft pool initialization filled approximately193GB/rank at the default0.85 memory fraction; mapped smoke tokens were only17,408. Units in original runtime logs labeledGB may be GiB; byte-level fields are authoritative.

## Pending
Graph+warmup smoke, state consistency guards,70K resident-prefix smoke, full16x10000 output progression, bounded repeat and final report. No mission-completion or production-throughput claim yet.
