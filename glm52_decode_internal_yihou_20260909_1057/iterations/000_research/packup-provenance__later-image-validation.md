# Later independent image-load validation — 2026-09-09

This addendum is **not part of the2026-09-08 mix measurement**. The original history remains unchanged. The coordinator explicitly supplied these two later evidence files for read/copy; no other files from the later experiment were inspected or copied.

- `later-validation/image_load.log`: `Loaded image: rocm-llm-bench:latest`.
- `later-validation/image_identity.txt`: captured2026-09-09T10:33:59Z on crsuse2-m2m-036; image `sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d`, SGLang `402df1e1e453e1e85ec0f5ac4052d36598cc691a`, AITER `2c71811b32c8ce2e1266aedaec199df7d90f597d`.
- Source paths, copy hashes, sizes and mtimes are in `later-validation-files.json`.

Coordinator reports the command `zstd -dc /shared_nfs/yihou/playground/glm52_mix_repro_20260908/image/rocm-llm-bench.tar.zst | docker load` completed with exit0 at approximately10:32Z; image identity was inspected and source HEADs queried inside a temporary container. The supplied log confirms the loaded tag and the supplied identity file records exact pins; the original command line and exit status are coordinator-reported rather than contained in these two small files. This package's author did not execute that load.

The later evidence closes the previously untested archive round-trip identity gap **as a later validation**. It does not establish a92GB requirement, historical node153/168 Docker backing-store capacity, complete runtime equivalence on a new host, or a new mix benchmark result. Coordinator-reported node036 DockerRootDir/NVMe capacity is not needed to establish image identity and is not promoted to historical host evidence.
