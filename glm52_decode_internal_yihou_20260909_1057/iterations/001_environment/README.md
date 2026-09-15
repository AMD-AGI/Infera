# Iteration001: establish pinned container environment

## Hypothesis
The empty /dev/dri in Spur's exec namespace is a namespace visibility issue; a Docker sibling container launched by the host daemon should see all8 GPUs. The Docker image store uses containerd, so /var/lib/docker root-disk capacity is not the image-layer capacity.

## Method and evidence
- Allocation: squeue -j126175 reports yihou RUNNING crsuse2-m2m-055. No allocation requested/cancelled. Nodes234/036 never contacted.
- `probe.log`: own read-only Docker container saw8 MI355X,0% VRAM/use; torch import failed with no writable tempfile directory.
- `probe2.log`: changed only writable TMPDIR via workspace mount; torch import and8-device discovery pass. Read-only host containerd mount shows /dev/mapper/nvme_vg-nvme_lv,28TiB capacity,24TiB free.
- `image_load.log`: source archive SHA256/zstd pass; docker load returned pinned sha256:b9a83742f631... with command exit0.
- `scripts/create_container_yihou.sh`: reproducible owned-container launch, guarded by node/job/name. Preserves all pre-existing containers. Model mount read-only.
- `image_identity.log`: SGLang402df1e1, AITER2c71811b, torch2.9.1+rocm7.2.0.git7e1940d4,8 MI355X, driver6.14.14. AITER core JIT compiled successfully.
- `packup-recheck.json`: monitored input hashes unchanged since workspace snapshot; final audit results now present. Packup audit states offline PASS, not GPU benchmark validation.

## Outcome
Environment startup passed; benchmark has not yet run. Stopped probe containers retained (no deletion). Persistent task container is yihou-glm52-internal-20260909, holding only sleep until benchmark starts. Existing other-user idle container left untouched.
