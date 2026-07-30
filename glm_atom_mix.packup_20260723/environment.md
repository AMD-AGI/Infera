# Environment

## When
Bring-up + verification: **2026-07-23**. Single session, a few min bring-up.

## Hardware / node
| role | node | GPUs used |
|------|------|-----------|
| single-node (prefill+decode) | **chi2866** | card4-7 |

- **GPU:** AMD Instinct **MI355X** (gfx950), 8 per node, ~283 MB/card idle baseline.
- chi2866 is also the slurm jump host; card0-3 held foreign `titan` training
  (+ `zirui` / `primus-*` service containers) throughout — untouched. This run
  used card4-7 (`HIP/ROCM_VISIBLE_DEVICES=4,5,6,7`).
- **No RDMA fabric used.** Single-node = one server on one node; no ionic / MoRIIO
  / Mooncake / cross-node KV transfer.

## ⚠️ Disk-tight caveat (READ before reproducing)
chi2866 `/` (md0, 838G) runs **~96% full, ~34GB free**, shared with the foreign
`titan` container (263GB writable layer). The ATOM image is 45GB; **re-loading its
tar (`docker load -i /mnt/vast/c_huggingface/dsv4_repro_atom_img.tar`) briefly
filled `/` to 100% and hung ssh** — the node nearly went down. The image is ALREADY
loaded (`infera/engine-atom:kimi`), so reproduction does NOT need to load it. The
model is RO-mounted from /mnt/vast, so the running container's writable layer stays
tiny (disk held at 33-34G free during the whole run). Never `docker load`/`pull`/
`prune` on this node.

## Software
- **Docker image:** `infera/engine-atom:kimi` (image id
  `c7505e171e31`, 45.1GB). Provenance: staged tar
  `/mnt/vast/c_huggingface/dsv4_repro_atom_img.tar` (built ~11 days prior). ATOM
  base is `rocm/atom:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0_atom0.1.4_20260612`
  (the reference DSv4 kit's image); this `:kimi` tag is the infera-staged variant
  present on the node. `atom.entrypoints.openai_server` is the launch entrypoint.
- Note: the image was originally built targeting gfx942/MI300, but runs correctly
  on gfx950/MI355X for GLM here (plain decode correct — see notes.md).
- This packup is engine-image only (server launched directly via
  `atom.entrypoints.openai_server`). For the infera-native launch (via
  `python -m infera.engine.atom`) see the e2e case added alongside this kit
  (`tests/e2e/pd_mixed/atom/matrix.py`).

## Per-run env vars
```
HIP_VISIBLE_DEVICES=4,5,6,7
ROCM_VISIBLE_DEVICES=4,5,6,7
HSA_NO_SCRATCH_RECLAIM=1     # harmless on gfx950; mandatory on gfx942
AITER_LOG_LEVEL=WARNING
```

## Server config chosen
```
-tp 4 ; --kv_cache_dtype fp8 ; --server-port 8000
cudagraph capture [1..512] ; post-init VRAM ~91.6%
NO --method mtp (GLM has no MTP/nextn draft weights; gfx950 plain decode correct)
```

## External dependencies (absolute paths, not in repo)
- **GLM weights:** `/mnt/vast/xiaobo/models/GLM-5.1-FP8` (VAST shared mount, same
  path in-container, RO).
- **Shared work dir:** `/mnt/vast/c_huggingface/` (scripts, `glm_atom_mix.log`,
  the staged atom tar). `/mnt/vast` is shared VAST; `/tmp` is NOT shared.

## Required secrets (names only — no values)
- **Cluster SSH:** ProxyJump preconfigured in `~/.ssh/config` (`ssh chi2866`).
- **Docker image:** present on the node; the staged tar is on shared VAST.

## Not captured (honest gaps)
- Exact host kernel / ROCm driver point-versions not snapshotted. This is a
  correctness bring-up (no perf number), so kernel drift should not change the
  verdict.
- No throughput/latency numbers taken — correctness bring-up only.
