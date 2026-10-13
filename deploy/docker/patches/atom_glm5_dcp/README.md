# `patches/atom_glm5_dcp/`: GLM-5.2 DCP PD fixes for ATOM

Opt-in source patches for GLM-5.2 PD serving with DCP, MTP and LMCache CPU
offload, written against ATOM `d9f0720e2f99168f6f4e5dd07d674cba5b68d610` and
its LMCache 0.5.5rc3 (`rocm/atom-dev:nightly_202609221542`). `Dockerfile.atom` runs them only with
`ATOM_GLM5_DCP_PATCHES=1`, which also pins `protobuf<7` for that base. The
release build for this stack is:

```bash
docker build -f deploy/docker/Dockerfile.atom \
  --build-arg ATOM_BASE_IMAGE=rocm/atom-dev:nightly_202609221542 \
  --build-arg BUILD_MOONCAKE=0 \
  --build-arg ATOM_GLM5_DCP_PATCHES=1 \
  -t <tag> .
```

## Patches

Each script's docstring has the full rationale.

| script | files | fixes | needed for 4P4D DCP |
|---|---|---|---|
| `patch_dcp_kv_staging.py` | `atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py`, `atom/model_ops/attentions/aiter_mla.py` | per-token RDMA descriptors for DCP latent MLA KV; stage on GPU, send whole pages | yes (performance) |
| `patch_dcp_prefill_mtp_metadata.py` | `atom/spec_decode/eagle_proposer.py` | MTP draft after a DCP prefill crashes on `None` `dcp_token_block_tables` | yes (crash) |
| `patch_moe_eager_decode_pad.py` | `atom/model_ops/moe.py` | DP attention + eager decode hands MoE unpadded rows | no; carried to match the experiment image |
| `patch_lmcache_lookup_pin.py` | `lmcache/v1/cache_engine.py` | `retrieve()` and `lookup_unpin()` both release a CPU lookup pin, so a prefix shared by two requests becomes evictable early | yes, with LMCache CPU offload (correctness of the CPU tier) |

On `d9f0720`, the existing `patches/atom/` loop followed by the three ATOM
scripts gives a tree byte-identical to that loop followed by `git apply` of the
experiment diffs. The LMCache script is new (the experiment image did not have
it): a backport of the `cache_engine.py` hunk of LMCache#5098. The experiment image did not run that loop, so the release image also
carries its `mooncake_consumer_slot` fix and hunk A of `minimax_m2_qknorm_rope`
(hunk B does not apply on `d9f0720`); both compose with these patches.

## Contract

The scripts share `anchor_patch.py`. All files of a patch resolve under one
package root: the importable `atom` (else `/app/ATOM/atom`), or `lmcache` for
the LMCache script. Every anchor must occur exactly once before any file is
written; a missing or repeated anchor, or a missing file, exits 1 and leaves
the package unchanged. A file that already holds the patch's marker is
skipped, so a rerun is a no-op. Unlike `patches/atom/`,
which swallows failures, a base that drifted from `d9f0720` fails the build.

## Dropping them

Re-check each patch whenever `ATOM_BASE_IMAGE` moves. As of 2026-10-13,
ROCm/ATOM `main` publishes the DCP token table unconditionally
(`_publish_dcp_token_block_tables`), so the MTP script fails there by design:
drop it. `main` still has the per-token connector path and the unpadded eager
MoE path; read the new source before dropping either of the other two. Drop
the LMCache script once the base's LMCache contains LMCache#5098 (`bdb66b8`);
its marker comment then matches and the script reports it as applied.
