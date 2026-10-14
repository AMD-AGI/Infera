# `patches/atom_dcp/`: DCP PD fixes for ATOM

Opt-in source patches for PD serving with DCP, MTP and LMCache CPU
offload, written against ATOM `d9f0720e2f99168f6f4e5dd07d674cba5b68d610` and
its LMCache 0.5.5rc3 (`rocm/atom-dev:nightly_202609221542`). `Dockerfile.atom` runs them only with
`ATOM_DCP_PATCHES=1`, which also pins `protobuf<7` for that base. The
release build for this stack is:

```bash
docker build -f deploy/docker/Dockerfile.atom \
  --build-arg ATOM_BASE_IMAGE=rocm/atom-dev:nightly_202609221542 \
  --build-arg BUILD_MOONCAKE=0 \
  --build-arg ATOM_DCP_PATCHES=1 \
  -t <tag> .
```

## Patches

Each script's docstring has the full rationale.

| script | files | fixes | needed for 4P4D DCP |
|---|---|---|---|
| `patch_dcp_prefill_mtp_metadata.py` | `atom/spec_decode/eagle_proposer.py` | MTP draft after a DCP prefill crashes on `None` `dcp_token_block_tables` | yes (crash) |
| `patch_moe_eager_decode_pad.py` | `atom/model_ops/moe.py` | DP attention + eager decode hands MoE unpadded rows | no; carried to match the experiment image |
| `patch_lmcache_lookup_pin.py` | `lmcache/v1/cache_engine.py` | `retrieve()` and `lookup_unpin()` both release a CPU lookup pin, so a prefix shared by two requests becomes evictable early | yes, with LMCache CPU offload (correctness of the CPU tier) |

On `d9f0720`, the existing `patches/atom/` loop followed by the two ATOM
scripts gives the same tree as that loop followed by `git apply` of the
experiment diffs, minus the experiment's latent-MLA KV staging in the Mooncake
connector. That staging only runs when a DCP1 producer feeds a DCP>1 consumer;
with DCP on both legs ATOM sends whole blocks, so it only cost producer memory.
The LMCache script is new (the experiment image did not have
it): a backport of the `cache_engine.py` hunk of LMCache#5098. The experiment image did not run that loop, so the release image also
carries its `mooncake_consumer_slot` fix and hunk A of `minimax_m2_qknorm_rope`
(hunk B does not apply on `d9f0720`); both compose with these patches.

## AITER kernel prebuild

`prebuild_aiter_mha_varlen.py` is not a source patch and runs after them. The
base prebuilds five AITER `mha_varlen_fwd` variants but not
`..._bf16_nlogits_nbias_mask_lse_ndropout_nskip_nqscale`, which ATOM's chunked-MLA
prefill calls once a prompt extends a cached prefix. Its first call JIT-compiled
it for ~30 s, and in a 4P4D AgentX run that compile stalled the prefill inside
the measured window. The script builds it into AITER's JIT directory for
`PYTORCH_ROCM_ARCH` (gfx942 and gfx950, like the other prebuilt variants); it
needs no GPU and adds about a minute to the build. Drop it once the base
prebuilds that variant itself.

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
drop it. `main` still has the unpadded eager MoE path; read the new source
before dropping the MoE script. Drop
the LMCache script once the base's LMCache contains LMCache#5098 (`bdb66b8`);
its marker comment then matches and the script reports it as applied.
