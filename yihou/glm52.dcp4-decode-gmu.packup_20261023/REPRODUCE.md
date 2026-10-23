# REPRODUCE

Prerequisite: the base kit, image and overlay of `../glm52.llying-dcp-s1s3.packup_20261022/` (see its REPRODUCE.md
§0-§4: image tar + ID `d3d6a9c20ce0`, overlay `sglang-ab7a898b77`, kit at
`/shared_nfs/yihou_glm52/llying_dcp_20261022/kit/`, job-bound container guard). All paths below are crusoe NFS.

## 1. Build the patched overlay (once, on a compute node)
```bash
# memlog (01) -> memlog-ec (02, TP0-only; kept because 03 was built on top of it) -> memlog-ec2 (03, all ranks)
spur exec <JOB> bash scripts/gmu/mk_memlog_overlay.yihou.sh   # -> src/sglang-ab7a898b77-memlog.yihou
spur exec <JOB> bash scripts/gmu/mk_ec_overlay.yihou.sh       # -> src/sglang-ab7a898b77-memlog-ec.yihou
spur exec <JOB> bash scripts/gmu/mk_ec2_overlay.yihou.sh      # copies; its first patch step asserted on 2 matches,
spur exec <JOB> bash scripts/gmu/fix_ec2.yihou.sh             # so fix_ec2 applies 03 to both event loops
```
Equivalent from git: `git checkout ab7a898b77 && git apply patches/01-*.diff patches/02-*.diff patches/03-*.diff`
(01 and 03 checked with `git apply --check`), then `git archive` `python/sglang` into the overlay path.
The overlay path is set per stage in `SGLANG_OVERLAY` (`scripts/stages/S2n.env`, `S2m.env`).

## 2. Run one load config (one 30-min AgentX C80)
```bash
# copy scripts/stages/S2<x>.env to /shared_nfs/yihou_glm52/llying_dcp_20261022/kit/stages/ and
# scripts/gmu/launch_stage2.yihou.sh to .../llying_dcp_20261022/work/, then on the node:
spur exec <JOB> bash -c 'cd /tmp && bash /shared_nfs/yihou_glm52/llying_dcp_20261022/work/launch_stage2.yihou.sh <2n|2m|...> <NK> r1'
spur exec <JOB> bash -c 'cd /tmp && bash .../gmu/tools/start_sampler.v2.yihou.sh .../gmu/vram.<S>-<NK>.v2.txt 7200'
```
- `launch_stage2` puts `/opt/rocm/bin` on PATH (rocm-smi left the default PATH on these nodes ~18:29Z 10-22; the
  kit's GPU-idle preflight otherwise waits forever).
- Stage files never set `DOWN=1`: stop the stack yourself after the run (`docker stop` the `yihou-dcp-pd-<NK>-s2<x>-*`
  containers; container names are per stage because stopped containers keep their names).
- Burst v3 nodes: use the stage variants with `HSA_ENABLE_SDMA=0` on both legs (S2f, S2o).

## 3. Analyse
```bash
spur exec <JOB> bash -c 'cd /tmp && bash /shared_nfs/yihou_glm52/llying_dcp_20261022/kit/analyze_ll.yihou.sh <RUN_DIR> yihou-dcp-pd-<NK>-s2<x>'
grep "TP0.*mem: dev_free" <RUN_DIR>/logs-live/decode.log     # allocator trace (memlog builds only)
```
Per chip = total tok/s / 8 (prefill 4 + decode 4 GPUs; the kit's agentx.sh has PREFILL_TP=4).

## 4. Decode-only start-up probe (no prefill)
`NK=<nk> SLOT=a|b TAG=<tag> MEM=0.92 EXTRA="<sglang args>" ENVX="<K=V>" bash scripts/gmu/decode_only.v2.yihou.sh`
or a series file via `scripts/gmu/node_go.v2.yihou.sh <JOB> <NK> <series.txt> [GLOBAL_ENVX]`.
