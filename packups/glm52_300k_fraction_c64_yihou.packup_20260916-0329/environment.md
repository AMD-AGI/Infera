# Environment

Re-read live from the host and the running container at packup time, not copied forward from the
previous packup.

## Host

| | |
|---|---|
| node | `smci355-ccs-aus-n10-29` |
| GPU | 8 x AMD Instinct MI355X (`0x75a3`) |
| ROCm driver | 6.14.14 |
| kernel | 6.8.0-107-generic |
| scheduler | Slurm, partition `Compute-DCPT`; our hold was job `30723` (`yihou-hold`) |

**The node is shared and its containers do not respect Slurm.** During this session the GPUs were
at one point fully occupied by three other workloads while Slurm showed all 8 GPUs allocated to us.
Check `rocm-smi --showpids` before trusting any measurement — see `notes.md` §3.

## Container and image

| | |
|---|---|
| container | `yihou-glm52-tp8ep1-pr50-51` |
| image | `sha256:bdd783512f3db4d046fcf6e76e7039da054fe2634ad1d831a16a97da65dad656` |
| tag | `rocm-llm-bench:yihou-pr50-51-20260915` |
| base | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` |
| python | 3.10.12 |
| torch | 2.9.1+rocm7.2.0.git7e1940d4 |

Provenance, read from `/image_provenance_yihou.txt` inside the container at packup time:

```
sglang_base=56fae0a51efd8359f0ea9fe051104eb515e151cb
sglang_patch=integration_patch_yihou.diff
aiter_base=2c71811b32c8ce2e1266aedaec199df7d90f597d
sglang_head=51f33fafb integrate sglang PR #51 + #50 (round 1; #54 deliberately excluded)
aiter_head=unpatched 2c71811b32c8ce2e1266aedaec199df7d90f597d
```

sglang fork branch `dev_glm52_0907` at `56fae0a51` **plus PRs #51 and #50**; aiter at the pin,
unpatched. The image build recipe is not here — it is in
`packups/glm52_tp8_ep1_c256_pr51_50_54_yihou.packup_20260915-1019/`.

## Repository code

The `--input-len-spec` feature this experiment exercises is committed on
`dev.yihou.sglang.bench.fast.script`:

```
bce29162 bench: support heterogeneous input lengths in the decode harness
41020f5f packups: heterogeneous-ISL batch support, with its verification runs
```

The harness is bind-mounted from the repo into the container, so the code that ran is the working
tree at those commits — **not** anything baked into the image. Each point's `git_head.txt` and
`code_hashes.sha256` in `results/points/` record what was actually present at run time.

## Model

| | |
|---|---|
| path | `/perf_apps/data/models/GLM-5.2-MXFP4` (absolute, shared `perf_apps` mount) |
| `max_position_embeddings` | 1048576 — this is why ISL 300000 is legal |
| `index_topk` | 2048 |
| `num_attention_heads` | 64 |

`num_attention_heads = 64` is why FlyDSL declines and TileLang runs: with DP attention
`attn_tp_size = 8 // 8 // 1 = 1`, so every rank carries all 64 q heads against a gate of 8 or 16.

## Benchmark configuration, common to all six points

```
--tp-size 8 --ep-size 1 --enable-dp-attention
--batch-size 64 --max-running-requests 64
--output-len 10000 --accept-length 3.61 --warmup-steps 10
--enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope --mem-fraction-static 0.85
```

Derived: `dp_size = 8`, `local_batch_size = 8`, EAGLE steps=5 / draft-tokens=6 / topk=1,
KV dtype `fp8_e4m3`, `moe_a2a_backend = none`, CUDA graph **on**.

Peak allocated memory was 236.0 GB at every point — set by `--mem-fraction-static 0.85`, not by the
workload.

## Timing

Series ran 2026-09-15 ~12:36–13:20 UTC in one contiguous window, six runs with 60 s gaps.
The earlier `c64_all70k_control_yihou` and `c64_one300k_yihou` runs referenced in the analysis are
from 12:00 and 12:11 UTC, outside that window.

## Secrets

None. Access requires SSH to the node (configured alias, `BatchMode=yes` works) and membership of
the `docker`, `video` and `render` groups. No registry login, no API key.
