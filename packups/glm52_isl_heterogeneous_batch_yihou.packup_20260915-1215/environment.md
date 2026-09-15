# Environment

Everything below was read from the running container and host on 2026-09-15, not copied forward
from a previous packup.

## Host

| | |
|---|---|
| node | `smci355-ccs-aus-n10-29.prov.aus.ccs.cpe.ice.amd.com` |
| GPU | 8 x AMD Instinct MI355X (`0x75a3`) |
| ROCm driver | 6.14.14 |
| kernel | 6.8.0-107-generic |
| scheduler | Slurm, partition `Compute-DCPT`. Our hold was job `30723` (`yihou-hold`). |

**The node is shared and containers on it do not respect Slurm.** Holding the allocation does not
make the GPUs idle; check `rocm-smi --showpids` before believing a number. See `notes.md`.

## Container and image

| | |
|---|---|
| container | `yihou-glm52-tp8ep1-pr50-51` |
| image digest | `sha256:bdd783512f3db4d046fcf6e76e7039da054fe2634ad1d831a16a97da65dad656` |
| image tag | `rocm-llm-bench:yihou-pr50-51-20260915` |
| base image | `lmsysorg/sglang:v0.5.18-rocm720-mi35x` |
| python | 3.10.12 |
| torch | 2.9.1+rocm7.2.0.git7e1940d4 |

Image provenance, read from `/image_provenance_yihou.txt` inside the container:

```
sglang_base=56fae0a51efd8359f0ea9fe051104eb515e151cb
sglang_patch=integration_patch_yihou.diff
aiter_base=2c71811b32c8ce2e1266aedaec199df7d90f597d
sglang_head=51f33fafb integrate sglang PR #51 + #50 (round 1; #54 deliberately excluded)
aiter_head=unpatched 2c71811b32c8ce2e1266aedaec199df7d90f597d
```

So: sglang fork branch `dev_glm52_0907` at `56fae0a51` **plus PRs #51 and #50**; aiter at the pin,
unpatched. That image and its build recipe come from the previous task and are packed in
`packups/glm52_tp8_ep1_c256_pr51_50_54_yihou.packup_20260915-1019/` — this packup does not rebuild
it.

**The ISL feature is independent of that image.** It is pure benchmark-harness code, bind-mounted
from the repo, so any container with a compatible sglang will do. The image matters here only
because the numbers in `results/` were taken on it.

## Model

| | |
|---|---|
| path | `/perf_apps/data/models/GLM-5.2-MXFP4` (absolute, on the shared `perf_apps` mount) |
| `max_position_embeddings` | 1048576 — this is why ISL 300000 is legal |
| `index_topk` | 2048 |
| `num_attention_heads` | 64 |

`num_attention_heads = 64` is the reason the FlyDSL decode kernel declines and TileLang runs: with
DP attention, `attn_tp_size = 8 // 8 // 1 = 1`, so every rank carries all 64 q heads against a gate
of 8 or 16. See `notes.md`.

## Python used for the CPU-only tests

The unit suite needs no container, no venv and no GPU:

```
python3 -m pytest tests -q      # host python 3.13.13
```

`bench/isl_spec.py` is stdlib-only by contract and there is a test that enforces it by importing
the module in a subprocess and asserting `torch` / `numpy` / `sglang` are absent from `sys.modules`.

## Secrets

None. No registry login, no API key, no S3/etcd credential. Access needs only SSH to
`smci355-ccs-aus-n10-29` (configured host alias, `BatchMode=yes` works) and membership of the
`docker`, `video` and `render` groups on it.
