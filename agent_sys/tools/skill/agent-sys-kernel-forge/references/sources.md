# Workflow sources and relationships

This skill adapts the campaign procedures below. AKLR's kernel knowledge stays
external; no KB pages are included in this skill.

| Source at the pinned AKLR revision | Responsibility here |
|---|---|
| [Forge task walkthrough](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/docs/forge_task.md) and [adapter README](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/scripts/forge_task/README.md) | Native schema preparation, the authoritative AKLR driver and local publication |
| [forge-loop-campaign](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/forge-loop-campaign/SKILL.md) | Campaign decisions, supervision, steering, recovery and post-run review |
| [report-progress](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/report-progress/SKILL.md) | Evidence-based progress and explicit liveness/timer state; adapted to the invoking user's interval/tools |
| [aklr-read-validation](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-read-validation/SKILL.md) | Interpret native AKLR reports per Point |
| [aklr-port-generated-kernels](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-port-generated-kernels/SKILL.md), [aklr-review-kernel-pr](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-review-kernel-pr/SKILL.md) and [aklr-bad-cases](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-bad-cases/SKILL.md) | Provenance, public-path integration, unchanged contracts and independent/fair validation |

`forge_task` is an AKLR adapter, not the campaign engine. The campaign engine is
Hyperloom's embedded Forge. The operator knowledge is a separate reference.
AKLR's resource-allocation, image-build and downstream-release skills are not
mandatory dependencies of an Infera local-GPU campaign.

## Pinned implementation

- Hyperloom repository: <https://github.com/AMD-AGI/Hyperloom>.
- Required branch baseline: `AKLR`, commit
  `fa94e001b88d77e69d8d50973c868a8f4a11a15d`.
- AKLR repository and KB/adapter baseline: <https://github.com/AMD-AGI/AKLR>,
  commit `5d59c935e7e47410e1f84bc4c2a5077816d4071d`.

The inspected Hyperloom main revision
`d265d9b85d09cfe25b0996712367559aac206651` does not register the AKLR branch's
`forge-say`, `forge-ablate`, `forge-finalize` and `forge-distill` commands.
See the [required CLI registration](https://github.com/AMD-AGI/Hyperloom/blob/fa94e001b88d77e69d8d50973c868a8f4a11a15d/src/kernelforge/cli/__init__.py).
Do not replace this baseline with floating main while retaining those steps.
A future update requires checking the command, backend, driver, result and
skill-delivery contracts together.

Install from a checkout of that revision with `pip install -e "<checkout>[forge]"`.
The canonical entry is `python -m kernelforge.cli`; backend values are plain
registry names such as `triton`, `gluon`, `flydsl`, `hip` and `ck`, without a
`-fellow` suffix. The source/install/capability helper is
`../scripts/check_forge.py` relative to this directory.

## External kernel knowledge

`.claude/skills/aklr` in AKLR is a symlink to `../../harness`; the canonical KB
entry is [`harness/skills/aklr-kb/SKILL.md`](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-kb/SKILL.md).
Start there and read only the pages it routes to for the present question.

| Context | External entry |
|---|---|
| FlyDSL | [flydsl/SKILL.md](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-kb/flydsl/SKILL.md) |
| Triton or Gluon | [triton/SKILL.md](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-kb/triton/SKILL.md) |
| HIP | [hip/SKILL.md](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-kb/hip/SKILL.md) |
| GEMM / MoE / attention | [gemm](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-kb/gemm/INDEX.md), [moe](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-kb/moe/INDEX.md), [attention](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-kb/attention/INDEX.md) |
| MI300X / MI355X | [gfx942](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-kb/hardware/gfx942/INDEX.md), [gfx950](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-kb/hardware/gfx950/INDEX.md) |

Keep these paths in the external checkout. Put revision/path/use evidence and
links in handoffs, not copied knowledge. AKLR-specific contract rules apply to
native AKLR kernels; a serving workset retains its own declared contract.

## Attribution for adapted workflow guidance

AKLR source license at the pinned revision:
[LICENSE](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/LICENSE).
The following notice covers the adapted workflow material, not an included KB.

Copyright © Advanced Micro Devices, Inc. All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
