# 03 · optimise one kernel with KernelForge

Level 3 of the [demo ladder](../README.md). One kernel — the vocabulary softmax
in sglang's sampler, which Qwen3.6-27B runs over 248,320 logits per token — goes
through [KernelForge](https://github.com/AMD-AGI/Hyperloom), and the result is
accepted only if the workset's own tests say it is correct and a validator,
re-measuring it, agrees it is faster.

```
main
├─ publish_inputs            program   -> deploy_kit, operator_workset   (from assets/fixture/)
└─ m4_kernel_opt             (e2e-flow stage 4)
   └─ optimize_kernel        AI agent  -> kernel_optimization
```

- **The inputs are fixed, and real.** `assets/fixture/workset/` is a
  one-operator **workset** — the operator's definition, its shapes
  (`B16/B32/B64 × 248320`), a reference implementation, the baseline, correctness
  and performance harnesses, and the baseline **measured on an MI300X**.
  `assets/fixture/deploy_kit_qwen36/` is a deploy kit produced by level 2a. In the full
  flow stages 1–3 produce these; here a program task publishes them, and they are
  validated on publication exactly as if those stages had.
- **The incumbent is deliberately naive.** Stock sglang computes this softmax
  with one `torch.softmax`, which a one-hour campaign could not beat (measured:
  4 iterations, 0.99x, nothing kept). So the workset's baseline is a row-at-a-time
  softmax that finds each row's max by sorting it, and m4 installs it at the call
  site (`integration.call_site`) before forge starts. The demo shows the
  mechanism working; the size of the speed-up says how slow the incumbent was.
- **The agent runs the workset's own one-liner**, `operators/<id>/run_forge.sh`,
  which starts a `python -m kernelforge.cli forge-loop` campaign on the operator, then the
  workset's correctness and performance tests on what it produced.
- **`check_speedup_substantiated` re-measures** the candidate and the baseline
  itself and refuses a speed-up it cannot reproduce.

How the fixture was built and measured: `assets/fixture/build_fixture.py`.

## Prerequisites

Everything level 2a needs, plus:

Use Hyperloom's `AKLR` branch at `fa94e001b88d77e69d8d50973c868a8f4a11a15d`
and AKLR at `5d59c935e7e47410e1f84bc4c2a5077816d4071d`. Hyperloom supplies
the maintained Forge implementation; AKLR supplies the externally referenced
[kernel knowledge base](https://github.com/AMD-AGI/AKLR/blob/5d59c935e7e47410e1f84bc4c2a5077816d4071d/harness/skills/aklr-kb/SKILL.md).
Pass both clean checkouts with `hyperloom_repo` and `aklr_repo`; the legacy
`kernelforge_repo` variable accepts only the same verified Hyperloom checkout.
Both paths must be visible inside the serving container at the supplied locations.
The runtime installs `Hyperloom[forge]`, checks its module provenance and commands,
and supplies the KB with `--skill-dir`. No KB content is copied into this package.
The campaign workflow is shared with `agent-sys-kernel-forge` under `tools/skill/`.

- **A pinned Hyperloom AKLR checkout** (`git clone --branch AKLR https://github.com/AMD-AGI/Hyperloom`).
  The task installs it from that verified checkout inside the container on each
  real run (`--var hyperloom_repo=`), even when another Forge is on PATH.
- **Model credentials for KernelForge's own agent.** They are taken from your
  `claude` settings, the same as the outer agent, and handed to the container
  through a 0600 env file that is removed afterwards.
- **One free GPU**, and time: a campaign has a one-hour floor (`forge_max_hours`).
- Local disk for scratch and work: the container mounts `/shared_nfs`,
  `/home/<user>`, `/mnt/m2m_nobackup/<user>` or `/var/tmp/<user>`.

## Run it on one host

```bash
PYTHONPATH=$PWD/agent_sys python3 -m cli.main run \
  --package agent_sys/examples/03_kernelforge_one_kernel \
  --demo-root /var/tmp/$USER/l3 \
  --var model_name=Qwen3.6-27B --var model_path=/data/models/Qwen3.6-27B \
  --var image=infera:sglang-gfx942-glm52 --var tp=1 \
  --var container=${USER}_l3 --var gpu=3 --var measure_gpu=3 --var gpu_devices=3 \
  --var port_router=18301 --var port_worker=18302 --var port_etcd=18303 \
  --var validate_port_base=18340 \
  --var work_root=/var/tmp/$USER/l3/work --var scratch_root=/var/tmp/$USER/l3/scratch \
  --var transport=local --var node=$(hostname) --var node_ip=127.0.0.1 --var jobid=local \
  --var transport_env=E2E_REMOTE_HOME=/var/tmp/$USER \
  --var mock_stages=none \
  --var hyperloom_repo=/path/to/Hyperloom --var aklr_repo=/path/to/AKLR \
  --var forge_max_hours=1
```

| var | why |
|---|---|
| `gpu` | the one card the campaign and the tests use |
| `measure_gpu` | the card the validators re-measure on |
| `transport_env` | lets the validators' measuring container mount your local disk |
| `forge_max_hours` | the campaign budget (minimum 1; above 2 enables forge's profiling-driven analysis) |
| `forge_mock=1` | skip the campaign: run the tests on the baseline only — the cheap way to check the setup |

## What a run looked like

The measurements below predate this migration and describe the original Forge
revision used by that run. They have not been remeasured with Hyperloom.

Measured on one MI300X (`gpu=4`, `forge_max_hours=1`), 59 minutes end to end.
The inputs validated (the deploy kit served again under load; the workset's
baseline re-measured within 25 %). forge passed its own preflight and ran two
iterations, both kept: first `torch.softmax` in place of the naive incumbent,
then a column-split Triton softmax (grid `batch × ⌈V/4096⌉`, so even 16 rows
fill the GPU). The workset's tests then measured the result against the
workset's own baseline, and `check_speedup_substantiated` re-measured both:

| shape | naive baseline | optimised | re-measured speed-up |
|---|---|---|---|
| B16 × 248320 | 2.637 ms | 0.084 ms | 30.9x |
| B32 × 248320 | 5.244 ms | 0.084 ms | 62.1x |
| B64 × 248320 | 10.494 ms | 0.082 ms | 126.0x |

All three validators on `kernel_optimization` passed (mean 73.0x re-measured
against 73.8x claimed). Most of that is the distance from the naive incumbent to
`torch.softmax`; by forge's own timing the Triton kernel is about 2x faster than
`torch.softmax` itself. The delivered `apply/patches/sampler_vocab_softmax.patch`
is one diff of `srt/layers/sampler.py`: the call site and the new function.

## What you get

```bash
R=$(ls -d /var/tmp/$USER/l3/runs/* | tail -1)
cat $R/handoffs/handoff.kernel_optimization.*/v*/content/README.md    # what forge produced, and the measured speed-up
cat $R/handoffs/handoff.kernel_optimization.*/v*/validation.yaml
```

## Next

The full flow, [`llm_e2e_performance_optimization`](../llm_e2e_performance_optimization/),
finds the kernel itself (stages 2–3) and integrates the optimised one back into
the engine (stage 5).
