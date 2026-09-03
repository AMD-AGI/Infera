# `llm_e2e_performance_optimization` — stage 4, kernel optimization

The fourth of the six stages in the end-to-end LLM performance-optimization
flow, as a runnable `agent_sys` task package. It takes a **workset** describing
one operator, hands it to **KernelForge**, and hands back a reproduction kit
carrying the optimized kernel and the evidence for it.

Stages 1, 2, 3 and 5 are not here. What is here is the stage 4 contract — the
four objects the mission asks to be agreed — and a working implementation of it.

## The four objects

| | name | what it is |
|---|---|---|
| **task** | `optimize_kernel` | one leaf: workset in, `kernel_optimization` out. Plus `publish_workset`, a program leaf that puts the workset into the graph |
| **agent** | `kernel_opt_lead` | `kind: ai`, Claude Code SDK, **Opus**. Its readme is a KernelForge operating manual: how to invoke the loop, the five traps that fail quietly, and how to convert forge's own output tree into a handoff |
| **handoff** | `workset`, `kernel_optimization` | both `content_type: code`, both packup-shaped, both self-contained |
| **validator** | 3, see below | one on the workset, two on the output — cheap shape first, expensive re-measurement second |

```
main                        non-leaf: readme, no entry.sh, NO agent
│                           inputs [] · outputs [kernel_optimization]
│
├── publish_workset         program: workset_publisher
│                           out: workset                    [code]
│                             check_workset_shape         program · seconds · strong
│
└── optimize_kernel         is_end · ai: kernel_opt_lead (opus)
                            in:  workset
                            out: kernel_optimization        [code]
                              check_optimization_shape    program · seconds · strong
                              check_speedup_substantiated re-measure · minutes · weak
```

## Running it

Four variables have no default, because each is a fact about a site rather than
about this package:

```bash
docker exec -u "$(id -u):$(id -g)" <container> bash -lc '
agent-sys run \
  --package agent_sys/examples/llm_e2e_performance_optimization/kernel-opt-demo \
  --demo-root <a fresh run root> \
  --var kernelforge_repo=$HOME/dev/git.16-19/KernelForge \
  --var gpu=0 \
  --var scratch_root=/tmp/yihou/<yours> \
  --var packup_skill=$HOME/.claude/skills/experiment-result-packup'
```

**`scratch_root` must be on local disk, and that is now the only such
requirement.** It used to be true of `--demo-root` as well, for a reason nobody
had written down: `env_mgr` points `TMPDIR` at `<zone>/tmp` and the zone lives
under the run root, and on a cluster whose shared filesystem is NFS a ROCm
kernel launch with `TMPDIR` there dies with **SIGSEGV** — first launch, no
message, exit 139, after every device query has returned `hipSuccess`. Both
places that could inherit it now name a directory under `scratch_root` instead:
`kernel_opt_lead`'s `env` sets `TMPDIR`, and `check_speedup_substantiated` takes
a `scratch_dir` argument, because a validation zone's `TMPDIR` is an invariant
an agent spec cannot reach. So the run root may live wherever the site keeps
workspaces.

**Run as the host user, not container root.** That one flag removes four
separate failures at once; `temp/ws_kernel_opt_20260901/scratch/run_pkg.sh`
carries the full argument and refuses to start as root.

`--var gpu_target=` / `--var gpu_type=` name the architecture forge compiles
for. **They default to empty, which means "detect it from the card you are
on"**, and that is deliberate: the task readme used to hard-code `gfx942` /
`mi300x`, which is a fact about the machine this package was written on. The old
fixed behaviour is one flag away if something depended on it.

Useful knobs: `--var mock=1` (wiring only, no campaign), `--var max_hours=`
(forge's budget; **> 2.0** or forge silently degrades), `--var lead_model=`,
`--var forge_model=`, `--var workset=`, `--var workset_dir=`.

## The two models, and why they are set by different mechanisms

The outer wrapper runs on **Opus** and the nested KernelForge loop on
**`Claude-Sonnet-5[1m]`**. The outer one is set with
`backends[].config.options.model`, which is the *session's* model and travels
nowhere; the inner one reaches forge as `--model` on its command line.

`env: {ANTHROPIC_MODEL: ...}` would also work for the outer and is deliberately
**not** used: it is a process variable, so it would be inherited by the nested
`claude` that KernelForge starts, and the two models would collapse into one.

## What the validators actually check

- **`check_workset_shape`** — the workset carries the files an optimizer needs,
  its driver implements the stdout measurement contract, and it declares ≥3
  correctness cases. Runs on *both* sides of the handoff boundary.
- **`check_optimization_shape`** — one packup, four documents with substance,
  the four evidence files, the measurement apparatus, and **mock/smoke
  consistency**: a run that did not optimize may not read like one that did.
- **`check_speedup_substantiated`** — re-runs the 5-round protocol against the
  seed and the optimized kernel and compares the median it gets to the number
  the handoff claims. One-sided: under-claiming passes.

## Where the workset comes from

Stage 3's job is to produce worksets. It does not exist yet, so one ships as
package data at `assets/worksets/sampler_vocab_softmax/` and `publish_workset`
puts it into the graph. It is a **real** operator, not a fixture: the sglang
sampler's vocabulary softmax, `[8, 151936]` fp32, traced from a live Qwen3-0.6B
decode step at 55.59 µs/call and 14.5% of decode GPU time. Its baseline
re-measures at 55.40 µs — 0.3% from the profile, which is the cross-check that
makes every later speedup number mean something.

When stage 3 lands: `--var workset_dir=<what it produced>`, or replace the entry
in `main.yaml`'s subgraph. Nothing in `optimize_kernel` changes.

## Few-shot

`assets/few_shot/` — the "assert" directory in the mission's sense; the loader
supplies its name as `${TASK_PACKAGE_ASSERT_DIR}`, whose value is the literal
string `assets`. `01_pass/` is a complete handoff built from a **real** campaign
(2.83× reported, 2.61× re-measured) and it is used two ways: as worked material
for the agent, and as the fixture the validators are tested against.

## Known gaps, stated rather than left to be discovered

- **No `resources` block.** Both leaves want a GPU; `cli/build.py` is the only
  reader and the CLI declares no pools, so a block here would name a pool that
  does not exist. `--var gpu=` is the whole mechanism and **nothing reserves the
  card**. Two concurrent runs pointed at one GPU will collide and will blame
  each other's noise.
- **`repos:` is declared and read by nothing.** It is where the schema says the
  fact goes.
- **Permission enforcement is off** (the default), so nothing is confined and no
  grant is enforced. The run says so itself, in capitals.
- **No end-to-end claim.** This package measures a kernel. A kernel that is
  14.5% of decode GPU time bounds its own service-level benefit well below its
  own speedup, and nobody has measured that. It is stage 5's.
- **`agent_sys` does not import on Python 3.10**, which is the only Python in
  the image used here. See `bugs/001-typeddict-py310.md`.
