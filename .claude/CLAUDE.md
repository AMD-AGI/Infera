# Task — `e2e_deploy_standardized`: the first stage of the LLM e2e optimisation task package

Build the **"e2e 运行"** module of
`agent_sys/examples/llm_e2e_performance_optimization/` and drive it to a stable
pass. One closure — `e2e_deploy_standardized` — with its **task / ai agent /
handoff / validator** fully specified, run by `agent_sys` on real hardware for
two models.

Acceptance for "integrated": after debug and fixes are done, **three
consecutive runs must pass with no intervention and no edits**. Environment
preparation and peripheral repair before that first clean run do not count
against it.

Scope agreed with the user for this round (mission steps 2–5):

1. **Step 2** — author the first-cut spec of `e2e_deploy_standardized`.
2. **Step 3** — drive it with **Qwen3.6-27B** on gfx950 until stable; sanitise
   the resulting deliverable and add it as a **few-shot exemplar + validator
   assertion**.
3. **Step 4** — reproduce **GLM-5.3-Flash** on the cluster from the given packup.
4. **Step 5** — drive `e2e_deploy_standardized` + **GLM-5.3-Flash** on gfx950;
   same few-shot/assert treatment.

Steps 6 (mxfp4) and 7 (remote) are **out of scope this round** unless the user
re-opens them. MXFP4 GLM-5.3-Flash weights do not exist on this cluster.

## Background

`agent_sys` is a decoupled multi-agent task-flow system living inside the
`infera` repo. `examples/single_real_task` already proved the shape on gfx942:
an AI agent brought Qwen3.6-27B up in mix mode and handed back a packup-shaped
reproduction kit, checked by one program validator and one AI validator. This
round **generalises that one-off into a standardised, model-agnostic e2e deploy
step** and proves it on a second model and a second architecture.

Mission书: `agent_sys/examples/llm_e2e_performance_optimization/temp/mission.md`
(the series book) and the user's current `mission.md` at the repo root.

## Context — this environment, measured 2026-09-01

| | |
|---|---|
| where I am | login node `crs-m2m-cpu-spur-012`, **no GPU, no docker daemon access** |
| GPU nodes | slurm holds `94842`→`crsuse2-m2m-020`, `94843`→`crsuse2-m2m-188`, 8×MI355X **gfx950**, 288 GiB/GPU, idle |
| reaching them | `spur exec <jobid> bash -c '...'` — runs as **`yihou`**, docker talks to the **host** daemon |
| **walltime** | both jobs are 4 h, started 07:08 UTC → **hard stop ≈ 11:07 UTC**. Resubmit 8 h sleeper holds when they die |
| docker disk | on the 28 T NVMe, **~26 T free** — pulls and builds are fine (the 87 G on `/` is the exec namespace's own rootfs, a red herring) |
| local scratch | `/mnt/m2m_nobackup` (28 T, per-node, 10 GB/s) — **this is where a container workdir goes** |
| shared FS | `/shared_nfs` 360 T but **98 % full (7.6 T free)**, ~700 MB/s read, **no root_squash** (container root can write) |
| agent_sys roots | one knob: `--demo-root /shared_nfs/yihou/agent_sys_debug/runroot` relocates runs/handoffs/playground/workspace together |
| permissions | already default **off** (`AGENT_SYS_NO_PERMISSIONS` unset reads as `1`) |
| ports | 8080 taken; 8000–8079, 8081–8200, 30000 free. `--network host` works |
| GPU into containers | `--device /dev/kfd --device /dev/dri` + **numeric** `--group-add 44 --group-add 992` (named `render` fails) |

Weights: `GLM-5.3-Flash` (FP8, 62 shards) and `-BF16` complete under
`/shared_nfs/models/`; that directory is **not writable by me**, so Qwen3.6-27B
was staged to **`/shared_nfs/yihou/models/Qwen3.6-27B`** (copied from
`/shared_nfs/jiahazha/models/`, 15 shards / 51.75 GiB).

Images: no infera image exists on these nodes and `rocm/infera:sglang-dev` is
**not on docker.io**. `lmsysorg/sglang:v0.5.17-rocm720-mi35x` (gfx950) carries
`qwen3_5.py` but **not** `glm5_next`; infera is pure Python and installs into it
in seconds.

## Key references

- `agent_sys/examples/single_real_task/` — the template: `main.yaml`,
  `steps/serve.yaml`, `assets/serve_qwen.task/readme.md` (the agent brief worth
  copying wholesale), `assets/check_packup_shape.validator/`, `assets/lib/zone.py`.
- `/shared_nfs/yihou/packups/packups/`:
  - `qwen36-27b.mix.gfx942-agent_sys.packup_20260901/` — the agent_sys-driven
    reference run, its acceptance file `spec/B4-acceptance.md`, and
    `notes-src/FINDINGS.md` (read this before re-deriving anything).
  - `qwen36-27b.mix.gfx950-manual.packup_20260901/` — the gfx950 recipe.
  - `glm53flash.mix.packup_20260830/` — the GLM-5.3-Flash recipe, incl. the
    `Dockerfile.sglang.glm53` this repo does not carry.
- `/shared_nfs/yihou/agent_sys_debug/recon/RECON_20260901.md` — the measured
  environment recon this table summarises.
- Packup skill: `agent_sys/examples/llm_e2e_performance_optimization/temp/claude_code_skill_used_by_human/experiment-result-packup/`.

## Core principles

1. **Read the artefact, not the exit code.** A previous stage reported 14/14
   tasks and ten validators PASS over a run in which every result was zero.
   Every acceptance claim names a file to open and a condition that fails.
2. **Write acceptance criteria before the run, not after.**
3. **Pin the run id at launch.** `ls -1td runs/ | head -1` is not "my run" —
   this box is shared.
4. **Every identifier bound on a shared host is a parameter**: container names,
   ports, workdir, served model name. `: "${VAR:=…}"`, never `export VAR=`, and
   never `docker rm -f` a name you did not create.
5. **Research → gather → analyse → plan → work.** All temp activity stays in
   `/shared_nfs/yihou/agent_sys_debug/ws/`; the repo only receives the
   deliverable under `agent_sys/examples/llm_e2e_performance_optimization/`.
6. **Do not change host state** beyond docker images/containers we own and our
   own directories. Ask before anything wider.
7. Bugs in `agent_sys` are recorded under
   `agent_sys/examples/llm_e2e_performance_optimization/temp/bugs/` first, then
   worked around; fixed only when the evidence is unambiguous.
8. Work in English; report to the user in Chinese.

## Other notable details

### DCO sign-off is required on every commit (unchanged repository convention)

CI blocks any PR containing a commit without a `Signed-off-by:` trailer.

```bash
git commit -s -m "..."
git config user.name && git config user.email   # check before the first commit
```

Sign off **as yourself** — never a colleague's line, never a bot identity.
Cherry-picks do not inherit the trailer (`git cherry-pick -s`). To repair a
range without duplicating trailers:

```bash
git log --format='%h %s | %(trailers:key=Signed-off-by,valueonly)' origin/main..HEAD
git rebase --signoff <last-already-signed-commit>
git push --force-with-lease origin <branch>
```

`Co-Authored-By` is separate, is not a substitute, and must not be added when
contributing to third-party upstreams.

### Branch / PR

Branch `dev.yihou.aiopt.task_package`, PR 142. Activity is limited to
`agent_sys/examples/llm_e2e_performance_optimization/`.
