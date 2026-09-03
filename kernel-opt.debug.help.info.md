# `kernel-opt-demo` — debug / localisation notes, run of 2026-09-02

Module: **kernel-opt-demo** (stage 4, `optimize_kernel`).
Node: job `101078`, `crsuse2-m2m-080`, **GPU 4 only**. Port band 8160–8179
(not used — this package binds no port).
Worktree HEAD at start: `3613481`.

Licence for this round (from the leader, quoting the user): *"kernel opt,
没有提升也行，有标准的 handoff 即可"* — no speedup required; a structurally
standard, validator-passing `kernel_optimization` handoff is the whole
deliverable. Every reduction below is taken under that licence and is named.

---

## 1. `mock=1` is the sanctioned cheap path, and here is exactly what it does

Read first-hand in `steps/kernel_optimization.yaml:125` and
`assets/optimize_kernel.task/readme.md:176-188`.

`--var mock=1` sets `KFO_MOCK=1` on the `kernel_opt_lead` agent's `env:` row.
It is **not** a framework feature; it is an instruction the agent's readme
tells it to obey:

- do **not** run the `kernel-agents forge-loop` campaign;
- **do** run the workset driver's correctness and bench modes for real (cheap,
  and they prove the wiring) — so **torch on a GPU is still required**;
- write every file `check_optimization_shape` demands;
- set `"mock": true` in `results/forge_result.json` and
  `results/verification.json`;
- put `MOCK RUN — no optimization was performed` in the first line of the
  packup `README.md`'s `## Result` section.

Both output validators accept this, by design and by construction:

- `check_optimization_shape/check.py:152-167` — if `mock` is true it *requires*
  the README to say `MOCK` and *refuses* any claimed speedup. So a mock is
  passable, but only if it is visibly a mock.
- `check_speedup_substantiated/check.py:213-217` — `if bool(claim.get("mock")):`
  it records "mock run: no speedup claimed, nothing to re-measure" and does not
  re-measure. This is also why mock mode does **not** need a torch-capable
  interpreter *in the validator zone* (bug `002`'s trap is side-stepped, not
  fixed).

**Conclusion:** mock produces a real, sealed, validator-passing
`kernel_optimization` handoff without a forge campaign. That is the path taken.
Everything in it except the driver's own correctness/bench numbers is synthetic
and is labelled as such.

## 2. The brief's "settle budget is 1800 s, hard-coded" is **stale**

Checked first-hand in this worktree:
`agent_sys/cli/main.py:903` — `_SETTLE_TIMEOUT = 14400.0` (4 hours), and the
comment above it records the value's history: 300 s → 1800 s → 14400 s, each
raise caused by a healthy run being reported as a hang. `--timeout` overrides it.

So `max_hours=3.0` (the package default forge budget) is **no longer**
irreconcilable with the framework ceiling the way the brief states. It is still
irreconcilable with this round's licence and with an 8 h shared hold, which is
the actual reason mock was chosen.

Anyone re-reading `temp/bugs/003` and `005` should re-check the constant before
designing around 1800 s.

## 3. Environment facts measured on this node (2026-09-02)

| fact | measured value |
|---|---|
| `spur exec 101078` identity | **`yihou`**, `uid=50112975 gid=1000(ubuntu)`, groups incl. `44(video) 992(render)` — **not root** on this cluster |
| node `python3` | 3.12.3, **no torch**, **no `agent-sys`** installed |
| `claude` CLI | `/home/yihou/.local/bin/claude` exists and `/home/yihou` is visible from the node |
| `/mnt/m2m_nobackup/yihou` | did **not** exist; created by me |
| torch-capable image present | `infera/engine-sglang:test-local` — torch `2.9.1+rocm7.2.0`, HIP 7.2.26015, **Python 3.10.12**, pydantic 2.13.4, `/opt/venv/bin/python3` present |
| absent | `infera/engine-sglang:gfx950-local`, `lmsysorg/sglang:v0.5.17-rocm720-mi35x` (only `v0.5.12` is here) |

The README's `docker exec -u "$(id -u):$(id -g)"` advice still applies, but the
reason shifts: under `spur exec` on **this** cluster you are already `yihou`, so
`$(id -u):$(id -g)` evaluates to `50112975:1000` and the flag stops the
container from running as *container* root over an NFS tree that `root_squash`
would deny. Do not assume you arrived as root here; the brief says root, the
node says otherwise.

## 4. Python 3.10 in the only torch image → `bugs/001` is live

The one image on this node that can `import torch` is Python **3.10.12**, and
`agent_sys` does not import on 3.10 (`bugs/001-typeddict-py310.md`:
`spec_loader/protocols.py` imports `TypedDict` from `typing`, pydantic ≥2 refuses
it below 3.12). The bug note's own workaround is used: a **local copy** of the
repo on node-local disk with the one-line import change, never the worktree.

## 5. The run refuses to start unless the repo sets `extensions.preciousObjects`

First launch attempt died before any task, with:

```
done  <repo> does not set extensions.preciousObjects, and `env_mgr.workspace.cut`
      refuses without it — so every output-producing task would die in `prepare`.
Run:  git -C <repo> config extensions.preciousObjects true
Or pass: --allow-repo-config, and the demo sets it for you.
Note: in a git worktree this lands in the SHARED common config, so it affects the
      main checkout and every other worktree, and `git gc` will refuse in all of them.
```

The note in the message is the important part and it is why I did **not** set
it: five agents share this worktree's common git dir in this round, and
`--allow-repo-config` would have flipped a setting for all of them.

**What was done instead:** run the package out of the standalone clone the
BRIEF names, `/shared_nfs/yihou/agent_sys_debug/repo`, which already has
`extensions.preciousObjects=true`. Verified before using it that its copy of
`kernel-opt-demo/` is byte-identical to the worktree's (`diff -r` → no output);
the clone is at `532da57` and the worktree at `3613481`, and the only two
commits between them touch `temp/mission.md` and the packup skill, not this
package.

**A later reader should check the diff again rather than trusting this
sentence.** If the clone has drifted, `git -C <clone> fetch` + checkout is the
fix — not `--allow-repo-config` on a shared worktree.

## 6. Install recipe that works on this node (no image build)

`agent-sys` is **not** installed on the node and the node has no torch, so
everything runs in a container built from an image already present:

```bash
docker run -d --name dbg_kernelopt_yihou \
  --network host --ipc host --shm-size 32g \
  --device /dev/kfd --device /dev/dri --group-add 44 --group-add 992 \
  --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -v /mnt/m2m_nobackup/yihou:/mnt/m2m_nobackup/yihou \
  -v /home/yihou:/home/yihou -v /shared_nfs:/shared_nfs \
  -e HOME=/home/yihou --entrypoint bash \
  infera/engine-sglang:test-local -lc "sleep infinity"
```

Then, **as `-u 50112975:1000`, never as container root**:

```bash
python3 -m venv --system-site-packages /mnt/m2m_nobackup/yihou/kernel-opt/venv
venv/bin/pip install -U pip setuptools wheel          # see the trap below
venv/bin/pip install "<local patched repo>/agent_sys[claude]"
```

Three traps, all measured today:

1. **`pip install -e` fails**: *"build backend is missing the `build_editable`
   hook"*. `agent_sys`'s backend does not implement PEP 660. Install
   non-editable. That is fine here because the console script is
   `agent-sys = cli.main:main` and a non-editable install puts `cli` on the
   path, so the BRIEF's *"cd into the repo first"* stops mattering.
2. **The image's venv ships pip 22.0.2 / setuptools 59.6.0**, which cannot read
   PEP 621 metadata: the install "succeeds" as `unknown 0.0.0`, silently
   installs no dependencies, writes no console script, and the only visible
   symptom is `WARNING: unknown 0.0.0 does not provide the extra 'claude'`
   followed by `agent-sys: No such file or directory`. **Upgrade pip and
   setuptools first.**
3. Use a `--system-site-packages` venv on **local disk**, not `pip install
   --user`: `--user` would write into the shared `/home/yihou/.local`, which
   four other agents on this node also see.

`claude` 2.1.252 from `/home/yihou/.local/bin` runs unmodified inside this
container, so no node/npm install is needed — just put it on `PATH` and export
`HOME=/home/yihou`.

## 7. The run

```
run id     20260902T083759-9f9b9e
demo root  /shared_nfs/yihou/agent_sys_debug/ws2/runroot/kernel-opt-mock1
driver     /mnt/m2m_nobackup/yihou/kernel-opt/run_pkg.sh   (TAG=mock1)
log        /mnt/m2m_nobackup/yihou/kernel-opt/run_mock1.log
vars       kernelforge_repo=/mnt/m2m_nobackup/yihou/kernel-opt/KernelForge
           gpu=4  scratch_root=/mnt/m2m_nobackup/yihou/kernel-opt/scratch
           packup_skill=/home/yihou/.claude/skills/experiment-result-packup
           mock=1
```

KernelForge: `/shared_nfs/hyperloom/KernelForge` verified complete
(`src/kernel_agents/cli.py` carries `MIN_MAX_HOURS = 1.0` at line 54) and copied
to node-local disk (62 MB) rather than run off NFS. In mock mode it is never
invoked; the variable is required and must point at something real.

## 8. THE localisation trap of this module: a zone on NFS **segfaults every ROCm kernel launch**

This is the finding worth the whole session. First run of `optimize_kernel`
went 25 minutes without producing a byte of handoff, because the agent was
bisecting — correctly and methodically — a fault it could not have named:
`torch.ones(4, device="cuda")` died, a bare Triton kernel died, and it had got
as far as writing a raw HIP program (`h.hip`) and disassembling object files.

**Cause, measured directly, three times:**

```
docker exec -u 50112975:1000 -e HIP_VISIBLE_DEVICES=4 [-e TMPDIR=X] <ctr> \
  /opt/venv/bin/python3 -c 'import torch;torch.ones(4,device="cuda");torch.cuda.synchronize();print("OK")'

TMPDIR unset (container default /tmp)                    -> OK          exit 0
TMPDIR=<zone dir under /shared_nfs/...>                  -> (no output) exit 139
TMPDIR=/shared_nfs/yihou                                 -> (no output) exit 139
TMPDIR=/shared_nfs/yihou/agent_sys_debug/ws2             -> "Segmentation fault (core dumped)"  exit 139
TMPDIR=/mnt/m2m_nobackup/yihou/kernel-opt/scratch/tmpprobe -> OK        exit 0
```

`139 = 128 + SIGSEGV`. It is not a hang and it is not a permission error; it is
a **hard crash in the ROCm runtime when `TMPDIR` is on this NFS mount**, and it
happens on the *first kernel launch*, after `hipGetDeviceCount` and device
property queries have all returned `hipSuccess`. That is why the ROCm debug log
(`AMD_LOG_LEVEL`) looks completely healthy right up to the point it dies.

**Why the agent was exposed to it:** `env_mgr` puts the task zone under
`--demo-root` and sets `TMPDIR` (and `CLAUDE_CODE_TMPDIR`) to `<zone>/tmp`. I had
put `--demo-root` on `/shared_nfs` because the shared BRIEF asks for a run root
both nodes can see. So every GPU process the agent started inherited an NFS
`TMPDIR` and segfaulted.

**The package already says this and I did not read it as a hard requirement.**
`kernel-opt-demo/README.md:44` — `--demo-root <a fresh run root on LOCAL disk>`.
Now there is a measured reason behind the word LOCAL, so it is recorded here.

**Fix applied:** `--demo-root /mnt/m2m_nobackup/yihou/kernel-opt/runroot`
(node-local). The deliverable is copied to `/shared_nfs` afterwards, which gives
the login node its visibility without putting the *zone* on NFS.

**For the other four modules:** any package whose task touches a GPU inherits
this. If your agent reports an inexplicable crash or silence from anything that
launches a kernel, check `TMPDIR` before anything else. The BRIEF's
"`--demo-root` under `ws2/runroot`" is safe only for packages that never launch
a GPU kernel *from inside a zone*.

**A cleaner fix exists and is not applied here** (out of this round's scope, and
untested against `env_mgr`'s ordering): give `kernel_opt_lead` an explicit
`TMPDIR: '${scratch_root}/tmp'` in its `env:` row in
`steps/kernel_optimization.yaml`, so the package is immune to where the run root
was put. Whether a spec `env:` row beats the zone's own `TMPDIR` needs checking
in `env_mgr` before anyone relies on it.

## 9. The node died — `crsuse2-m2m-080` went `down` at ~08:58 UTC

Five minutes into the relaunched run, `spur exec 101078` began answering

```
job 101078 is not running (state: CANCELLED)
```

and `sinfo -n crsuse2-m2m-080 -o "%n %t"` says `down`. A replacement hold
(`101146`) failed to launch with *"dispatch confirmation failed (0/1
confirmed): 1 agent unreachable"*, so this is the node itself, not a scheduler
decision. deploy-demo and analyze-demo were on the same node and are blocked
with me.

**Everything on `/mnt/m2m_nobackup/yihou/kernel-opt` went with it** — container,
venv, KernelForge copy, and the second run root, which I had just moved to local
disk to escape the NFS `TMPDIR` segfault. There is no version of this session in
which both hazards are avoided by placing the run root alone: **local disk is
required for correctness and is lost with the node.** The right shape, for
anyone doing this again, is a local run root plus an *incremental* copy of the
sealed handoff out to `/shared_nfs` the moment it appears, not at the end.

Recreating the environment on a fresh node is §6 plus the `run_pkg.sh` in this
note; budget ten minutes, not an hour.

## 10. Recovery on a spare node, and a better image than the one the README implies

`101155` landed on `crsuse2-m2m-019`. Two things differed there and both are
worth knowing:

**The image is not the same image.** `infera/engine-sglang:test-local` does not
exist on 019 — `docker: pull access denied` — which is deploy-demo's lesson
restated: **docker images on this cluster are per-node, with no shared
registry**. Check `docker images` on *your* node before planning around a tag.

**019 carries a strictly better image for this package:**

```
rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0
  /opt/venv/bin/python3 -> Python 3.12.3
  torch 2.10.0+rocm7.2.4   triton 3.6.0+rocm7.2.4
```

Python **3.12**, so `bugs/001` does not apply at all and no source patch is
needed; and `/opt/venv/bin/python3` is exactly the package's default for
`KFO_PYTHON`, so `--var python_bin=` can stay unset. If you get to choose an
image for this package, choose a `rocm/pytorch` py3.12 one over an sglang one:
this package needs torch and triton, not an inference server.

Bootstrap on the fresh node took **about two minutes**, end to end, including
the venv and a GPU launch probe, using
`/shared_nfs/yihou/agent_sys_debug/ws2/kernel-opt-bootstrap.sh <jobid> <gpu>`
(written during the outage, `IMAGE=` overridable). That script is the thing to
reuse; the ten-minute estimate in §9 was pessimistic.

Run 3: `20260902T090855-446cad`, node `crsuse2-m2m-019`, GPU 4, demo root
`/mnt/m2m_nobackup/yihou/kernel-opt/runroot/kernel-opt-mock3` (local disk).

## 11. Outcome — run `20260902T090855-446cad`, three tasks succeeded, three verdicts PASS

```
final  main: succeeded            final  optimize_kernel: succeeded
final  publish_workset: succeeded
verdict  check_optimization_shape:    PASS   completeness / strong
verdict  check_speedup_substantiated: PASS   trustworthiness / weak
verdict  check_workset_shape:         PASS   completeness / strong
done  run complete; ... 0 validation(s) dropped
```

Wall time of the whole graph: **~13.5 minutes** (09:08:55 launch →
09:22:16 last verdict), of which the `optimize_kernel` agent turn was ~13.

**Read the artefact, not the exit code — what I opened and what I checked:**

- `md5sum results/optimized_kernel.py scripts/kernel/sampler_softmax_kernel.py`
  → both `7027af15299d622cf4d8967c5a962e02`. The "no optimization was
  performed" claim is *true*, not merely asserted.
- `results/verification.json` parses; `mock: true`; `mean_case_speedup` is
  `1.0`, i.e. **not** above 1.0 — which is the exact condition
  `check_optimization_shape` refuses on a mock.
- the packup `README.md`'s `## Result` section opens with
  `**MOCK RUN — no optimization was performed.**` — the other mock condition.
- `content/README.md` carries `## Purpose`, `## Interface`, `## Boundary`.
- all seven paths in `required_evidence` exist.
- the same two verdicts are sealed inside the handoff at `validation.yaml`
  with `result: true`, so the report and the artefact agree.

**Note what `check_speedup_substantiated`'s PASS does and does not mean here.**
It passed *without measuring anything* — `check.py:213-217` returns early on
`mock: true`. Its `weak` verdict carries no evidence about performance, and a
reader who counts it as one would be counting the mock's own honesty as a
measurement. The producing agent noticed this too and ran that validator's
procedure by hand as a fifth protocol pass; that is
`results/raw_measurements/validator_sim_*.json`, not a validator output.

### The agent did better than the mock asked for

Two things worth keeping when this becomes a few-shot exemplar:

1. **It made the mock an A/A null control** rather than a placeholder: the seed
   measured against an identical copy of itself, 4 protocol passes, 20 fresh
   processes, and reported the resulting **±0.62% noise floor**. That is a real,
   reusable number for this machine, and it independently corroborates the
   workset's "under 1.05× is noise" guidance.
2. **It caught a baseline divergence and refused to bury it.** The workset's
   baseline (55.402 µs for the traced `B8_V151936` shape) was taken on
   gfx942/MI300X; on gfx950/MI355X the same driver measures **50.141 µs**, −9.5%,
   with the sign uniform across all three cases and their ordering preserved.
   The kit marks the cross-check `DIVERGED`, argues why it reads as a
   hardware/stack change rather than a broken oracle, and says plainly that it
   is *not* independently confirmed against a trace on this host.

   **Consequence for stage 4 on gfx950:** the workset ships a gfx942 baseline,
   so any speedup measured here must be taken against 50.141 µs. A future real
   campaign that compares its optimized kernel to the workset's printed 55.40 µs
   would book a free 9.5% it did not earn.

### One package-level gap this run did not exercise

`assets/optimize_kernel.task/readme.md` §4 hard-codes `--gpu-target gfx942` and
`--gpu-type mi300x` in the forge invocation it tells the agent to copy. On this
cluster the card is **gfx950 / MI355X**. Mock mode never runs that command, so
nothing failed — but the first real campaign on gfx950 will either be told the
wrong architecture or will have to notice the mismatch itself. It should become
a variable, or at least a sentence telling the agent to read the target off the
workset's `environment.md`. **Not changed in this round** — reported, not fixed.

## 12. Delivered

`/shared_nfs/yihou/agent_sys/debugging/kernel-opt/`

```
PROVENANCE.md            mock stated in the first line; what is measured vs synthetic;
                         the run; the verdicts; every reduction
kernel_optimization.v1/  THE DELIVERABLE — content/ + manifest.yaml + validation.yaml
                         sha256 8ad847057776153ac7486657af56a41b0cb09b7e29c1230f1f43d0be4c8a8d83
workset.v0/              the internal handoff too, since it was free
run_mock3.log            the run report the verdicts above are quoted from
run_pkg.sh               the driver, verbatim
```

Reusable bootstrap for a fresh node:
`/shared_nfs/yihou/agent_sys_debug/ws2/kernel-opt-bootstrap.sh <jobid> <gpu>`
(`IMAGE=` overridable; ~2 minutes; ends with a GPU launch probe so it fails
before a model call rather than after one).

## 13. The committable fix, and the run that proves it (commit `f478f16`)

§8's workaround — move `--demo-root` to local disk — breaks the standing
instruction that workspace/playground/handoff live on `/shared_nfs`. The fix
belongs in the package, and it takes **two different mechanisms** because the
two places that inherit `TMPDIR` are not the same kind of place.

**Producer side — an agent spec's `env` may set `TMPDIR`, and this is written
down.** `env_mgr/harness.py:39-44` lists `TMPDIR` in `_RESERVED` (never
forwarded from the operator's ambient environment) and then says: *"An agent
spec's declared `env` may still set any of them: an author saying so outranks a
default, and that update is applied after this module."* So
`kernel_opt_lead` now declares

```yaml
TMPDIR: '${scratch_root}/tmp'
```

`scratch_root` is already required and already documented as local disk, so this
adds no variable and states the fact once.

**Validator side — it does not work there, and the reason is structural.**
`validator/environment.py:233` sets `TMPDIR` to `<zone>/tmp` unconditionally,
and `CHANNELS` at `:86` lists `("tmp", "TMPDIR points inside the zone")` as an
**invariant of the zone**, not a default. No agent spec reaches it.
`check_speedup_substantiated/check.py` did
`mkdtemp(dir=os.environ.get("TMPDIR") or None)` and then ran `measure_baseline.py`
under it — so on a `/shared_nfs` run root it would have segfaulted **after** the
copies and the first measurement rounds, the most expensive and least legible
place to lose a run. It now takes the directory from a package arg:

```yaml
scratch_dir: '${scratch_root}/verify_scratch'
```

and the body prints which directory it chose, so a wrong value is visible in the
transcript instead of arriving as a signal. Setting it empty falls back to
`TMPDIR` for a site where the two are one filesystem.

**A syntax trap found while writing this.** The obvious form
`'${verify_scratch_dir:-${scratch_root}/verify_scratch}'` is a **load error**,
not a fallback: `spec_loader/variables.py:87` matches a default as `[^}]*`, so a
default may not contain `}`. Same family as the handoff seal's `@NAME@`
restriction the BRIEF warns about. If you want both a knob and a computed
default, you need two entries, not a nested one.

### Proof, not assertion — run `20260902T093218-5b0ebd`

Re-ran the whole package with `--demo-root` back on
`/shared_nfs/yihou/agent_sys_debug/ws2/runroot/kernel-opt-mock4` (mode 777),
package from a **private** clone at
`/shared_nfs/yihou/agent_sys_debug/ws2/kernel-opt/pkgrepo` at `f478f16`.

Read out of the live agent's `/proc/<pid>/environ`:

```
TMPDIR=/mnt/m2m_nobackup/yihou/kernel-opt/scratch/tmp          <- the override won
CLAUDE_CODE_TMPDIR=/shared_nfs/.../zones/task.../tmp           <- zone, untouched
HIP_VISIBLE_DEVICES=4
TRITON_CACHE_DIR=/mnt/m2m_nobackup/yihou/kernel-opt/scratch/triton_cache
```

Outcome: **no segfault**, correctness identical to the local-disk run
(SNR 128.74 dB, allclose true on all three shapes), and

```
final main / optimize_kernel / publish_workset: succeeded
check_optimization_shape: PASS   check_speedup_substantiated: PASS
check_workset_shape: PASS        0 validation(s) dropped
```

**Cost of the sanctioned layout: ~25 minutes against ~13.5.** The zone is on
NFS, so every file the agent writes pays for it. That is a real tax, it is not a
fault, and it should be expected rather than diagnosed.

Note what this run does **not** prove: under `mock=1`,
`check_speedup_substantiated` returns early and never reaches its `mkdtemp`, so
the `scratch_dir` argument is **exercised only by construction, not by
execution**. A real (non-mock) campaign on a `/shared_nfs` run root is what
would test it. Said plainly rather than left for someone to assume.

### One more artefact difference worth knowing

In this run the agent prepended a 12-line `MOCK RUN — THIS IS NOT AN OPTIMIZED
KERNEL` banner to `results/optimized_kernel.py`. The 41 code lines below it are
byte-identical to the seed — I checked line by line — but **`md5sum` of the two
files now differs**, where in the earlier run they matched. `verification.json`
still says "byte-identical copy", which is true of the copy that was *measured*
and not of the file on disk. Both readings are honest; only one survives a
digest check. If this becomes a few-shot exemplar, say which is meant.

## 14. Correction to §8, from `profiling`: the rule is narrower than I wrote

`profiling` ran a full 7-task graph — two engine bring-ups, two AIPerf replays,
four torch-profiler captures, thousands of ROCm kernel launches — with
`--demo-root` **on `/shared_nfs`** and saw **no segfaults**. That is not a
counter-example; it is the missing half of the rule, and their diagnosis is
right:

> **`TMPDIR` on NFS kills ROCm kernel launches only for processes that run
> inside the agent_sys zone.**

`profiling-demo` never launches a kernel in a zone. Every kernel it causes runs
inside a docker container the zone *starts*, and that container has its own
`/tmp`; the zone's `TMPDIR` never reaches the process that touches the GPU.
`kernel-opt-demo` is the opposite shape — the agent runs `driver.py` and
`measure_baseline.py` **in the zone**, on the host's python — which is exactly
where the fault bites.

So **do not read §8 as "keep your run root off `/shared_nfs`"**. That would push
people off the layout the user has mandated, for a fault most packages cannot
hit. The accurate advice, and the two halves that look inconsistent until you
know why:

| what | where | why |
|---|---|---|
| `--demo-root` (workspace, playground, handoffs) | **`/shared_nfs`**, mode 777 | the user's standing instruction; it is also how "remote" works here, since every node sees the same bytes |
| `TMPDIR` for a zone-launched GPU process | **node-local disk** | on NFS the first ROCm kernel launch is SIGSEGV, rc 139 |

They differ because they are answering different questions: the first is about
*where the artefacts live*, the second about *where a ROCm process may scribble*.
Only a package of this shape needs the second, and it now carries it in its own
spec rather than in an operator's command line.

## 15. The gfx942 baseline, and the 9.5% nobody would have caught

**The defect.** The shipped workset was traced and baselined on **gfx942 /
MI300X, 304 CU**: `baseline_measurement.md` prints **55.402 µs** for the
production `B8_V151936` shape and `environment.md` names the card. This cluster
is **gfx950 / MI355X, 256 CU**, and the same driver, same shapes, same dtype,
same correctness bars measures **50.141 µs** (run `…446cad`) and **50.181 µs**
(run `…5b0ebd`) — two independent runs, 0.08% apart, **9.5% faster** than the
workset's figure.

**Why it matters more than it looks.** An agent that optimizes on gfx950 and
divides by the workset's printed 55.402 µs books **9.5% of speedup it did not
earn**. Nothing in the report looks wrong: the number is real, the workset is
real, the arithmetic is right. `check_speedup_substantiated` would not catch it
either — it re-measures *both* sides on this host, so it substantiates the ratio
the handoff claims, not the denominator the handoff chose. It is a failure that
survives every gate this package has.

**A second copy of the same mistake, worse for being copy-paste.**
`assets/optimize_kernel.task/readme.md` §4 hard-coded
`--gpu-target gfx942 --gpu-type mi300x` into the command block it tells the
agent to run. On this cluster that compiles for an architecture nobody is
measuring, and forge does it without complaint.

**Fixed in `45721e6`**, both halves:

- `gpu_target` / `gpu_type` are now package variables on `kernel_opt_lead`'s
  `env`, defaulting to **empty**, and §4 defines empty as *detect it*:
  `torch.cuda.get_device_properties(0).gcnArchName`. A variable default cannot
  run `rocminfo`, so the honest default is no answer rather than one host's
  answer — and §4 says to stop rather than guess if detection fails.
  `--var gpu_target=gfx942 --var gpu_type=mi300x` restores the old behaviour for
  anyone outside this cluster who depended on it.
- §3 gains the case itself, with the rule stated as a rule: **your baseline is
  the number you measured on this host, never the number printed in the
  workset.** It also tells the agent how to tell a hardware shift from a broken
  oracle — a hardware shift is uniform in sign across every case and preserves
  the ordering; a broken oracle is not — because the pre-existing paragraph said
  "if they disagree by more than a few percent, stop", and stopping is the wrong
  response to a legitimate 9.5% hardware delta.

**Not fixed, deliberately.** The workset's own `environment.md` and
`baseline_measurement.md` still say gfx942/55.402 µs. They are a **true record
of where that baseline was taken** and rewriting them would destroy the only
evidence the cross-check works against. The fix belongs in what the consumer is
told to do with them, which is where it now is.

**Not proven by a run.** Under `mock=1` forge is never invoked, so
`KFO_GPU_TARGET` is never read. `agent-sys show` confirms the package still
loads with the new variables (3 tasks, same validator counts), and both runs
independently produced the `baseline_cross_check: DIVERGED` object the §3 text
now requires — but the detection path itself is **unexercised**, and a real
campaign is what would exercise it. Said plainly rather than left to be assumed.

## 16. Do the delivered handoffs work as fixtures? **As delivered, no.** Four findings

The leader's question: can a downstream stage consume the delivered handoffs
without the producing run? I tested rather than reasoned — the harness is
`/tmp/yihou_fixture_probe/{probe,dig,loc,loc2}.py` and every claim below is a
command's output.

### F1 — the delivered layout is not a store layout, and nothing can read it

`analyze-demo/assets/lib/store.py` is the only code in this tree written to read
a published handoff. It needs `<root>/<hid>/v<N>/` with `manifest.yaml` inside,
duplicating `handoff.store.version_dir` and `MANIFEST_FILE`. Pointed at the
deliveries as they stand:

| store root | hid tried | `versions()` | `kind_of()` | `content_dir()` |
|---|---|---|---|---|
| `debugging/kernel-opt` | `kernel_optimization.v1` | `[]` | `''` | `None` |
| `debugging/kernel-opt` | the real handoff uuid | `[]` | `''` | `None` |
| `debugging/profiling` | `kernel_table` | `[]` | `''` | `None` |
| relaid out as `<root>/<uuid>/v1` | the real uuid | `[1]` | `'kernel_optimization'` | found |

**Mine is recoverable by pure relayout** — one `mkdir` and one `mv`, no content
change. **`profiling`'s is not**: they delivered the `content/` payload only, so
there is no `manifest.yaml`, no `claim/`, no `validation.yaml`, and a version
without a manifest is by definition unpublished. The digest and the verdicts are
simply absent.

Root cause is the shared BRIEF: *"Copy the sealed content directory out of your
run's store"*. `profiling` read that as `content/` and I read it as the version
directory. Neither reading yields a store. **The instruction should have been:
copy `<hid>/v<N>/` and keep the `<hid>` level.**

### F2 — `chmod -R 777` breaks the digest, and consumption *verifies* the digest

`handoff.digest.tree_digest` is git-shaped and records the executable bit per
file: `100755` if `S_IXUSR` else `100644`. So `chmod -R 777` over a delivered
handoff changes every file's mode and therefore the tree digest.

**I did this to my own delivery** — over-reading the user's "mode 777"
instruction, which is about directories, as `-R` over a sealed artefact.
Measured before repair: all three handoffs recomputed to a digest their own
manifest did not record. The untouched originals in the run store verified
`MATCH=True`, which is what identified the copy as the damaged party.

This is not cosmetic. `handoff.store.copy_out` **recomputes and compares on
every consumption** and raises `DigestMismatch` — its own docstring anticipates
exactly this mistake: *"an agent copying with the defaults would produce a
different digest and no obvious cause"*. A mode-damaged fixture does not read
oddly; it **fails the consuming task**.

Repaired by re-copying from the run store with `cp -a` and then
`find <dir> -type d -exec chmod 777 {} +` — **directories only**. All four
handoffs now verify. Directory permissions are safe to change: a directory
enters the digest as the constant `040000` plus its entries, so 777 on a
directory costs nothing.

**Every module that was told to `chmod 777` its delivery should re-verify.**

### F3 — the locality rule is **not enforced at publication**

My delivered `kernel_optimization.v1` contains, in
`results/logs/run_measurements.log`, four lines naming
`/mnt/m2m_nobackup/yihou/kernel-opt/scratch/work/...` — a path that exists only
on the producing node. It sealed without complaint.

It sealed because **nothing calls the check**. `grep -rn locality agent_sys/`
outside `handoff/locality.py` finds only `analyze-demo`'s offline helper
`check_locality.py` and two comments. No module in `handoff/` imports it, so
`handoff.locality.check` is a rule the tree *documents* and does not *apply*.
Related, and not the same: `temp/bugs/002-handoff-dependencies-never-reach-
locality-check.md` is about the dependency list not reaching the check; this is
about the check not running at all.

Consequence for the fixtures: a node-only path can be anywhere in any of them,
and no gate has looked. Mine is in a log, so it does not break reproduction —
`REPRODUCE.md` uses `$PACKUP` and `$KFO_SCRATCH_ROOT` throughout — but the
guarantee people think they have here does not exist.

### F4 — a locality false positive worth knowing before anyone wires the check in

Run over my packup, the check's first hit is `REPRODUCE.md:48`:

```sh
cp "$PACKUP"/scripts/kernel/*.py "$W/baseline/"
```

flagged as the absolute path `/scripts/kernel/`. It is not one — it is a
relative path under a variable. The scanner sees `/` immediately after a closing
quote and starts a fresh absolute-path candidate, which is the **same
composition failure** as the seal's `@NAME@` rule and the variable grammar's
`[^}]*` default: this system does not compose across `}` or `"` before a `/`.
If the check is ever wired in, `"$VAR"/subdir` will fail honest handoffs.

### F5 — NFS attribute caching will lie to you about a fixture

After repairing the modes from the compute node, the login node still reported
the **old** modes and therefore a digest mismatch, for about a minute. The
compute node reported `MATCH=True` for the same path at the same moment. Anyone
verifying a fixture from a different host than the one that wrote it should
re-stat before believing a mismatch.

### What I did not check

Whether `env_mgr`'s input staging reads the store through `handoff.store` or by
its own path composition — I read the consumption API (`copy_out`) and the
package-level reader (`store.py`), not `env_mgr`'s call site. If staging
composes its own paths, F1's relayout may not be sufficient, and that is the
next thing to look at.

## 17. Closing §16's open question — and correcting F2 in the process

§16 ended by saying I had not checked whether `env_mgr`'s input staging goes
through `handoff.store` or composes its own paths. I have now. Two answers, one
reassuring and one that **corrects what §16 claimed**.

### The layout: F1's relayout is sufficient

`env_mgr/fs/layout.py:142` composes it itself —

```python
def handoff_version_dir(store_root, handoff_id, version):
    return os.path.join(store_root, str(handoff_id), f"v{version}")
```

— which is the *same* `<root>/<hid>/v<N>` as `handoff.store.version_dir` and as
`analyze-demo/assets/lib/store.py`. Three independent spellings of one fact, all
currently agreeing. So the relayout in §16's F1 is exactly what staging wants,
and a relaid-out fixture is consumable. `stage()` then copies **`content/` only**
(`narrow=True`, `layout.py:95`) and maps it so the artefact's files land *at* the
staged path with no `content/` level below it — deliberately, so a body never
sees the producer's `manifest.yaml`, `validation.yaml` or `claim/`.

### The digest: **staging does not verify, and §16 said it did**

§16's F2 said a mode-damaged fixture "fails the consuming task" with
`DigestMismatch`. **That is true of only one of the two routes, and it is not the
route input staging takes.**

| route | verifies? |
|---|---|
| `handoff.store.copy_out(hid, version, dst)` | **yes** — recomputes `tree_digest` and raises `DigestMismatch`. Used by in-process code |
| `env_mgr.fs.layout.stage()` → `env_mgr.fs.layout.copy_out()` (`layout.py:327`) | **no** — a plain `shutil.copytree`, no digest anywhere |

The two functions have the **same name** and different guarantees, and
`env_mgr/grants.py:177` asserts *"`copy_out` verifies the digest before
returning"* while `env_mgr/fs/layout.py`'s own `stage()` docstring says the
opposite in as many words: *"`handoff.get_manifest` verifies a digest where a
staged copy does not"*. Both are correct about different functions; a reader who
takes the `grants.py` sentence as being about the staging path — which is what I
did — concludes the wrong thing.

**So the corrected consequence of a `chmod -R 777` is worse, not better.** A
mode-damaged handoff does not fail loudly at staging. It **stages silently** and
the consuming task runs against it, with the manifest's digest now describing an
artefact that no longer exists and nothing in the consumption path looking. The
grants docstring's argument for why a body need not check its own input —
*"integrity arrives as content the body can trust"* — does not hold on this
route.

That does not change what a producer must do; it raises the stakes. **Verify the
digest yourself before delivering a fixture, because nothing downstream will.**
The check is four lines:

```python
import yaml; from handoff import digest
m = yaml.safe_load((v/"manifest.yaml").read_text())
assert digest.tree_digest(str(v/"content").encode()).hex() == m["digest"]["sha256"]
```

### What I still have not checked

Whether anything at all calls `handoff.store.copy_out` on the consumption side
in a normal run — i.e. whether the verifying route is exercised in practice or
is only available. `grants.py:172-175` says `agent/gate.py:91` is the sole
`get_manifest` caller outside `handoff`, and that it runs in the supervisor
*after* the executor returns, which suggests verification is not on the path a
consumed input takes. I have not confirmed that by reading `gate.py`.

## 18. Settled by experiment: staging does **not** verify, and a damaged fixture passes silently

§17 argued this from a code read. Because four modules were re-copying on the
opposite guidance, I ran it instead. `/tmp/yihou_fixture_probe/stagetest.py`:
build a correctly-laid-out store from the delivered handoff, call
`env_mgr.fs.layout.stage()` on it, damage every file's mode exactly as
`chmod -R 777` would, call `stage()` again.

```
A. pristine copy in a correctly-laid-out store
   digest matches manifest = True
   stage() -> <root>/into/a/ea05f78e-.../v1   (no exception)

B. after chmod 777 on every file
   digest matches manifest = False
   stage() -> <root>/into/b/ea05f78e-.../v1
   *** stage() RAISED NOTHING. The damaged fixture staged silently. ***
   staged tree digest = 9768a9f2d2a8c713...
   manifest  records  = e6ae4461ee32b7c2...
```

**A also confirms the relayout is correct** — `stage()` accepts
`<root>/<hid>/v<N>/` and returns the staged path. F1's instruction stands.

**B is the finding.** The framework staged an artefact whose content no longer
matches the digest its own manifest records, returned a path, and said nothing.
`env_mgr/fs/layout.py` imports no `copy_out` from anywhere — there is no
`handoff` import in the module at all — so the `copy_out(src, dst)` at
`stage()`'s line 99 is the plain `shutil.copytree` defined at line 327 of the
same file.

### Why two careful readers both concluded the opposite

`env_mgr/grants.py:172-183` says, in a docstring about read grants, that
*"`copy_out` verifies the digest before returning, so integrity arrives as
content the body can trust rather than a manifest it must check"* — and uses
that to justify not granting a body its input's `manifest.yaml`. The sentence is
**true of `handoff.store.copy_out`** and **false of the `copy_out` that staging
actually calls**. Two functions, one name, one module apart, opposite
guarantees. I believed it first; the leader believed it independently from the
same docstring. That is the shape of the trap, and it is worth more than the
fact itself: **a docstring asserting another module's guarantee is a claim, not
a mechanism, and it goes stale silently.**

`layout.py`'s own `stage()` docstring says the opposite and is correct:
*"`handoff.get_manifest` verifies a digest where a staged copy does not"*.

### What follows

- The justification for withholding `manifest.yaml` from a consuming body —
  that integrity has already been established — **does not hold on the staging
  route**. A body gets `content/` it cannot check, from a copy nothing checked.
- **Verify before you deliver**, because no one downstream will:

  ```python
  import yaml; from handoff import digest
  m = yaml.safe_load((v/"manifest.yaml").read_text())
  assert digest.tree_digest(str(v/"content").encode()).hex() == m["digest"]["sha256"]
  ```

- This is `agent_sys`'s rather than any package's, so it belongs in
  `temp/bugs/` if anyone wants it fixed rather than documented. I have not
  written that note — it is not my module's, and the leader is routing.

## 19. The relayout, applied — and what "applied" had to mean

§16's F1 diagnosed the delivered layout as unreadable and put the repair
commands in `PROVENANCE.md`. The leader's patrol caught that the repair had been
**documented and not performed**, which was fair: mine is the delivery the other
four are pointed at.

Applied now. `/shared_nfs/yihou/agent_sys/debugging/kernel-opt/` gains a
`store/` that **is** a store root:

```
store/ea05f78e-5c18-4798-8f5f-eafb980fc0e2/v1/   kernel_optimization
store/cd5440dc-0163-407e-9c0e-ed6d06f4bec8/v0/   workset
```

Two choices worth recording.

**`mv`, not `cp`.** Copying would have left the flat `<kind>.<v>/` shape beside
the correct one, and a directory offering two readings of the same artefact is
how the wrong one gets picked. There is now exactly one live shape.

**A `store/` level rather than uuid directories at the top of the delivery.**
The leader asked for `<uuid>/v<N>/` and this satisfies that shape exactly; the
extra level answers a question the flat form leaves open — *which path is the
store root?* With uuids beside `PROVENANCE.md` and the logs, the answer is "the
delivery directory, and please ignore the non-uuid entries". With `store/` it is
one unambiguous path to export. Flag it if the other four should match.

**Verified three ways rather than declared** (`/tmp/yihou_fixture_probe/final.py`):

| check | result |
|---|---|
| `tree_digest(content)` vs each `manifest.yaml` | matches, both |
| same vs the untouched run-store originals | byte-identical, both |
| `analyze-demo/assets/lib/store.py` | `versions=[1]`/`[0]`, kinds correct, content found |
| `env_mgr.fs.layout.stage()` — the real staging path | staged 2/2, 29 and 14 entries, no exception |

The last row is the one that matters: it is the same function §18 showed will
stage damage silently, so its success here says the layout is right and says
**nothing** about integrity. The digest rows are what say that, and they had to
be run separately for exactly that reason.

`superseded/` is deliberately left in the old flat layout — it is a reference
copy, not a fixture, and relaying it out would suggest otherwise.

## 20. Closing §18: the verifying route is the **producer** side

§18 left one thing unverified — whether `handoff.store.copy_out`, the route that
*does* check the digest, is called anywhere in a normal run. The team lead read
`gate.py` and answered it: it is called at **`agent/gate.py:226`**, and
`gate.py:211` explains why it always runs — *"`copy_out` is the only way to
learn the item keys"*. That is the **output gate**, checking a handoff on its
way to being sealed.

So the complete picture, and it is the worse of the two possible ones:

> **A handoff's digest is verified when it is produced and never when it is
> consumed.**

Damage introduced *after* sealing — a recursive chmod, a copy that resets modes,
an edit — passes silently through `env_mgr.fs.layout.stage()` and into the
consuming task. Recorded upstream by the lead as
`temp/bugs/2026-09-02-two-copy_out-functions-and-a-docstring-that-promises-the-wrong-ones-guarantee.md`
(`dd59cf0`), which argues the cheap fix is renaming one of the two functions:
two `copy_out`s one level apart in the same import graph with opposite safety
properties is a trap independent of any docstring.

### The self-check is therefore part of delivering, not part of auditing

Run against the delivered store and recorded in `PROVENANCE.md` and in
`digest-selfcheck.txt` beside it:

```
workset              cd5440dc-…/v0   VERIFIED True   d5f76738…
kernel_optimization  ea05f78e-…/v1   VERIFIED True   e6ae4461…
checked 2026-09-02T10:15:10Z from crs-m2m-cpu-spur-012
```

Both recomputed digests equal the value in their own manifest. Since nothing
downstream will repeat this, that line is the only integrity assurance a
consumer of these fixtures gets, and it belongs beside the artefact rather than
in a debug note.

## 21. The exec bits are original to the seal — because my chmod got there first

The lead's verification noticed executable `.md` files inside the delivered
workset and reasoned, correctly from the evidence they had, that since the
digests match the untouched originals the bits must be original and therefore
harmless. The first half is right. The second is not, and the difference is
worth recording because it makes F2 bigger than I first stated it.

**Measured:**

| | executable files |
|---|---|
| git (`git ls-files -s .../assets/worksets/`) | **0** — all nine are `100644` |
| `superseded/workset.mock3.v0`, run from the **clean shared clone** | **0** of 12 |
| delivered `workset` (mock4), run from **my private clone** | **9**, including five `.md` |

The private clone is the one I had `chmod -R 777`'d at creation, and it was
`run_pkg.sh`'s `--package` source for the mock4 run. So the chain is:
recursive chmod on the package source → package data staged with the exec bit →
`publish_workset` copies it into the handoff → **the seal records `100755` and
the digest covers it**.

**That is the part worth generalising.** F2 said a recursive chmod on a
*delivered* handoff breaks its digest — loud, once you check. This is the
quieter form: a recursive chmod on a *package source*, before the run, produces
a handoff whose digest is **perfectly valid** and whose content is subtly not
what the repository says. It verifies forever, and at that point the damage is
indistinguishable from intent. The only thing that caught it here was a
differential against a run from a clean clone.

Nothing is functionally wrong — they are documents, an exec bit does not change
how they are read, every validator and both digests agree — so this is a
provenance blemish rather than a defect, and I have not re-run to launder it.
It is stated in `PROVENANCE.md` beside the artefact.

**Practical rule, and it is the one I would give someone starting tomorrow:**
never `chmod -R` anything a run will read or write. Directories only
(`find <dir> -type d -exec chmod 777 {} +`), and if a git tree needs it, set
`core.fileMode false` first so the flip does not silently become a commit —
which is the other way this same mistake bit me today.

## 22. `relayout_handoffs.py` — the delivery step, scripted

I did my two handoffs by hand. Ten is where a uuid gets reused from an earlier
run, so the step is now a tool:
`/shared_nfs/yihou/agent_sys/temp/kernel-opt/relayout_handoffs.py`

```
python3 relayout_handoffs.py <run>/handoffs <delivery>/store [--apply] [--repo <agent_sys>]
```

Dry run by default. Every rule in it is a mistake made today: the uuid and the
version are **read from each `manifest.yaml`'s own directory**, never typed
(`v0` is common and is not a typo for `v1`); a version without a manifest is
skipped as unpublished rather than delivered empty; modes are preserved and only
**directories** are chmod-ed; and both the source *and* the copy are verified
against the manifest — the source first, because copying a damaged handoff
faithfully just launders the damage into the delivery.

Tested four ways against my own run: dry run lists 2/2; `--apply` copies and
re-verifies 2/2; a source with one file's mode flipped is reported
`SOURCE FAILS ITS OWN MANIFEST — not copied` and the tool exits 1 while still
delivering the good handoff beside it; an existing destination is refused rather
than merged. Non-zero exit means it can gate a delivery.

## 23. Delivering `integration`'s nine — and the check I did not do

Ran the copy side of `integration`'s recovery on the lead's authorisation
(their run was OOM-killed at 12:44; they were inactive). Nine sealed handoffs
from run `20260902T104817-8a2995` into
`/shared_nfs/yihou/agent_sys/debugging/integration/store/`, all at **`v0`**.

The tool held up on its first real use — uuids and versions read from each
manifest, sources verified before copying, 9/9 both ways:

- **integrity**: every copy re-hashed against its own manifest *and* against the
  untouched run-store original — 9/9 on both;
- **layout**: `analyze-demo`'s reader 9/9, `stage()` staged 9/9 — and labelled
  as proving nothing about integrity, since it is the function §18 showed will
  stage damage silently.

One expectation corrected by measurement: the lead expected a tenth directory
for the unsealed `integration_packup`, to be skipped as unpublished. **There is
no tenth directory** — the run's `handoffs/` holds exactly nine, so nothing was
skipped.

### What I missed, and it was the important thing

I verified **integrity** and **shape** and reported "9/9 verified" — and never
opened a single `validation.yaml`. `deploy` did, and found that
`integration_report` carries `check_no_regression: result=False`, `strength:
strong`. I then read all nine myself: **8 pass, 1 fails**, and the failure is on
the terminal report of the measurement chain. That verdict is what blocks
`compare`'s consumer and is why `integration_packup` never ran — the missing
tenth is a consequence of the verdict, not of the OOM kill.

So my "9/9 verified" was true and misleading in the same breath. **A digest
proves the bytes have not changed since sealing; it says nothing about whether
what was sealed was acceptable.** Those are two independent questions and I
answered one while sounding like I had answered both. Recorded in
`store/COPIED-BY-KERNEL-OPT.md` with the per-handoff verdict table, so a reader
does not take the refused report for a passing sample.

**This is the same shape as everything else today** — the seal faithfully
records what it was told — but pointed at me rather than at the framework, and
it is the third time today the fix was to go and read the artefact I had been
summarising. Worth adding to the delivery checklist: **verify the digest, verify
the layout, and read the verdicts. Three checks, not two.**

**One correction to my own framing, from `deploy`.** I wrote that the digest
"says nothing about acceptability", which is true, and then let that stand as if
it made the check secondary. It does not. `deploy`'s diagnosis of that `false`
verdict *depended* on my 9/9: because the bytes were provably the sealed bytes,
they could treat the verdict as a property of the run rather than wondering
whether my copy had damaged something. **A check that rules out an entire class
of confusion is doing real work even when it is silent on the question you care
about.**

**And the reason neither of us had the habit is worth more than the rule.**
`deploy`: *"I only checked my own `validation.yaml` because the run report had
printed verdicts I wanted to reconcile against — on a delivery with no run
report I had no rule that would have made me open it."* Mine vanished for the
same reason. The habit was anchored to an artefact that exists only when you ran
the thing yourself, so it disappeared silently the moment the delivery was
second-hand — which is exactly the case where it matters. That is why it belongs
in the tool rather than in either of our heads.

`relayout_handoffs.py` now **prints each handoff's verdicts, at the dry run** —
before anything is written, which is when the information is worth having. Held
the edit until nobody was mid-delivery: `deploy` turned out never to have used
the tool at all, and `profiling` and `analyze` had finished. *Editing a tool
underneath someone is how a good day ends badly*, and asking cost one message.

**Two colleagues then wanted opposite things about the exit status, and both
were right.** I had argued a failing verdict should not affect it: a refused
handoff is still a legitimate artefact — `integration_report` is a good sample
*of a report that was refused* — so failing on it would be lying by omission in
the other direction. `profiling` argued the opposite and pointed at the case
that actually happened: *"the difference between something a reader might skim
past and something a caller can gate on — and `integration_report` is exactly a
set that would otherwise be reported as delivered."* A human did skim past it.
Me, in a message whose entire purpose was to say what state those artefacts were
in.

**Resolved with a distinct exit code rather than by picking a side**, because
collapsing the two conditions into one number throws away *which* problem you
have — which is the day's whole lesson applied to my own tool:

```
0  copied, intact, and every verdict passes
1  a source or a copy failed its digest   -> the delivery is BROKEN
2  usage: no source, or no published versions under it
3  copied and intact, but something is NOT VALID — a failing verdict,
   or no verdicts at all.  `--allow-refused` makes this 0.
```

A caller that just wants a sound delivery tests `!= 0`; one that has read the
refusal block and accepts it passes `--allow-refused`, which turns 3 into 0 and
nothing else — **it cannot hide a digest failure**, which exits 1 regardless
(tested). The refusal block prints either way, so the flag moves the exit code
and never the information.

**I removed that flag and then put it back, and the round trip is the useful
part.** `deploy` argued against gating at all: *"the first thing anyone would
add is `--force`, and within a day passing it would be reflex."* I found that
convincing and deleted the opt-out. The lead's reply answered it better than my
deletion did: *"a flag named for exactly what it permits addresses the objection
honestly instead of routing around it."* `--force` suppresses whatever went
wrong; `--allow-refused` can only mean *this set contains a refused handoff and
I accept that*, and a caller passing it has said which condition they accept.
The decay argument is real and is an argument about **naming**, not about
whether an opt-out should exist.

The rest of `deploy`'s case does survive, and one part of it is the best
argument in the thread:

- **A verdict must not share a code with an integrity failure**, because
  integrity and copy failures are facts about *this tool's own operation* while
  a verdict is a fact about *someone else's judgement* — including validators
  whose bars are provisional, as `check_no_regression`'s 5% placeholder is.
  Inheriting another module's provisional bar as a delivery gate would be wrong.
  That argument is why `1` and `3` are separate rather than merged.

The lead ruled for three states independently and asked for the codes in
`--help`; they are there, `argparse` prints the module docstring. Their ruling
numbered the third state `2`; it is **`3`** here only because `2` was already
usage, which I have told them.

**A cost of the round trip worth naming, because it is today's own failure mode
committed by me.** Between removing the flag and restoring it, both the lead and
`profiling` wrote `--allow-refused` into their records — the lead into the BRIEF,
`profiling` into their notes at `fdeb5d6` — describing a flag that at that moment
did not exist. Two modules' documentation was true when written, false an hour
later, and neither would have found out. **Removing something you have already
told people about is a documentation change, not just a code change**, and I
should have said so at the time instead of discovering it from their replies.

`profiling`'s second suggestion is in too, and it is the subtler one:
**"no verdicts recorded" now prints differently from "all passed"**, and
distinguishes *no `validation.yaml`* from *a `validation.yaml` with an empty
list*. An unvalidated handoff is not a passing one, and reading it as one is the
same error as reading a digest as an acceptability check.

All four exit paths tested: clean set 0, `integration`'s refused set 3,
`--allow-refused` 0 with the block still printed, mode-damaged source 1, missing
source 2.
