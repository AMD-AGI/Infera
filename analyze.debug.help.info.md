# analyze-demo — localisation and debug notes

Written while driving `agent_sys/examples/llm_e2e_performance_optimization/analyze-demo`
to one valid `analyze_packup` handoff on the **spur** cluster
(`crsuse2-m2m-080`, job `101078`, GPUs 2–3), 2026-09-02.

The package was authored on a different cluster. Every section below is a place
where "the package as written" and "this cluster" disagreed.

---

## 0. Environment, measured

| | |
|---|---|
| node | `crsuse2-m2m-080`, reached as `spur exec 101078 bash -lc '...'` |
| my GPUs | **2 and 3 only** — deploy-demo holds 0–1, kernel-opt-demo holds 4 |
| my ports | 8140–8159 (this package binds none — no server is started) |
| node-local scratch | `/mnt/m2m_nobackup/yihou/analyze` (28 T, 23 T free) |
| python on the node | `/usr/bin/python3` = **3.12.3** — clears the package's 3.12 floor |
| `HOME` inside `spur exec` | `/opt/spur`, not `/home/yihou` |

### Image

`lmsysorg/sglang:v0.5.18-rocm720-mi35x` — the package default for `AD_IMAGE` —
**is not on this node and was not pulled.** What is here and was used instead:

```
harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix
```

Verified inside it, on GPUs 2–3, before anything else was attempted:

```
/sgl-workspace/aiter                    exists
/sgl-workspace/sglang/python/sglang     exists
torch 2.9.1+rocm7.2.0   torch.cuda.device_count() == 2   (HIP_VISIBLE_DEVICES=2,3 honoured)
triton 3.6.0            import aiter -> ok               import tilelang -> ok
```

So the whole of what `verify_workset` needs is in the stock serving image, as
the README claims — no image build, no model weights.

`--group-add video` (the flag the package's `verify.py` hard-codes) **works
here**; the numeric `--group-add 44 --group-add 992` workaround recorded in the
repo `CLAUDE.md` for other images was not needed.

---

## 1. `srun` does not exist on this cluster — `verify_workset` cannot reach the GPU as written

**What happened.** `assets/verify_workset.task/verify.py` reaches the GPU node
with

```python
argv = ["srun", f"--jobid={jobid}", "--overlap", "-N1", "-n1", "-w", node,
        "--chdir=/tmp", "bash", "-lc", inner]
```

in both `run_remote()` and `stage()`. Measured inside `spur exec 101078`:

```
command -v srun   -> NO_SRUN
command -v sinfo  -> NO_SINFO
```

Slurm client binaries are not in the `spur exec` namespace at all. Every GPU
call would have died with `FileNotFoundError: 'srun'`.

**Why it is not just a missing binary.** The package's own README says
"`identify` runs where `agent-sys` runs, `verify_workset` runs on the GPU node,
and the two need not be the same machine" — `srun` is how it crosses that gap.
On this cluster the gap does not exist: `agent-sys` is itself started by
`spur exec` **on** `crsuse2-m2m-080`, so the orchestrator and the GPU are the
same machine and the docker call can be made directly.

**Confirmed, not assumed:** `hostname` inside `spur exec 101078` returns
`crsuse2-m2m-080`, which is the node the allocation holds, and the probe
container above ran GPUs from that same shell.

**The fix.** `verify.py` now selects its transport from a package variable
instead of hard-coding one. `assets/verify_workset.task/verify.py`:

```python
TRANSPORTS = ("srun", "spur", "local")

def _launcher(timeout_hint: str) -> list[str]:
    """The argv prefix that puts `bash -lc <...>` on the GPU node."""
```

`srun` stays the default so the originating cluster is unaffected; `spur` is
`spur exec <jobid> bash -lc` for a site that has the scheduler but not the
orchestrator on the node; `local` is `bash -lc`, for this one. Both call sites
— `run_remote()` and `stage()`, which had the `srun` argv duplicated — now go
through it. Declared in `shared.yaml` as
`AD_NODE_TRANSPORT: '${node_transport:-srun}'`.

The same edit added `AD_VISIBLE_DEVICES: '${visible_devices:-}'`, which becomes
`-e HIP_VISIBLE_DEVICES=...` on the `docker run`. Without it a driver opens card
0, which on this node belongs to deploy-demo. An empty value means "the whole
node", which is the right default and the wrong behaviour the moment the node is
shared — so it is a variable, per the repo's own rule that every identifier
bound on a shared host is a parameter.

**Measured after the fix**, with a throwaway one-operator workset whose driver
is a bf16 matmul against an fp32 reference
(`/shared_nfs/yihou/agent_sys_debug/ws2/analyze/transport_probe.sh`):

```
verify_workset: 1/1 passed, 1 ran
  PASS probe_op: 0.0926 ms, rsd 0.0038
check_locality: clean
real 0m35.148s
```

**35 seconds for one operator's whole protocol** — one correctness run plus five
timed groups, six container launches. A cold `docker run` of this image doing
`import torch, aiter` and a GPU op costs 10–13 s. So `verify_workset` is not a
budget item on this cluster, which is worth knowing because the package's README
treats it as the thing that eats the settle budget.

---

## 2. The seed CSV does not exist and had to be synthesised

`AD_SEED_CSV` defaults to
`/apps/tas/yaoc/.../gap_analysis/gap_analysis.csv`. **`/apps` does not exist on
this cluster.** No Magpie gap-analysis CSV was found under `/shared_nfs`.

**The seed used in this run is synthetic.** It is not a measurement of anything.
See `PROVENANCE.md` beside the delivered handoff, and section 3 below for how it
was built and against what specification.

---

### How it was built, and against what

`assets/lib/csv_io.py` is the specification and was read before a byte was
written. The generated table satisfies, by construction and then by
measurement:

| rule | source | required | got |
|---|---|---|---|
| six base columns present | `csv_io.BASE_COLUMNS` | all six | all six, `enriched: False` |
| data rows | `check_kernel_table` `min_kernel_rows` | ≥ 20 | **34** |
| `% Total` sum | `check_kernel_table` `pct_total_min/max` | 50.0 – 100.5 | **99.90** |
| unclassified share | `check_worklist_shape` `max_unknown_ratio` | ≤ 0.3 | **0.088** (3 of 34) |
| selectable candidates | `rank.select` after both floors and a shape check | ≥ `top_n` | **16** |

Bucket mix, from `assets/lib/taxonomy.py` over the generated names:
`collective` 3 rows / 63.21 %, `routable` 19 / 29.37 %, `vendor_tuned` 4 /
4.87 %, `framework_native` 5 / 2.22 %, `unknown` 3 / 0.23 %. The
collective-dominant shape is deliberate — it is what makes `rank`'s
classify-before-sort step do visible work, which is the behaviour the package's
DESIGN.md argues for.

Three rows exist only to exercise the exclusion reasons that are otherwise
never reached: `_hadamard_transform_kernel` at 0.05 % (`below_pct_floor`),
`_copy_blocks_kernel` at 50 calls (`below_calls_floor`), and
`_reshape_and_cache_kernel` with an empty `Input Shapes` cell
(`no_shape_evidence`).

Two mistakes worth not repeating if you regenerate it:

- **Quote the `Name` column.** ATen symbols contain commas inside their
  template arguments (`at::detail::Array<char*, 3>`) and an unquoted row shifts
  every later field. Write the file with `csv.writer`, not with a heredoc.
- **The sum overshoots easily.** A first draft summed to 105.17 and would have
  failed `pct_total_max: 100.5`. The percentages are the thing to check, not the
  microsecond columns, which nothing validates.

### It reproduces the package's own documented result

Running the first three leaves directly against it (the README's "call the body
directly" route, scripted as
`/shared_nfs/yihou/agent_sys_debug/ws2/analyze/offline_probe.sh`) selects the
same operators the package's README reports measuring on the real GLM-5.2
profile, and `identify` resolves them to the same three files:

```
#1 6.80% moe_gemm   ck-fellow      mfma_moe1_silu_mul_afp4_wfp4_bf16_g1u1
#2 5.40% moe_gemm   ck-fellow      mfma_moe2_afp4_wfp4_bf16_g1u1
#3 3.90% gemm       triton-fellow  _gemm_a16_w16_kernel

moe_gemm_mfma_moe1_silu_mul   -> aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py
moe_gemm_mfma_moe2_afp4_wfp4  -> aiter/ops/flydsl/kernels/moe_gemm_2stage.py
gemm_gemm_a16_w16             -> aiter/ops/triton/gemm/basic/gemm_a16w16.py
```

**`identify` reached resolve_ratio 1.0 here, against 0.0 on the originating
cluster.** Not because anything improved: `check_identity_resolved`'s
`min_resolve_ratio` is 0.0 with a long comment explaining that Triton and
TileLang symbols have no source file to find, and the three selected here happen
to be symbols whose names *do* grep to a definition in the extracted checkouts.
Do not read 1.0 as a quality signal.

One genuine inconsistency in the package, found in passing and **not fixed**
(out of scope): for `gemm_gemm_a16_w16`, `identify` reports
`image_repo_path: ${SGLANG_ROOT}` and `repo_subdir: sglang` while
`source_file_path` is `aiter/ops/triton/gemm/basic/gemm_a16w16.py` — the owner
was guessed from the `^_gemm_` pattern in `identify.OWNER_PATTERNS` (which maps
to sglang) while the file was actually found in the *aiter* checkout. The
validator does not catch it, because it only requires the path to be relative
and the root to be a placeholder. A consumer resolving that path under
`${SGLANG_ROOT}` will not find the file.

### Magpie's kernel finder is absent, and that is a supported outcome

`identify.kernel_finder` wants
`$AD_MAGPIE_ROOT/Magpie/tools/amd_kernel_finder/finder.py`. Neither
`/shared_nfs/chaox/Magpie` nor `/shared_nfs/age/hl/Magpie` contains
`tools/amd_kernel_finder`. The function is written to survive this — it returns
`{"available": False, ...}` and resolution falls back to `symbol_search` — so
`--var magpie_root=/nonexistent` is passed explicitly rather than leaving the
`/apps/...` default, so the log says what is true.

---

## 3. `preciousObjects` cannot be set on a worktree without hitting every other worktree

`agent-sys run` refuses to start without
`extensions.preciousObjects=true` on the repository, because
`env_mgr.workspace.cut` needs it. Its own message says the rest:

> Note: in a git worktree this lands in the SHARED common config, so it affects
> the main checkout and every other worktree, and `git gc` will refuse in all of
> them until it is unset.

The worktree `/home/yihou/dev/git/infera.aiopt.real.task_package` does **not**
have it set, and setting it would change state outside this module's remit.

**What to do instead:** run from the standalone clone
`/shared_nfs/yihou/agent_sys_debug/repo`, which already has it, and whose
`origin` *is* the worktree. So the loop is: commit in the worktree, then
`git -C /shared_nfs/yihou/agent_sys_debug/repo pull --ff-only origin dev.yihou.aiopt.task_package`.

Note that this splits where the code comes from, and it does not matter:
`agent-sys` is an editable install pointing at the **worktree**
(`python -c "import cli.main; print(cli.main.__file__)"` →
`/home/yihou/dev/git/.../agent_sys/cli/main.py`), so the *framework* is always
the worktree's, while `--package` resolves against the clone. Both were at the
same commit, so this is a distinction with no consequence — but it would matter
if you edited `agent_sys/` itself.

---

## 4. The cards are CPX-partitioned, so "GPU 2–3" is not HIP device 2 and 3

**This nearly cost another team their run, and the note is the point of this
file.**

`crsuse2-m2m-080` reports eight MI355X, but the compute partition is CPX. Inside
the sglang image:

```
torch.cuda.device_count() -> 64
every device: 36.0 GiB
get_device_properties(i).uuid identical within each group of eight
```

So HIP devices 0–7 are physical card 0, 8–15 card 1, 16–23 card 2, 24–31 card 3,
32–39 card 4. The three modules sharing this node were assigned "GPUs 0–1",
"GPUs 2–3" and "GPU 4" — **physical cards**, and deploy-demo read it that way and
put its TP4 group on HIP devices 0,1,2,3.

I read "GPUs 2–3" literally and launched with `HIP_VISIBLE_DEVICES=2,3`, which is
two partitions of **deploy's card 0**. Caught before `verify_workset` reached the
GPU. The correct value for analyze-demo is HIP **16–31**.

If you are placing more than one tenant on a CPX node, write down HIP device
ranges, not card numbers. A card number is ambiguous by a factor of eight.

---

## 5. Job 101078 was cancelled mid-run

At 08:56 UTC, with `build_workset` eleven minutes into a real run,
`spur exec 101078` began returning
`job 101078 is not running (state: CANCELLED)`. Not cancelled by this module,
and the node was shared with deploy-demo and kernel-opt-demo, both of which lost
work too.

What survived, because it was written to `/shared_nfs` rather than to node
scratch: the seed CSV, the launcher, the committed code fix, and the run store
(`/shared_nfs/yihou/agent_sys_debug/ws2/runroot/analyze-dry2/`) with
`seed_table`, `rank` and `identify` all `succeeded`. What did not: the 7.2 GB of
extracted checkouts on `/mnt/m2m_nobackup`, which re-extract in 17 seconds.

The lesson is the placement, not the cancellation: **put `--demo-root` on
`/shared_nfs` and only bulk, regenerable data on node scratch.** The BRIEF says
this; it is worth restating because losing the node is what tests it.

---

## 6. Where each leaf stands, and how each was proven

Recorded here because the node died before one contiguous run existed, and
"which leaf has actually been seen to work" is the question a later reader will
ask first.

| leaf | proven how | result |
|---|---|---|
| `seed_table` | real `agent-sys run` `20260902T084427-888c06` | `succeeded`, 34 kernels |
| `rank` | same run | `succeeded`, buckets as above, 2 selected |
| `identify` | same run | `succeeded`, 3/3 resolved offline, 2/2 in-run |
| `build_workset` | same run, killed at 11 min | **one complete operator directory**, all 11 files, second one started |
| `verify_workset` | standalone, `transport_probe.sh` | 1/1 PASS, 0.0926 ms, rsd 0.0038, 35 s |
| `pack_analyze` | standalone, `packup_probe.sh` | validator PASS, locality clean |

The probes are in `/shared_nfs/yihou/agent_sys_debug/ws2/analyze/`:
`offline_probe.sh` (seed/rank/identify), `transport_probe.sh` (verify with a
throwaway driver), `packup_probe.sh` (the terminal leaf, fed from the other
two plus the killed run's partial workset).

### `build_workset` costs about ten minutes per operator

Measured: eleven minutes of wall clock produced one operator's complete
directory —

```
moe_gemm_mfma_moe1_silu_mul/{README.md,forge_task.yaml,invocation_spec.json,
  program.md,provenance.json,run_forge.sh,reference/naive_torch.py,
  scripts/{forge_driver,standalone_driver,task_runner}.py,tests/cases.json}
```

— and had begun the second. It obeys the brief's "finish one operator
completely before starting the next", which is what makes an interrupted run
salvageable at all.

So the whole graph at `top_n=2` should be roughly: seed+rank+identify ≈ 10 s,
`build_workset` ≈ 20 min, `verify_workset` ≈ 70 s for two operators,
`pack_analyze` ≈ 5 s. **About 22 minutes.** With the settle ceiling now at four
hours (section 7), nothing here is near a budget.

### `top_n` was set to 2, and the reasoning is not "smaller is cheaper"

`check_workset_runs` has `min_pass_ratio: 0.5`, so `top_n=2` needs one of two
operators to measure cleanly and `top_n=3` needs two of three. The first two
selected are both FlyDSL MoE GEMMs — the hardest thing in the candidate pool for
an agent to write a reference against — and the third is a plain Triton GEMM
that should pass easily. So `top_n=3` needs *the same* "at least one MoE works"
plus the GEMM, and `top_n=2` needs only the MoE condition. **2 is not merely
cheaper, it is the weaker requirement.** Do not raise it thinking you are
buying margin.

---

## 7. The 1800 s settle budget no longer exists

Both `analyze-demo/README.md` ("Known limits — the settle bound is a real limit
for the whole graph", citing `cli/main.py:790`) and the shared BRIEF describe a
ceiling that has since been raised. Read first-hand in `agent_sys/cli/main.py`:

```python
_SETTLE_TIMEOUT = 14400.0            # line 902 — four hours
run.add_argument("--timeout", ..., default=_SETTLE_TIMEOUT)   # line 166
_settle(registry, stream, timeout=getattr(args, "timeout", None) or _SETTLE_TIMEOUT)  # line 405
```

with a comment at the constant explaining exactly why it moved: "every time the
unit of work grew … the ceiling became an execution budget for the one case it
was never meant to bound, and it did so by reporting a healthy run as a hang".

The 20 s `stall_after` branch still ends a genuinely stuck run quickly, so the
generous ceiling costs a finishing run nothing. **`analyze-demo/README.md`'s
"Known limits" section should be corrected**; it is left alone here because
editing it is outside this round's scope, and this file is the record.

What is *not* disproven: the other half of `temp/bugs/003`, that an interrupted
task leaves its output slot open so `--resume` cannot finish it. I did not
re-test that.

---

## 8. Run `analyze-r1`: five of six sealed, and the sixth was lost in a 20-second window

Run `20260902T091144-096985`, node `crsuse2-m2m-019`, `top_n=2`. It got further
than anything before it and then failed in a way worth writing down carefully,
because the failure is **not** the one the package's README warns about.

### What passed

```
verdict  check_kernel_table:        PASS   usability       / strong
verdict  check_worklist_shape:      PASS   completeness    / strong
verdict  check_identity_resolved:   PASS   trustworthiness / strong
verdict  check_workset_shape:       PASS   completeness    / strong
verdict  check_workset_runs:        PASS   trustworthiness / strong
handoff  analyze_packup slot v0:    generating          <-- never sealed
```

`build_workset` took **1419 s (23.7 min), 67 turns, $13.79** and its validator
passed. `verify_workset` measured **both** operators cleanly — confirmed
afterwards by running `pack_analyze`'s body over the same sealed handoffs, which
printed `packup: 2 operators, 2 measured`.

### What failed, and what it was not

The run ended with

```
done  run complete; this package promises no failure, and the run did NOT
      finish: main: running, pack_analyze: running, 0 validation(s) dropped
```

**It was not the gate's executable rule.** That is the failure the package's
README and the shared BRIEF both point at, and it is the first thing I checked:

```
$ ls -la .../818786a1-.../v0/content/items/command
-rwxr-xr-x 1 yihou ubuntu 286 Sep  2 09:38 .../items/command
```

`packup.py:122` does `command.chmod(0o755)` and it took. Mode 0755. Not it.

**It was not a slow body.** Timed against the very same sealed handoffs:

```
packup: 2 operators, 2 measured
real  0m0.444s
check_locality: clean
real  0m0.151s
```

**It was not an incomplete output.** The content it wrote is whole — 36 files —
and passes its own validator when `check_analyze_packup_shape.check()` is
applied to it directly with the real args: `PASS — 4 mandated file(s) present
with substance`.

### What it was

The timeline, from zone and file mtimes:

| time | what |
|---|---|
| 09:38:41 | `verify_workset: succeeded`; `pack_analyze: input_validating -> running`. **Last state change.** |
| 09:38:41–49 | zone prepare: `workspace/` cut at :47, `config`/`handoffs`/`tmp` at :49. Four input handoffs staged, 28 files for `operator_workset` |
| ~09:38:50 | body runs, writes 36 files |
| 09:39:00 | run ends. `_settle`'s `stall_after` is **20.0 s** and 09:38:41 + 20 = 09:39:01 |

So the run was ended by the stall branch **exactly 20 s after the last state
change**, with the body's output written and the seal not yet taken. Eight of
those twenty seconds went to zone prepare before the body was even started.

`_settle`'s own docstring describes the mechanism: the loop ends "when no task's
status or attempt count has moved for `stall_after` and no attempt holds a
thread", and a task "parked rather than working" does not count as holding. The
console names it: `main is waiting on a decision no one will make — the
escalation reached the top and this entry point installs a sink that records and
does not answer (nothing to push: the attempt holds no executor: it is not in
its main phase)`.

**So the effective budget for the terminal task is not the four-hour
`--timeout`; it is twenty seconds from the previous task's completion to the
terminal handoff being sealed** — prepare, body and seal all inside it. There is
no CLI flag for `stall_after`; unlike `timeout` it is only a keyword default of
`_settle` (`cli/main.py:905-912`).

This is worth separating from `temp/bugs/003`. 003 says *a gate failure is
reported as a timeout*. This is *a task that did nothing wrong is reported as a
timeout*, and the distinguishing evidence is that the output is complete, valid
and correctly permissioned.

### `--resume` cannot repair it, exactly as `temp/bugs/003` says

```
final  pack_analyze: failed — HandoffStateError:
       818786a1-64b6-4f9d-a96f-627bbb3f085f v0 is already open by
       task 73c1d79b-d299-47e7-86e5-24535d755861
```

The interrupted task left its output slot open and the resumed task cannot
claim it. Confirmed first-hand; that half of 003 still holds. The only route to
a sealed terminal handoff is a fresh run.

### Run 2 reproduced it exactly, and then the cause turned out not to be the clock

Run `20260902T094842-4dc2da`, same settings, same five PASS, same
`analyze_packup slot v0: generating`. Deterministic, not a race. Timeline:

| 10:13:27 | `verify_workset: succeeded` — last state change |
| 10:13:35 | `pack_analyze` zone created |
| 10:13:36.1 | body **finished** writing all 37 files |
| 10:13:48 | run ended (10:13:27 + 20 s) |

The body finished **eleven seconds inside the window** and the seal still never
happened. In neither run was a `validation.<pack_analyze>.output_validation`
zone created, while every other task has one. So output validation never
started — the twenty seconds is how the run *ended*, not why the handoff was
lost.

**The filesystem is not the cause either, and I measured rather than assumed.**
Copying the 37-file / 229 K content tree, three runs each:

```
nfs   0.31 s   nfs   0.21 s   nfs   0.22 s        (/shared_nfs)
local 0.01 s   local 0.01 s   local 0.01 s        (/mnt/m2m_nobackup)
```

A 0.2 s gap cannot spend eleven seconds. I had expected NFS to be the answer.
It is not, and moving `--demo-root` to node-local storage will not fix this.

### The actual cause: the output version was never pinned, so nothing sealed

Following the escalation text back through the framework:

1. `monitor/pusher.py:147` emits `nothing to push: {why}` from **`_decide_gate`**
   — so a `GATE_KIND` event fired. The gate refused.
2. `agent/gate.py:_one_output` produces `OUTPUT_ABSENT` in two ways:
   `declared output {hid} was never delivered`, or `exists with no version`.
3. `agent/runner.py:_seal_outputs` (called **before** the gate) is what publishes
   a program task's output, and it has two branches that **skip silently**:
   `store is None`, and `versions.get(hid) is None` — *"Nothing was pinned for
   this output, so there is no directory to seal. The gate reports the absence."*

`cli/README.md:1117` describes the resulting symptom in the same words as what I
observed: *"`exit 0`, a complete artefact in the granted path, no
`manifest.yaml`, gate says `output_absent`"*. And on disk, that is exactly it:

```
818786a1-.../v0/content/   37 files, complete, validator PASS
818786a1-.../v0/claim/     empty
818786a1-.../v0/manifest.yaml   absent
```

against a sealed sibling which has a manifest carrying `kind: operator_workset`.

**Seal refusal is ruled out too.** I ran the framework's own
`handoff.locality.check` — not the package's offline `check_locality.py`, the
real one — over both the lost `analyze_packup` content and the
`operator_workset` content that sealed fine:

```
run2 analyze_packup (never sealed): locality OK
run2 operator_workset (SEALED OK):  locality OK
```

So the content was acceptable; the version simply was never pinned.

### And the reason the version was never pinned is in THIS package

`_seal_outputs` calls `store.seal(hid, version, producer=…)`, and
`FilesystemStore.seal` runs two admission checks before writing the manifest:
`readme.check` and `content.check_items` against **the kind's own
`items_schema`**. Run both directly against the real unsealed content and the
answer is immediate:

```
readme.check: OK
check_items REFUSED: items $: Additional properties are not allowed
  ('REPRODUCE.md', 'environment.md', 'notes.md', 'results' were unexpected)
  (the kind's items_schema; a key it never declared needs
   additionalProperties to permit it)
```

`analyze_packup`'s `items_schema` declared six items —
`result, env, command, code, logs, watchout` — with `additionalProperties:
false`. `packup.py` writes **ten**. And the four undeclared ones are exactly the
ones `check_analyze_packup_shape` **requires**: `required_files` names
REPRODUCE.md, environment.md and notes.md, `required_dirs` names results.

**The producer, the validator and the kind had drifted apart, and the kind was
the one nobody ran.** Five of the six handoffs in this package seal fine, so
nothing exercised the mismatch until the terminal one.

`seal` returns the refusal **as a string, not an exception** — deliberately, so
that `agent` need not import `handoff` — and `_seal_outputs` puts it in a
`refused` dict under `seal_refused`, which `agent/runner.py`'s own docstring
says "has no reader outside these tests yet". So the reason exists, is correct,
is specific, and is discarded. What the operator sees instead is a task stuck in
`running` and a twenty-second timeout.

**Fixed in `acb8bfe`** by declaring the four items. Verified against the real
unsealed content of run 2: `check_items` refuses under the old schema and
accepts under the new one.

### How to find this class of fault in one command

If a terminal handoff never seals and the run reports the last task as
`running`, do not start with the clock or the executable bit. Ask the seal:

```python
from handoff import content as c
c.check_items(c.load(Path(content_dir)), c.content_type("reproducible"), items_schema)
```

Any handoff whose producer writes more items than its kind declares is exposed,
and `additionalProperties: false` is the house style in these packages. The
framework's own `handoff.locality.check` is worth running the same way — it is
the real rule, and `assets/lib/check_locality.py` is only an offline
approximation of it. Here both agreed (clean), which is what ruled locality out.

### What a later reader should do

- Do not shrink the graph to fit the settle budget — that budget is four hours
  now (section 7). Shrink instead so that **the terminal task's prepare + body +
  seal fits in twenty seconds**. Prepare grows with the number and size of the
  declared inputs, and `pack_analyze` declares four, the most in the package.
- A cheap lever if it bites again: lower `top_n`. It halves the operator
  directories the terminal handoff copies and the seal scans, and it halves
  `build_workset`. It does not touch the eight seconds of zone prepare.
- Check `items/command`'s mode first anyway. It is the documented failure and
  ruling it out takes one `ls`.

---

## 9. `max_rsd` makes a busy node fatal, and it is a hard fail rather than a note

Run `analyze-r5` on `crsuse2-m2m-267`, with the schema fix in place, got past the
seal problem and failed somewhere new:

```
check_workset_shape:  PASS
workset_evidence slot v0: invalid
check_workset_runs:   FAIL
```

The evidence document says both operators were fine:

```
summary: {'operators': 2, 'ran': 2, 'passed': 2, 'pass_ratio': 1.0}
  moe_gemm_mfma_moe1_silu_mul   ran True correct True snr 99.49
  moe_gemm_mfma_moe2_afp4_wfp4  ran True correct True snr 46.92
```

The failure is the spread:

```
moe1: per_group_ms [0.1344, 0.1437, 0.1315, 0.1424, 0.1272]  rsd 0.047   ok
moe2: per_group_ms [0.1772, 0.4358, 0.1876, 0.1842, 0.1965]  rsd 0.423   FAIL
```

One group at 0.4358 ms among four near 0.18. `rocm-smi` on that node:

```
GPU[0..7]: GPU use (%): 100        VRAM allocated 60-84%
```

**Every card was saturated by another tenant.** The outlier is real contention,
and `max_rsd: 0.1` is doing exactly the job its comment claims — "the machine was
not quiet, and a baseline measured then is not one forge-loop can optimize
against".

**Two things about this rule that are easy to get wrong.**

**It is a hard `return False`, not a per-operator note.** Read
`check_workset_runs/check.py`: `ran` false and `correct` false both `continue`
into `notes` and are then forgiven by `min_pass_ratio`, but an rsd breach
returns immediately. So `min_pass_ratio: 0.5` does **not** protect you from
noise: one noisy operator fails the whole step however many others were clean.

**Which inverts the `top_n` advice in section 8.** Two competing risks:

| risk | `top_n=1` | `top_n=2` |
|---|---|---|
| the agent writes one bad driver | fatal | survivable (`min_pass_ratio` 0.5) |
| one operator hits node noise | one chance to be unlucky | **two** chances, either fatal |

So there is no universally right value. **On a quiet node prefer 2** (driver risk
dominates, and noise is not a factor). **On a busy shared node neither value is
good** — go and find a quiet node instead, because no `top_n` makes a
100%-utilised machine produce a stable baseline.

**Check the node before you measure, not after:**

```sh
spur exec <job> bash -lc 'rocm-smi --showuse | grep "GPU use"'
```

Eight zeros is what you want. It takes two seconds and would have saved a
25-minute run.

### The driver failure worth knowing about separately

Run `analyze-r3` used `top_n=1` and failed for an unrelated reason — the agent's
driver for the FP4 MoE GEMM did `torch.zeros(..., dtype=<fp4>)`:

```
NotImplementedError: "fill_cuda" not implemented for 'Float4_e2m1fn_x2'
```

torch has no fill kernel for `Float4_e2m1fn_x2` on this build. The same operator
was driven correctly in two other runs, so this is generation variance in
`build_workset`, not a property of the operator — but it is the concrete reason
`top_n=1` is a gamble.

---

## 10. The green run

`20260902T120735-131786`, node `crsuse2-m2m-050`, commit `acb8bfe`, `top_n=2`,
27 minutes. **Six of six handoffs valid, six of six validators PASS**, and
`main: succeeded`:

```
final  seed_table / rank / identify / build_workset / verify_workset /
       pack_analyze / main: succeeded

check_kernel_table          PASS  usability       / strong
check_worklist_shape        PASS  completeness    / strong
check_identity_resolved     PASS  trustworthiness / strong
check_workset_shape         PASS  completeness    / strong
check_workset_runs          PASS  trustworthiness / strong
check_analyze_packup_shape  PASS  completeness    / strong

done  run complete; this package promises no failure, so nothing here was
      tested for one, 0 validation(s) dropped
```

Measured, on an idle node:

```
moe_gemm_mfma_moe1_silu_mul    SNR 99.49 dB   0.1304 ms   rsd 0.0124
moe_gemm_mfma_moe2_afp4_wfp4   SNR 47.03 dB   0.0588 ms   rsd 0.0260
```

The sealed handoff is at
`/shared_nfs/yihou/agent_sys/debugging/analyze/analyze_packup/`, with its
manifest, its recorded verdict, the run console and the synthetic seed CSV
beside it, and a `PROVENANCE.md` that leads with the synthetic-seed caveat.

**No validator threshold was weakened anywhere.** The two commits are
`4011eb7` (transport and visible cards become variables) and `acb8bfe` (the
`items_schema` fix). Everything else was environment.

### What it cost, and the short version for the next person

Six runs and about five hours, of which the debugging was maybe ninety minutes
and the rest was hardware. Four allocations died under this work
(080 cancelled then drained, 019, 260, plus 080's replacement). If you are
picking this up:

1. **Get a quiet node first.** `rocm-smi --showuse | grep "GPU use"` — eight
   zeros. `sinfo -h -p amd-spur -o "%t %N" | awk '$1=="idle"'` finds one, and
   `-q amd-burst-qos -w <node>` will place you on it when your team QOS is at
   its `QOSGrpNodeLimit`.
2. **Probe the node before trusting it.** `crsuse2-m2m-037` came back RUNNING
   with no `/shared_nfs` and no GPUs. And do *not* test with `ls /dev/dri/` —
   it is empty inside `spur exec` on perfectly good nodes; count devices from
   inside a container.
3. **Put `--demo-root` on `/shared_nfs`.** Node-local is faster and buys
   nothing here — the whole run is 27 minutes and the filesystem is not the
   bottleneck (section 8) — and it is what let two runs' worth of evidence
   survive a node being cancelled.
4. Everything else is in sections 1–9.

---

## 11. The locality check is disconnected on purpose — settled first-hand

`kernel-opt` found a handoff that sealed while carrying
`/mnt/m2m_nobackup/yihou/...` in a log, and asked whether
`handoff/protocols.py:294` — *"Runs the README check and the locality check
**before** anything is created"* — has a call site narrower than the docstring
implies. **It has no call site at all, and that is deliberate.**

`handoff/store.py` imports `locality` at `:39` and uses it only to build
`Oracles` at `:140`. Both publication points then skip it explicitly.
`seal()` at `:447`:

```python
# locality.check — NOT CALLED. User-ruled 2026-08-31; ROADMAP §6.4.
```

and `put()` at `:494`, with the reasoning:

```python
# **`locality.check` is not called, and criterion 17 is therefore not
# enforced.** User-ruled 2026-08-31 after it refused a correct artefact:
# the shape heuristic read an HTTP access-log line as a filesystem path,
# and the brief that produced the artefact *required* that line. Measured
# 97% false positive on a real kit. `ROADMAP.md` §6.4 carries the rebuild
# at P2, and `handoff/locality.py` is kept intact and tested — this is a
# disconnected caller, not a deleted module, so re-wiring it is one line.
```

So: the module is live and tested, the callers are cut, and **the Protocol
docstring is stale.** That docstring is the only thing in the tree still
claiming the check runs, and it is what sent two of us looking for a narrower
call site. Worth a one-line correction by whoever owns `handoff/`.

This is **not** `temp/bugs/002`, which is about a kind's `dependencies` never
reaching `Oracles.image_prefixes`. 002 assumes the check runs and is
under-informed. This is the check not running.

**A consequence for everything else in this package.** Several of these packages
— this one included — tell their agents "the seal will refuse an absolute path"
and hand them `check_locality.py` to check against. That instruction is now
*advice*, not a mechanism: nothing enforces it at publication. Keeping the advice
is still right — a handoff naming a host path is wrong on any other machine, and
ROADMAP §6.4 will re-wire the check — but nobody should be told the seal will
catch it, because it will not.

### And the same false positive was in our own helper

The reason the framework's rule was disconnected is a 97% false-positive rate,
and `assets/lib/check_locality.py` reproduced it faithfully, because reproducing
it faithfully was the point. `kernel-opt` hit it on a real packup:

```
cp "$PACKUP"/scripts/kernel/*.py .        ->  flagged as absolute /scripts/kernel/
```

The candidate regex's lookbehind is `(?<![A-Za-z0-9._~@+-])` — word characters
and nothing else — so any path run following a **closing delimiter** matches:

```
<operator_id>/scripts/forge_driver.py     preceded by >     refused a whole workset
"$PACKUP"/scripts/kernel/*.py             preceded by "     kernel-opt's packup
${AITER_ROOT}/ops/triton/gemm.py          preceded by }
```

All three are relative paths under something the reader substitutes.
**This system does not compose across a `}` or a quote before `/`** — the same
break appears in the seal's `@NAME@` rule and in the variable grammar's
`[^}]*`. Three separate places now.

`check_locality.py` now classifies rather than merges: a match following one of
`>`, `}`, `"`, `'`, `)`, `]` is reported as a composition artefact and **does
not** set the exit code; a genuine absolute path outside the allow-list still
does. Both are printed, because a `${VAR}/a/b` a consumer cannot expand is a
real problem — just not this rule's. Tested against all three artefact shapes
(exit 0), against real host paths (exit 1, with `/opt/` and `/usr/` correctly
allowed), and against the delivered packup (clean).

The helper can no longer claim to predict the seal, because the seal does not
ask. Its docstring says so now.

---

## 12. Delivery

`/shared_nfs/yihou/agent_sys/debugging/analyze/`:

```
store/                      six sealed versions from run 20260902T120735-131786
store-run2-superseded/      five sealed versions from the earlier partial run,
                            laid out by `profiling` at 11:24 UTC under the team
                            lead's authorisation; kept so their record survives
PROVENANCE.md  run-console.log  seed-gap_analysis.SYNTHETIC.csv
```

**Two things were in that directory and are not any more, and the reason is the
same for both.** An unpacked copy of the terminal handoff's content sat beside
`store/`, and its `manifest.yaml` and `validation.yaml` sat loose next to it. A
second content tree can drift from the one the digest covers, and a manifest
without its content is exactly the shape that misleads a reader into thinking a
handoff is present when it is not. The content is now at
`store/bfbbfd45-…/v0/content` and only there.

The near-miss worth recording: I first *moved* those two loose YAMLs into
`store-run2-superseded/` rather than deleting them. They describe run **6**, so
filing them under a directory named for run 2 would have made a correct artefact
into a false one — worse than leaving them loose. Caught by checking their
`sha256` against the store's manifest before believing the directory name.

Verified two ways, kept separate because `env_mgr.fs.layout.stage()` accepts a
mode-damaged handoff without raising and so proves layout only:

- **integrity** — `tree_digest` per `content/` MATCHes both its own
  `manifest.yaml` and the untouched original in the run store, 6/6;
- **layout** — this package's `assets/lib/store.py` resolves
  `versions`/`kind_of`/`content_dir` for all six, and `stage()` stages all six.

Script: `/shared_nfs/yihou/agent_sys/temp/analyze/verify_delivery.py`. Two traps
it hit while being written, both worth knowing: `tree_digest` takes a **bytes**
path (`os.fsencode`) and a `str`/`Path` fails inside its own recursion with a
`TypeError` that reads like a library bug; and it returns **bytes**, so comparing
it to a manifest's hex string reports a mismatch on a perfectly good tree — I
briefly had six false mismatches from exactly that.

**Never `chmod -R` a delivery.** `handoff/digest.py:82` records each file's exec
bit and `handoff/store.py:284` recomputes on consumption, so a recursive chmod
becomes `DigestMismatch` at the consumer. Directories only.

---

*(end)*
