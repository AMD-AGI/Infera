# analyze-demo

Stage 3 of `temp/mission.md`: turn a kernel-level GPU profile into a ranked list
of operators worth optimizing, and build a complete KernelForge workset for each.

`DESIGN.md` is the reviewed design. This file is how to run it and what to
expect.

## The graph

```
main (non-leaf)
 ├── seed_table       program → kernel_table        froms: []                     mock input
 ├── rank             program → kernel_worklist     froms: [seed_table]
 ├── identify         program → operator_identity   froms: [rank]
 ├── build_workset    ai      → operator_workset    froms: [rank, identify]
 ├── verify_workset   program → workset_evidence    froms: [build_workset]        needs a GPU
 └── packup           program → analyze_packup      froms: [all four]  is_end
```

Six handoffs, six validators, one per kind — `validator/report.py` blocks a
handoff bound to none.

## Run it

```sh
AGENT_SYS_NO_PERMISSIONS=1 agent-sys run \
  --package agent_sys/examples/llm_e2e_performance_optimization/analyze-demo \
  --var jobid=28080 \
  --var gpu_node=smci355-ccs-aus-n05-21 \
  --var sglang_src=/data/agent_sys_analyze/repos/sglang \
  --var aiter_src=/data/agent_sys_analyze/repos/aiter \
  --var top_n=2
```

`jobid` and `gpu_node` carry no default: a Slurm allocation changes every time
and a node name is a fact about one site. Missing either is a load-time fault
naming the file, the line and the variable.

`top_n=2` is a workaround, not a preference — see *Known limits* below.

## Before the first run

**Python 3.12 or newer.** Below that `agent-sys` does not import at all;
`temp/bugs/001-requires-python-3.10-but-fails-below-3.12.md` has the detail.

```sh
python3.13 -m venv <venv>
<venv>/bin/pip install -e agent_sys
<venv>/bin/pip install -e "agent_sys[claude]"
```

**Source trees to search.** `identify` resolves a device symbol to the source
that declares it by searching indexed repositories. Extract them from the
serving image, so what is searched is what actually ran:

```sh
CID=$(docker create lmsysorg/sglang:v0.5.18-rocm720-mi35x true)
docker cp "$CID:/sgl-workspace/aiter" /data/agent_sys_analyze/repos/aiter
docker cp "$CID:/sgl-workspace/sglang/python/sglang" /data/agent_sys_analyze/repos/sglang
docker rm -f "$CID"
```

About 7 GB. Put it on node-local storage: `identify` runs where `agent-sys`
runs, `verify_workset` runs on the GPU node, and the two need not be the same
machine.

**No model weights, and no GLM-5.3 engine image.** `verify_workset` measures one
operator at a time, not a served model, and torch, triton, tilelang and aiter are
all in the sglang base image already.

## What each leaf does

| leaf | needs | produces |
|---|---|---|
| `seed_table` | a CSV on disk | `kernel_table` — mock of `profiling-demo`'s `kernel_scan` |
| `rank` | nothing | `kernel_worklist` — every kernel classified, the routable top-N selected |
| `identify` | indexed source trees | `operator_identity` — repo, language, files, entry points |
| `build_workset` | an Anthropic endpoint | `operator_workset` — spec, task, drivers, reference, cases |
| `verify_workset` | 8×MI355X via `srun --overlap` | `workset_evidence` — measured correctness and time |
| `packup` | nothing | `analyze_packup` — the deliverable |

## What this stage produces, and what it does not

**The measuring apparatus, not the thing measured.** The kernel source stays in
the framework; forge-loop edits it there, in a git worktree. What forge-loop
cannot make for itself is the driver that decides whether an edited kernel is
still correct and how much faster it got — and it treats that driver as a
protected file its agent may not modify.

So this stage writes `forge_driver.py`, `naive_torch.py`, `cases.json`,
`program.md`, `invocation_spec.json` and `forge_task.yaml`. It does not write
kernel source; it locates it. `DESIGN.md` section 5.4 has the field-by-field
version.

## Measured on this package

On the sample profile — `sglang_1p1d_glm5.2` decode, 143 kernels — with the
source trees above:

```
buckets: collective 3 (78.98%), routable 41 (13.74%), vendor_tuned 50 (5.11%),
         framework_native 38 (2.10%), unknown 11 (0.01%)

identify: 5/5 selected operators resolved to a source file
  moe_gemm_mfma_moe1_silu_mul   -> aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py
  moe_gemm_mfma_moe2_afp4_wfp4  -> aiter/ops/flydsl/kernels/moe_gemm_2stage.py
  gemm_gemm_a16_w16_m           -> aiter/ops/triton/gemm/basic/gemm_a16w16.py
  router_topk_transform_decode  -> kernels/aot/csrc/elementwise/topk.cu
  layernorm_add_rmsnorm_quant   -> csrc/kernels/rmsnorm_quant_kernels.cu

verify_workset on smci355-ccs-aus-n05-21 (2 operators):
  PASS moe_gemm_mfma_moe1_silu_mul   SNR 99.49 dB   0.1848 ms   rsd 0.0055
  PASS moe_gemm_mfma_moe2_afp4_wfp4  SNR 47.03 dB   0.1058 ms   rsd 0.0139
```

Collectives being 78.98% of GPU time is why `rank` classifies before it sorts.
A plain top-N by percentage would put a kernel first that forge-loop's
single-GPU driver contract cannot measure at all.

## Known limits

**A gate failure is reported as a timeout.** The completeness gate refuses a
seal for mechanical reasons — `agent/gate.py` requires `script`, `command` and
`entry` items to be executable — and the monitor answers with `continue, do it
until finished`, which does not name the reason. The agent then loops until the
hard-coded 1800 s settle bound stops the graph, and the report reads
`build_workset: running`. Measured: a complete workset was lost to a missing
`chmod +x`, and reducing `top_n` changed nothing. Recorded as
`temp/bugs/003-push-message-does-not-name-the-gate-failure.md`. This package sets
the bit in both producers and checks it in both validators.

**The settle bound is a real limit for the whole graph.** With the above fixed,
`build_workset` succeeds in 16 min 22 s, which leaves under 13 minutes of the
1800 s for `verify_workset` and `pack_analyze`. `cli/main.py:790` is a keyword
default that nothing overrides.

**Handoff content may not name an absolute path.** `handoff/locality.py` refuses
to seal anything outside a small allow-list, and `/sgl-workspace/` and `/data/`
are both outside it. Container roots therefore travel as `${AITER_ROOT}`-style
placeholders, with the expansion in `assets/lib/container_roots.yaml`. The
framework has a mechanism for this — `Oracles.image_prefixes`, fed by a kind's
`dependencies` — and nothing wires the two together; that is
`temp/bugs/002-handoff-dependencies-never-reach-locality-check.md`.

Run `assets/lib/check_locality.py <dir>` on anything before it is sealed. It
applies the same regexes and allow-list and names the file and line. The trap is
not only host paths: `<operator_id>/scripts/x.py` in prose reads as an absolute
`/scripts/x.py`, because the rule's lookbehind does not exclude `>`.

**The mock profile is GLM-5.2, not GLM-5.3-Flash.** The shapes are real and
exercise the pipeline correctly; the operator mix is not the target model's,
which serves DSA through TileLang, KDA through Triton and MoE through the Triton
runner. Every handoff that carries it says so.

## Running one leaf on its own

`cli/main.py:669` fixes the root closure's name to `main`, so there is no flag
for "run just this task". Two ways round it, both used during bring-up:

**Trim the subgraph.** Comment out the later entries in `main.yaml`. Good for
development, not for a deliverable.

**Call the body directly.** Each task body reads its inputs and output from
environment variables, so it runs outside the framework unchanged:

```sh
AGENT_SYS_DEMO_PACKAGE=<package> \
AGENT_SYS_INPUT_OPERATOR_WORKSET=<a sealed workset content dir> \
AGENT_SYS_OUTPUT_WORKSET_EVIDENCE=<an empty dir> \
AD_JOBID=28080 AD_GPU_NODE=smci355-ccs-aus-n05-21 \
AD_IMAGE=lmsysorg/sglang:v0.5.18-rocm720-mi35x \
AD_WORK_ROOT=/data/agent_sys_analyze AD_SNR_THRESHOLD=30.0 \
python <package>/assets/verify_workset.task/verify.py
```

The same shape works for `rank`, `identify` and `packup`. A validator needs
`inputs.json`, `args.json` and `materials.json` in its working directory, which
is what `ScriptBodyRunner` writes for it.

## Joining up with profiling-demo

`profiling-demo`'s `kernel_scan` produces the same `kernel_table` **kind name**,
and until the two packages are merged that is all it shares.

`spec_loader/registry.py` keys the registry on `name` alone — `version` is
maintenance metadata, not a second slot — so a name held by a spec that is not
byte-identical is `SpecInconsistent` at load time. Loading both packages into one
`BaseSpecRegistry` originally collided on five names. **Two of them were plain
name clashes and have been renamed away:**

| was | is now | why it was never a real conflict |
|---|---|---|
| `packup` (task) | `pack_analyze` | shares no input kind, output kind, agent or body with `profiling-demo`'s `packup` |
| `check_packup_shape` | `check_analyze_packup_shape` | reads `analyze_packup`, not `profile_packup`; different rules, different `check.py` |

Both packages had independently chosen the obvious word for their own last step.
Renaming is the whole fix, not a workaround.

**Three collisions remain, and renaming is the wrong answer to all three:**

| name | module | why it is not a rename |
|---|---|---|
| `kernel_table` | handoff | one kind declared twice. Renaming apart leaves two kinds where the join wants one |
| `check_kernel_table` | validator | follows whichever `kernel_table` survives |
| `main` | closure | `cli/main.py` fixes the root closure's name, so each package needs its own to run standalone |

Each resolves by deletion or by merging rather than by a new name, which is what
the join below does anyway. Verified: with the two renames in place, building the
merged package described below loads **37 specs with zero problems**.

That is the state of the *names*. The *content* is a separate question and is
already compatible.

The two producers lay the content out differently because their `content_type`s
differ, and neither choice is free to change:

| | `seed_table` (the mock here) | `kernel_scan` (the real producer) |
|---|---|---|
| `content_type` | `structured_text` | `reproducible` |
| document | `items/text.json` | `items/result/text.json` |
| raw CSV | `items/gap_analysis/` | `items/result/gap_analysis/` |

`reproducible` requires `result` and `env` and one of `script`/`command`, which
is what makes a real capture reproducible. `structured_text` permits only
`text.json` / `text.yaml` / `text.xml` / `schema` at the top level, so a handoff
carrying directories of trace-derived evidence cannot use it at all.

`assets/lib/kernel_table.py` reads either layout and is the only file that knows
there are two; `rank` and `check_kernel_table` both go through it. The records
inside are identical either way — same field names, `self_us` and not
`self_cuda_us` — and the real producer emits every row rather than a top-N,
which matters because `rank` buckets every kernel before it sorts.

So the remaining work is **one decision about which package owns `kernel_table`**,
and then assembly — not a translation layer. The steps below were run end to end
against a scratch copy, and the result loads with zero problems:

1. take `profiling-demo` as the base package and copy this package's
   `steps/*.yaml` and `assets/*` into it, minus `steps/seed_table.yaml` and
   `assets/seed_table.task/`;
2. drop this package's `kernel_table` handoff and `check_kernel_table` validator,
   keeping `profiling-demo`'s — the producer's contract is the real one, and its
   validator is the one that can check launcher coverage and the head's share.
   **This is the decision**, and it is why the join is written down rather than
   committed;
3. merge the two `main` closures into one root: `handoffs` is the union minus
   `kernel_table`, `subgraph` is `profiling-demo`'s five leaves plus `packup`,
   then `rank` with `froms: [kernel_scan]`, then this package's remaining four,
   with `is_end` moving to `pack_analyze`. Eleven leaves.

Three of this package's `assets/` entries are also the base package's and are not
copied over: `main.task/` and `check_kernel_table.validator/` belong to whichever
side wins step 2 and step 3, and `lib/store.py` exists in both and is
byte-identical either way (checked, not assumed).

### What arrives from the real producer that the mock cannot supply

Each kernel may carry a `launcher` block: the Python frame that launched it, as
`source_file` / `line` / `function` / `sample_count` / `launch_api` plus
`container_root`, `owner` and `path_form`. `identify` reads it as resolution
level 1, which is why level 1 was unreachable until now.

`source_file` is always relative and `path_form` says to what. torch strips the
longest matching `sys.path` entry from a frame path, so one capture yields both
shapes: `container_absolute` means the path is exactly relative to
`container_root`, while `sys_path_relative` means torch stripped an entry the
producer cannot identify. `identify.bind_launcher` settles it by testing
candidates against the indexed checkout — the consumer can stat a file and the
producer cannot, so the producer reports and this side confirms.

## Layout

```
analyze-demo/
├── DESIGN.md                 the reviewed design
├── README.md                 this file
├── main.yaml                 root closure + subgraph
├── shared.yaml               program agent, all package variables, kernel_table kind
├── steps/*.yaml              one file per step: kinds, validators, task
└── assets/
    ├── lib/                  shared by task bodies and validator bodies
    │   ├── csv_io.py             both Magpie CSV header shapes
    │   ├── taxonomy.py           classification, driven by kernel_taxonomy.yaml
    │   ├── kernel_taxonomy.yaml  the bucket / fellow / category / dtype rules
    │   ├── symbols.py            device symbol -> the source that declares it
    │   ├── shapes.py             Input Shapes -> forge-loop case selectors
    │   ├── forge_export.py       one record -> invocation_spec + forge_task
    │   ├── bench_stats.py        the 5-group weighted average, shared with its validator
    │   ├── check_locality.py     the seal's rule, runnable before the seal
    │   ├── container_roots.yaml  placeholder -> container path
    │   └── store.py              reading a published handoff (from demo)
    ├── <task>.task/          readme.md, entry.sh, body
    └── <validator>.validator/  readme.md, entry.sh, check.py
```
